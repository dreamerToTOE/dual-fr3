"""Recover a synchronized U4 clip from an externally interrupted Isaac run."""
import argparse
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
import zlib

import seven_dataset
import u4_command_y


STREAMS = ("contact_states/sensor_inputs.jsonl.gz", "contact_states/reference.jsonl.gz",
           "wrench.jsonl.gz")


def recover(run, isaac_dir, output):
    report_path = run / "runner_report.json"
    original = report_path.read_bytes()
    report = json.loads(original)
    if report["status"] != "RUNNING":
        raise ValueError("Expected an interrupted run with a provisional report")
    if output.exists() or output.name != run.name:
        raise ValueError("Output must be a new folder named after the recorded trial")

    original_hashes = {"runner_report.json": hashlib.sha256(original).hexdigest()}
    records = []
    tails = {}
    for name in STREAMS:
        raw = (run / name).read_bytes()
        original_hashes[name] = hashlib.sha256(raw).hexdigest()
        decoder = zlib.decompressobj(wbits=31)
        content = decoder.decompress(raw)
        if decoder.unused_data:
            raise ValueError(f"Unexpected additional gzip data in {name}")
        *lines, partial_last_line = content.split(b"\n")
        records.append(lines)
        tails[name] = {"gzip_trailer_present": decoder.eof, "complete_rows": len(lines),
                       "trailing_partial_line_bytes": len(partial_last_line)}

    count = min(map(len, records))
    if count <= report["simulation_clock"]["control_steps"]:
        raise ValueError("The common prefix does not extend past the provisional report")
    for tick, (sensor, reference, wrench) in enumerate(
            zip(*(islice(stream, count) for stream in records), strict=True), start=1):
        obs, ref, force = map(json.loads, (sensor, reference, wrench))
        if not obs["physics_step"] == ref["physics_step"] == force["physics_step"] == tick:
            raise ValueError(f"Noncontiguous physical step at {tick}")
        if not obs["time_s"] == ref["time_s"] == force["time_s"]:
            raise ValueError(f"Mismatched timestamps at step {tick}")
        for arm in ("left", "right"):
            if (obs[arm + "_wrench_world"] != force[arm]["tcp_gravity_corrected_world"]
                    or obs["tcp_positions_m"][arm] != force[arm]["tcp_position_m"]):
                raise ValueError(f"Sensor/wrench data disagree at step {tick}")

    ended = {"status": "EXTERNALLY_INTERRUPTED",
             "reason": "Process exited without final report or closed gzip streams"}
    receipt = {"trial": run.name, "status": ended["status"],
               "original_status": report["status"],
               "original_report_step": report["simulation_clock"]["control_steps"],
               "last_common_step": count, "stream_tails": tails,
               "source_sha256": original_hashes,
               "input_checkpoint_sha256": report["upper_row"]["input_checkpoint"]["sha256"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        scratch = Path(scratch)
        staged = scratch / run.name
        (staged / "contact_states").mkdir(parents=True)
        for name, lines in zip(STREAMS, records, strict=True):
            with gzip.open(staged / name, "wb") as stream:
                stream.writelines(line + b"\n" for line in islice(lines, count))
        (staged / "metadata.json").write_bytes((run / "metadata.json").read_bytes())
        # Pass clean, truncated copies to the standard extractor without changing the raw report.
        report["status"] = "STOPPED"
        report["simulation_clock"]["control_steps"] = count
        (staged / "runner_report.json").write_text(json.dumps(report))
        paired, commands = scratch / "paired.jsonl.gz", scratch / "right_commands.json.gz"
        seven_dataset.extract(staged, "u4", paired, allow_incomplete=True)
        u4_command_y.extract(staged, paired, isaac_dir, commands)
        ready = scratch / "ready"
        ready.mkdir()
        with gzip.open(paired, "rt") as source, gzip.open(ready / paired.name, "wt") as dest:
            header = json.loads(next(source))
            header["incomplete_run"] = ended
            header["source_sha256"] = {key: original_hashes[key] for key in header["source_sha256"]}
            dest.write(json.dumps(header, allow_nan=False, separators=(",", ":")) + "\n")
            copyfileobj(source, dest)
        with gzip.open(commands, "rt") as stream:
            targets = json.load(stream)
        targets["sensor_inputs_sha256"] = original_hashes[STREAMS[0]]
        targets["wrench_archive_sha256"] = original_hashes[STREAMS[2]]
        with gzip.open(ready / commands.name, "wt") as dest:
            json.dump(targets, dest, allow_nan=False, separators=(",", ":"))
        (ready / "recovery.json").write_text(json.dumps(receipt, indent=2) + "\n")
        ready.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "isaac_dir", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    recover(args.run, args.isaac_dir, args.output)
