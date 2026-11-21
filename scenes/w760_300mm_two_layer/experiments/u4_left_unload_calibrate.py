"""Trial-level fitting and leave-one-withdrawal-out diagnostics on the historical U4 captures."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from seven_reference import SevenReference
from u4_contact_observer import SIDE, U4ContactObserver
from u4_handover_score import load_at
from u4_left_unload_observer import LeftUnloadObserver


def extract(path):
    with gzip.open(path,"rt") as stream:
        header = json.loads(next(stream))
        marks = {row["phase"]: round(120*row["sim_time_s"])
                 for row in header["phase_times_for_scoring_only"] if row["status"] == "RUNNING"}
        if "upper_B_side_unload" not in marks:
            return None
        start = marks["upper_B_side_unload"]
        pulse_step = next((round(120*row["sim_time_s"]) for row in header["phase_times_for_scoring_only"]
                           if row["phase"]=="upper_B_u4_side_probe" and row["status"]=="STARTED"), None)
        passive = U4ContactObserver(header["config"])
        observer = LeftUnloadObserver()
        reference = SevenReference(header["config"], 120, header["reference_positions"])
        event = None
        truth = []
        for line in stream:
            sensor, phys = json.loads(line)
            tick = sensor["physics_step"]
            if tick < start-480:
                continue
            if tick > start+180:
                break
            est = passive.update(sensor,header["optical_neighbors"])
            out = observer.update(sensor,est)
            if out["unload_event"] and tick >= start:
                if event:
                    raise ValueError("Multiple releases in one calibration window")
                e = out["unload_evidence"]
                event = {"event_step":tick,"feature":[e["left_normal_recoil_um"],e["right_shear_N"]]}
            if event and tick >= event["event_step"]:
                assert tick == phys["physics_step"]
                label = reference.update(phys)
                truth.append(load_at(phys["cube_contacts"],label,SIDE,0,120)[0])
                if len(truth) == 12:
                    break
        if event is None:
            return None
        if pulse_step is not None and event["event_step"]+11 >= pulse_step:
            raise ValueError("Calibration overlaps a transverse pulse")
        if len(truth) != 12 or None in truth:
            raise ValueError("Missing post-decision target")
        return dict(event,loaded_ticks=sum(truth),ticks=len(truth),target=sum(truth)>=7)


def fit(rows):
    x=np.asarray([r["feature"] for r in rows])
    y=np.asarray([r["target"] for r in rows],dtype=float)
    if min(sum(y),sum(1-y))==0:
        raise ValueError("Need independently recorded loaded and clear releases")
    w=np.where(y,.5/sum(y),.5/sum(1-y))
    center=x.mean(axis=0)
    scale=x.std(axis=0)
    z=(x-center)/scale
    z-=w@z
    beta=np.linalg.solve(z.T@(w[:,None]*z)+.3*np.eye(2),z.T@(w*(y-.5)))
    weights=beta/scale
    return float(.5-(w@x)@weights),weights


def main(output):
    results=Path(__file__).resolve().parent/"results"
    paths=sorted((results/"u4_handover").glob("*/paired.jsonl.gz"))
    paths+=sorted((results/"u4_side_pulse").glob("*/paired.jsonl.gz"))
    records={}
    skipped=[]
    for path in paths:
        row=extract(path)
        if row is None:
            skipped.append(path.parent.name)
        else:
            records[path.parent.name]=row
    for name, row in records.items():
        c,b=fit([v for k,v in records.items() if k!=name])
        p=float(c+np.dot(b,row["feature"]))
        row["loo_score"]=p
        row["loo_call"]=True if p>=.65 else False if p<=.35 else None
    c,b=fit(list(records.values()))
    report={"sources":["u4_handover","u4_side_pulse"],"unused_missing_event":skipped,
            "features":["left_normal_recoil_um","right_shear_N"],
            "target":"Side bearing on >=7 of 12 physical steps starting with the causal recoil decision. All features stop at the decision. Events precede any transverse pulse.",
            "fit":"Class-balanced standardized linear ridge, lambda=0.3, one sample per physical trial. Score is a decision score, not a calibrated probability. Thresholds 0.35/0.65. Whole-trial LOO is a development diagnostic.",
            "intercept":c,"weights":b.tolist(),"development_runs":records}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    main(p.parse_args().output)
