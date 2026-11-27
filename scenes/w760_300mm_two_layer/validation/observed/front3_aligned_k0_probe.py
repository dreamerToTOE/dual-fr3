"""Reuse the front-row probe after restoration of F3's measured horizontal alignment."""
from pathlib import Path
from sys import path
from textwrap import indent

scene = Path.cwd() / "scenes/w760_300mm_two_layer"
path.insert(0, str(scene / "validation/observed"))

import front_row_k0_probe as probe
from source_hooks import replace_once

def resume_alignment(source):
    source = probe.with_front_probe(source)
    source = replace_once(source, "        def active_square_hint(phase, support):\n",
                          '''        def active_square_hint(phase, support):
            if phase == "front_floor_landing_resume" and CHECKPOINT.get("front_alignment_resume"):
                # Locate fresh frames near the last observed pose after lateral alignment.
                prior = CHECKPOINT["front_alignment_resume"]["prior_aligned_rgb"]
                return np.asarray(prior["position_m"]), None
''')
    start = source.index('                    cube_start, attitude = observe_b_cube(f"{prefix}_landing_for_lateral")')
    stop = source.index('                    lateral_gap = push_sign *', start)
    source = source[:start] + '''                    alignment_resume = CHECKPOINT.get("front_alignment_resume")
                    if alignment_resume:
                        if slot != alignment_resume["slot"] or current_task["cube_id"] != alignment_resume["cube_id"]:
                            raise RuntimeError("F3 lateral-alignment checkpoint identity mismatch")
                        record(f"{prefix}_alignment_resume", "RESTORED_SIDE_ALIGNMENT",
                               source_step=alignment_resume["source_step"],
                               prior_side_gap_m=alignment_resume["side_gap_m"])
                        aligned, attitude = observe_b_cube(f"{prefix}_after_lateral_rgb")
                        lateral = aligned.copy()
                        lateral[0] = alignment_resume["target_x_m"]
                    else:
''' + indent(source[start:stop], '    ') + source[stop:]
    return replace_once(source, '''                if approach_count == (3 if slot in (1, 2) else 2 if slot in (3, 4) else 1):''',
                        '''                expected_count = (1 if CHECKPOINT.get("front_alignment_resume")
                                  else 3 if slot in (1, 2) else 2 if slot in (3, 4) else 1)
                if approach_count == expected_count:''')

if __name__ == "__main__":
    probe.observed_rear.install = resume_alignment
    probe.markerless_rows.main()
