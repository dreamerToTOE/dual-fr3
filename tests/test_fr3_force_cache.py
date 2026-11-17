from copy import deepcopy

import pytest

from fr3_ab_visual_runner import (
    A_FORCE_LIMITS, A_TRANSFER_REVISION, B1_FORCE_LIMITS, B2_CONTROLLER_PARAMETERS,
    CONFIG, FORCE_CONTROLLER_REVISION, compatible_force_cache, compatible_full_force_cache,
)


def _accepted_cache():
    return {"status": "COMPLETE", "controller_revision": FORCE_CONTROLLER_REVISION,
            "controller_parameters": deepcopy(B2_CONTROLLER_PARAMETERS),
            "scene_config": deepcopy(CONFIG), "force_validation_status": "PASS_B2_FORCE_CONTROL",
            "phases": [{"phase": "B2_push"}]}


def test_only_matching_force_validated_cache_can_be_loaded():
    assert compatible_force_cache(_accepted_cache())
    assert not compatible_force_cache({"status": "COMPLETE", "scene_id": CONFIG["scene_id"], "phases": []})
    assert not compatible_force_cache([])


@pytest.mark.parametrize("change", ["unvalidated", "revision", "scene", "controller"])
def test_stale_or_unvalidated_force_cache_is_rejected(change):
    cache = _accepted_cache()
    if change == "unvalidated":
        cache["force_validation_status"] = "UNVERIFIED"
    elif change == "revision":
        cache["controller_revision"] += "_old"
    elif change == "scene":
        cache["scene_config"]["cube_size_m"] += .01
    else:
        cache["controller_parameters"]["gain_m_per_Ns"] *= 2
    assert not compatible_force_cache(cache)


def test_full_force_cache_requires_all_phase_acceptance_and_current_a_contract():
    cache = _accepted_cache()
    assert not compatible_full_force_cache(cache)
    cache.update(a_transfer_revision=A_TRANSFER_REVISION, a_force_limits=deepcopy(A_FORCE_LIMITS),
                 b1_force_limits=deepcopy(B1_FORCE_LIMITS), task_force_status="FULL_FLOW_FORCE_PASS")
    assert compatible_full_force_cache(cache)
    for key, value in (("task_force_status", "FULL_FLOW_FORCE_FAIL"),
                       ("a_transfer_revision", "old_joint_spline"),
                       ("a_force_limits", {**A_FORCE_LIMITS, "max_internal_force_N": 500}),
                       ("b1_force_limits", None)):
        assert not compatible_full_force_cache({**cache, key: value})
