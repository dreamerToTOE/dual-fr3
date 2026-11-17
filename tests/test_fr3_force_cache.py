from copy import deepcopy

import pytest

from fr3_ab_visual_runner import (
    B2_CONTROLLER_PARAMETERS, CONFIG, FORCE_CONTROLLER_REVISION, compatible_force_cache,
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
