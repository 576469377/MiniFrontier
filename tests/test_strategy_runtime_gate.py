import json

import pytest

from minifrontier.training.strategy_gate import validate_runtime


def test_inherited_attention_and_actual_device_count_must_match_profile(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(dict(gpu_ids=[2, 3], phases=[dict(id="Q-SFT", attention_phase="sparse_cpt")]))
    )
    validate_runtime(plan, "Q-SFT", "sparse_cpt", 2)
    with pytest.raises(ValueError, match="attention phase"):
        validate_runtime(plan, "Q-SFT", "dense_pretrain", 2)
    with pytest.raises(ValueError, match="device count"):
        validate_runtime(plan, "Q-SFT", "sparse_cpt", 1)
