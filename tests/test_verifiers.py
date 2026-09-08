"""Generated code executes only after isolation, with trusted expected results outside."""

import pytest

from minifrontier.training.verifiers import check_python, reward


def test_pure_function_reward_and_forbidden_file_process_network_access(tmp_path):
    cases = [dict(args=[3, 7], expected=10), dict(args=[-9, 5], expected=-4)]
    try:
        result = check_python("def add(a,b):\n return a+b", "add", cases)
    except RuntimeError as error:
        pytest.skip(str(error))
    assert result["score"] == 1
    for body in (
        f"open({str(tmp_path / 'forbidden')!r}, 'w').write('bad')",
        "__import__('os').fork()",
        "__import__('socket').socket()",
    ):
        result = check_python("def add(a,b):\n return " + body, "add", cases)
        assert result["score"] == 0
    assert not (tmp_path / "forbidden").exists()
    malformed = check_python("import os\nos.write(1, b'[]')\nos._exit(0)", "add", cases)
    assert malformed["score"] == 0 and malformed["reason"] == "invalid_worker_result"


def test_structured_rewards_never_fall_back_to_arithmetic():
    assert reward('{"count":3}', dict(kind="json", expected={"count": 3})) == 1
    assert reward("three", dict(kind="integer", answer=3)) == 0
    with pytest.raises(ValueError, match="unimplemented verifier"):
        reward("3", dict(kind="missing", answer=3))
