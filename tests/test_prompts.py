"""Per-mode prompt slicing. The spec files are the fixture."""
import pytest

from ragforge.config import Settings
from ragforge.prompts import build_system_prompt, load_raw


@pytest.fixture(scope="module")
def full():
    agent, deployment = load_raw()
    return agent + deployment


def test_every_mode_keeps_the_absolute_constraints():
    """R0.1-R0.6 override everything, so they can never be trimmed away."""
    for mode in ("teach", "assess", "simulate"):
        prompt = build_system_prompt(mode)
        assert "R0.1" in prompt, f"{mode} lost the grounding rule"
        assert "R0.2" in prompt, f"{mode} lost the no-fabrication rule"


def test_every_mode_keeps_the_identity():
    for mode in ("teach", "assess", "simulate"):
        assert "3H Pedagogical Agent" in build_system_prompt(mode)


def test_teaching_carries_the_load_governor_and_mode_3():
    prompt = build_system_prompt("teach")
    assert "COGNITIVE LOAD GOVERNOR" in prompt
    assert "MODE 3: TEACHING" in prompt


def test_teaching_drops_the_grading_rubrics():
    """A teaching turn never scores anything.

    D1's router names all three request shapes in every prompt, so the check is
    for the §7 spec blocks themselves, not a bare mention of the mode.
    """
    prompt = build_system_prompt("teach")
    assert "SOLO Anchors" not in prompt
    assert "STANDARDIZED PATIENT" not in prompt, "MODE 6 spec block leaked in"
    assert "Blueprinting is mandatory" not in prompt, "MODE 4 spec block leaked in"


def test_assessment_carries_the_anchors_and_escalation():
    prompt = build_system_prompt("assess")
    assert "SOLO Anchors" in prompt
    assert "Behaviorally Anchored Rating Scale" in prompt
    assert "FACULTY REVIEW" in prompt
    assert "MODE 4" in prompt and "MODE 5" in prompt


def test_simulation_carries_mode_6_and_the_anchors():
    prompt = build_system_prompt("simulate")
    assert "MODE 6" in prompt
    assert "SOLO Anchors" in prompt
    assert "MODE 4" in prompt, "the debrief scores the case"


def test_each_mode_carries_its_own_deployment_block():
    assert "D3 — Output shape" in build_system_prompt("teach")
    assert "D6 — Assessment output" in build_system_prompt("assess")
    assert "D7 — Case simulation" in build_system_prompt("simulate")


def test_citation_rules_reach_every_mode():
    """D2 is what stops invented sources; no mode may lose it."""
    for mode in ("teach", "assess", "simulate"):
        assert "D2 — Citation format" in build_system_prompt(mode)


def test_slicing_actually_shortens_the_prompt(full):
    for mode in ("teach", "assess", "simulate"):
        sliced = build_system_prompt(mode)
        assert len(sliced) < len(full), f"{mode} is no shorter than the whole spec"


def test_teaching_prompt_is_at_least_a_third_shorter(full):
    """The point of the exercise: fewer prompt tokens per turn."""
    assert len(build_system_prompt("teach")) < len(full) * 0.7


def test_an_unknown_mode_falls_back_to_the_whole_spec():
    """Fail towards a slow prompt, never towards a missing rule."""
    prompt = build_system_prompt("nonsense")
    assert "MODE 1" in prompt and "MODE 6" in prompt
    assert "SOLO Anchors" in prompt


def test_a_missing_prompt_file_is_a_clear_error(tmp_path):
    config = Settings(data_dir=tmp_path / "d", prompts_dir=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="missing prompt file"):
        build_system_prompt("teach", config)


def test_the_document_title_survives_slicing():
    """The title frames the whole contract; it sits above section 0."""
    for mode in ("teach", "assess", "simulate"):
        assert "3H PEDAGOGICAL AGENT" in build_system_prompt(mode)
