"""Case simulation with a fake model. No network."""
import json

import pytest

from ragforge.llm import LLMError, LLMResponse
from ragforge.simulate import RESPOND_SCHEMA, START_SCHEMA, CaseSession, Simulator
from ragforge.store import Hit

HITS = [
    Hit("c1", "d1", "CRAO is a blockage of the retinal artery causing sudden vision loss.",
        "crao.pdf", 1, 1, 0.82),
    Hit("c2", "d1", "Ocular massage and paracentesis have been attempted.",
        "crao.pdf", 3, 4, 0.78),
]


def scene(vector="head", situation="She has just arrived.", prompt="What is going on?"):
    return {"vector": vector, "situation": situation, "prompt": prompt}


def start_reply(**overrides):
    base = {
        "title": "Sudden painless vision loss",
        "presentation": "An 82-year-old woman presents with sudden painless loss of "
                        "vision in her right eye two hours ago.",
        "persona": "Frightened and gripping her handbag; she lives alone and is "
                   "terrified of losing her independence.",
        "citations": ["crao.pdf p1"],
        "scene": scene("head"),
    }
    base.update(overrides)
    return base


def respond_reply(**overrides):
    base = {
        "reaction": "She listens, then her eyes fill with tears.",
        "scene": scene("heart", "She asks, 'Am I going to go blind?'",
                       "Reply to her, in your own words."),
    }
    base.update(overrides)
    return base


class ScriptedClient:
    """Returns queued payloads in order, recording each request."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def chat_json(self, system, user, schema, num_ctx=16384):
        if not self.payloads:
            raise AssertionError("the simulator made more calls than were scripted")
        payload = self.payloads.pop(0)
        self.calls.append({"system": system, "user": user, "schema": schema})
        return payload, LLMResponse(json.dumps(payload), "", 100, 50, 5.0)


class FakePipeline:
    def __init__(self, hits=None):
        self.hits = HITS if hits is None else hits

    def search(self, query, k=5):
        return self.hits[:k]


def simulator(payloads, hits=None):
    return Simulator(FakePipeline(hits), ScriptedClient(payloads))


def run_full_case(debrief_payload):
    """Play all three scenes and score."""
    sim = simulator([
        start_reply(),
        respond_reply(),
        respond_reply(scene=scene("hands", "She is stable.", "What do you do, in order?")),
        respond_reply(scene=None),
        debrief_payload,
    ])
    session = sim.start("central retinal artery occlusion")
    session = sim.respond(session, "Central retinal artery occlusion.")
    session = sim.respond(session, "I understand this is frightening. Let me explain.")
    session = sim.respond(session, "Ocular massage, lower the pressure, urgent referral.")
    return sim, session


# --- starting a case --------------------------------------------------------

def test_start_builds_a_case_from_the_corpus():
    session = simulator([start_reply()]).start("CRAO")
    assert session.title == "Sudden painless vision loss"
    assert "82-year-old" in session.presentation


def test_the_patient_has_an_emotional_arc():
    """Section 7 MODE 6 requires a persona with an emotional state."""
    session = simulator([start_reply()]).start("CRAO")
    assert "terrified" in session.persona


def test_the_first_scene_targets_head():
    session = simulator([start_reply()]).start("CRAO")
    assert session.scene.vector == "head"


def test_the_scene_order_is_fixed_regardless_of_what_the_model_labels():
    """The vector sequence is a design decision, not the model's to choose."""
    session = simulator([start_reply(scene=scene("hands"))]).start("CRAO")
    assert session.scene.vector == "head"


def test_case_citations_are_verified():
    session = simulator([start_reply(citations=["crao.pdf p1", "invented.pdf p9"])]).start("CRAO")
    assert session.citations == ["crao.pdf p1"]
    assert any("invented.pdf p9" in w for w in session.warnings)


def test_start_uses_the_simulation_prompt_and_schema():
    sim = simulator([start_reply()])
    sim.start("CRAO")
    call = sim.client.calls[0]
    assert call["schema"] == START_SCHEMA
    assert "MODE 6" in call["system"]
    assert "SIMULATION STEP: start" in call["user"]


def test_an_empty_topic_is_rejected():
    with pytest.raises(ValueError, match="topic is required"):
        simulator([]).start("   ")


def test_no_matching_passages_refuses_to_invent_a_case():
    with pytest.raises(LLMError, match="nothing in the corpus"):
        simulator([], hits=[]).start("sourdough bread")


def test_a_missing_opening_scene_is_an_error():
    with pytest.raises(LLMError, match="opening scene"):
        simulator([start_reply(scene={"vector": "head", "situation": "x", "prompt": ""})]).start("CRAO")


# --- playing the case -------------------------------------------------------

def test_responding_records_the_turn_and_advances():
    sim = simulator([start_reply(), respond_reply()])
    session = sim.respond(sim.start("CRAO"), "It is a CRAO.")
    assert session.turns[0].learner_reply == "It is a CRAO."
    assert session.turns[0].scene.vector == "head"
    assert session.scene.vector == "heart", "scene 2 must target HEART"


def test_the_heart_scene_puts_the_patient_in_the_learners_way():
    """The learner must have something to respond to, or HEART is unmeasurable."""
    sim = simulator([start_reply(), respond_reply()])
    session = sim.respond(sim.start("CRAO"), "It is a CRAO.")
    assert "blind" in session.scene.situation
    assert "your own words" in session.scene.prompt


def test_the_case_ends_after_three_scenes():
    _, session = run_full_case(None)
    assert session.finished
    assert len(session.turns) == 3
    assert [t.scene.vector for t in session.turns] == ["head", "heart", "hands"]


def test_the_transcript_carries_what_the_learner_said():
    _, session = run_full_case(None)
    transcript = session.transcript()
    assert "I understand this is frightening" in transcript
    assert "Ocular massage" in transcript


def test_the_model_sees_the_transcript_on_later_turns():
    sim, _ = run_full_case(None)
    assert "LEARNER SAID: Central retinal artery occlusion." in sim.client.calls[2]["user"]


def test_responding_uses_the_respond_schema():
    sim = simulator([start_reply(), respond_reply()])
    sim.respond(sim.start("CRAO"), "answer")
    assert sim.client.calls[1]["schema"] == RESPOND_SCHEMA


def test_an_empty_reply_is_rejected():
    sim = simulator([start_reply()])
    with pytest.raises(ValueError, match="reply is required"):
        sim.respond(sim.start("CRAO"), "  ")


def test_responding_to_a_finished_case_is_rejected():
    _, session = run_full_case(None)
    with pytest.raises(ValueError, match="already finished"):
        simulator([]).respond(session, "more")


def test_scene_number_tracks_progress():
    sim = simulator([start_reply(), respond_reply()])
    session = sim.start("CRAO")
    assert session.scene_number == 1
    assert sim.respond(session, "x").scene_number == 2


# --- the debrief ------------------------------------------------------------

def debrief_payload(**overrides):
    base = {
        "verdict": "partially_correct",
        "scores": [
            {"vector": "head", "assessed": True, "level": 3,
             "anchor": "Relational", "evidence": "Central retinal artery occlusion."},
            {"vector": "heart", "assessed": True, "level": 2,
             "anchor": "Acknowledges patient emotion but doesn't alter plan",
             "evidence": "I understand this is frightening."},
            {"vector": "hands", "assessed": True, "level": 2,
             "anchor": "Knows How", "evidence": "Ocular massage, lower the pressure."},
        ],
        "acknowledgement": "You named the diagnosis quickly.",
        "what_was_right": ["Correct diagnosis."],
        "what_was_missed": ["You did not answer her actual question."],
        "model_answer": "A strong response names the diagnosis, answers her fear "
                        "directly, then moves to urgent management.",
        "citations": ["crao.pdf p3-4"],
        "feed_forward": "Next time, answer the patient's question before explaining.",
        "grader_confidence": "high",
    }
    base.update(overrides)
    return base


def test_the_debrief_scores_all_three_vectors():
    _, session = run_full_case(debrief_payload())
    sim = simulator([debrief_payload()])
    result = sim.score(session)
    assert [s.vector for s in result.scores] == ["head", "heart", "hands"]
    assert all(s.assessed for s in result.scores)


def test_heart_is_scored_from_the_learners_own_words():
    """The whole point: empathy is measured from what they actually said."""
    _, session = run_full_case(debrief_payload())
    result = simulator([debrief_payload()]).score(session)
    heart = next(s for s in result.scores if s.vector == "heart")
    assert heart.evidence == "I understand this is frightening."
    assert heart.level == 2


def test_a_strong_hands_score_does_not_mask_a_weak_heart_score():
    """Section 7 MODE 4 forbids collapsing the vectors into one number."""
    _, session = run_full_case(debrief_payload())
    payload = debrief_payload(scores=[
        {"vector": "head", "assessed": True, "level": 4, "anchor": "a", "evidence": "e"},
        {"vector": "heart", "assessed": True, "level": 0,
         "anchor": "Patient absent from response", "evidence": "e"},
        {"vector": "hands", "assessed": True, "level": 4, "anchor": "a", "evidence": "e"},
    ])
    result = simulator([payload]).score(session)
    levels = {s.vector: s.level for s in result.scores}
    assert levels == {"head": 4, "heart": 0, "hands": 4}


def test_debrief_citations_are_verified():
    _, session = run_full_case(debrief_payload())
    payload = debrief_payload(citations=["crao.pdf p1", "fabricated.pdf p2"])
    result = simulator([payload]).score(session)
    assert result.citations == ["crao.pdf p1"]
    assert result.dropped_citations == ["fabricated.pdf p2"]


def test_low_confidence_in_the_debrief_escalates():
    _, session = run_full_case(debrief_payload())
    result = simulator([debrief_payload(grader_confidence="low")]).score(session)
    assert result.needs_faculty_review


def test_scoring_an_unplayed_case_is_rejected():
    session = CaseSession("t", "p", "persona", [], list(HITS), None)
    with pytest.raises(ValueError, match="no completed scenes"):
        simulator([]).score(session)


def test_the_debrief_is_kept_on_the_session():
    _, session = run_full_case(debrief_payload())
    simulator([debrief_payload()]).score(session)
    assert session.debrief is not None
