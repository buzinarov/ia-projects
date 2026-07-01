"""Tests for the router's dispatch logic, with the skills stubbed out so no
model is downloaded. Verifies the keyword fallback picks a sane skill and
that dispatch() wires arguments through and fails safely."""
import src.agent as agent


def test_keyword_route_translate():
    skill, args = agent.keyword_route("Translate this to Spanish: I love my car")
    assert skill == "translate"
    assert "love my car" in args["text"].lower()


def test_keyword_route_question_goes_to_answer():
    skill, args = agent.keyword_route("What did they like about the brand?")
    assert skill == "answer"
    assert "question" in args and "context" in args


def test_keyword_route_sentiment_question_goes_to_triage():
    # A sentiment-intent question must not be mis-routed to extractive QA.
    skill, _ = agent.keyword_route("Is this review positive? 'Best truck ever.'")
    assert skill == "triage"


def test_keyword_route_summarize():
    skill, args = agent.keyword_route("Summarize: this is a very long review ...")
    assert skill == "digest"


def test_keyword_route_defaults_to_triage():
    skill, _ = agent.keyword_route("This dealership ruined my weekend.")
    assert skill == "triage"


def test_dispatch_calls_the_named_skill(monkeypatch):
    captured = {}

    def fake_triage(review):
        captured["review"] = review
        return {"skill": "triage", "payload": {"sentiment": "NEGATIVE", "confidence": 0.8}}

    monkeypatch.setitem(agent.SKILL_FUNCS, "triage", fake_triage)
    out = agent.dispatch("triage", {"review": "bad car"})
    assert captured["review"] == "bad car"
    assert out["payload"]["sentiment"] == "NEGATIVE"


def test_dispatch_unknown_skill_returns_error():
    out = agent.dispatch("teleport", {"x": 1})
    assert "error" in out


def test_dispatch_bad_arguments_returns_error(monkeypatch):
    monkeypatch.setitem(agent.SKILL_FUNCS, "triage", lambda review: review)
    out = agent.dispatch("triage", {"wrong_arg": "oops"})
    assert "error" in out


def test_chat_falls_back_when_ollama_model_fails(monkeypatch):
    # Ollama reachable, but the model errors at runtime (e.g. OOM): the agent
    # must degrade to keyword routing, not raise.
    monkeypatch.setattr(agent, "_ollama_available", lambda: True)
    def boom(*a, **k):
        raise RuntimeError("out-of-memory during startup")
    monkeypatch.setattr(agent, "_chat_ollama", boom)
    monkeypatch.setitem(
        agent.SKILL_FUNCS, "triage",
        lambda review: {"skill": "triage", "payload": {"sentiment": "NEGATIVE", "confidence": 0.9}},
    )
    out = agent.chat("Is this review positive or negative? 'It broke down.'")
    assert out["skill"] == "triage"
    assert out["via"].startswith("keyword (ollama failed")
    assert "NEGATIVE" in out["reply"]


def test_chat_offline_uses_keyword_router(monkeypatch):
    monkeypatch.setitem(
        agent.SKILL_FUNCS, "triage",
        lambda review: {"skill": "triage", "payload": {"sentiment": "POSITIVE", "confidence": 0.9}},
    )
    out = agent.chat("Best car I have ever owned!", use_ollama=False)
    assert out["via"] == "keyword"
    assert out["skill"] == "triage"
    assert "POSITIVE" in out["reply"]
