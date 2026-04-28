import importlib


def test_agent_session_stateful_and_independent(monkeypatch):
    d = importlib.import_module("agent")

    # Stub the LLM call so the test is deterministic and doesn't require Ollama.
    calls = []

    def fake_chat(messages, *a, **k):
        # Record the last "User request:" block to confirm state/history grows.
        calls.append(messages[-1]["content"])
        return '{"action":"answer","answer":"ok","next_action":"finalize"}'

    monkeypatch.setattr(d, "call_ollama_chat", fake_chat)
    monkeypatch.setattr(d, "_route_requires_websearch", lambda *a, **k: None)

    s1 = d.AgentSession.from_prefs(None)
    s2 = d.AgentSession.from_prefs(None)

    # Session 1: run twice; should keep its history.
    a1, out1 = s1.run_query("hi")
    a2, out2 = s1.run_query("again")
    assert a1 and out1 == "ok"
    assert a2 and out2 == "ok"
    assert len(s1.messages) > 0

    # Session 2: separate history.
    b1, out3 = s2.run_query("separate")
    assert b1 and out3 == "ok"
    assert len(s2.messages) > 0
    assert s1.messages is not s2.messages


def test_agent_session_execute_settings_and_show(monkeypatch):
    d = importlib.import_module("agent")
    monkeypatch.setattr(d, "_route_requires_websearch", lambda *a, **k: None)
    monkeypatch.setattr(d, "call_ollama_chat", lambda *a, **k: '{"action":"answer","answer":"ok"}')

    s = d.AgentSession.from_prefs(None)
    r1 = s.execute('/settings verbose 1')
    assert "verbose level" in r1.output

    r2 = s.execute('/settings model tinyllama:latest')
    assert "OLLAMA_MODEL set" in r2.output

    r3 = s.execute("/show model")
    assert "Primary LLM:" in r3.output


