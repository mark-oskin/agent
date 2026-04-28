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


def test_agent_session_execute_use_skill_unknown(monkeypatch):
    d = importlib.import_module("agent")
    monkeypatch.setattr(d, "_route_requires_websearch", lambda *a, **k: None)
    monkeypatch.setattr(d, "call_ollama_chat", lambda *a, **k: '{"action":"answer","answer":"ok"}')

    s = d.AgentSession.from_prefs(None)
    r = s.execute("/skill definitely_not_a_skill hello")
    assert "unknown skill" in r.output.lower()


def test_agent_session_execute_skill_list(monkeypatch):
    d = importlib.import_module("agent")
    s = d.AgentSession.from_prefs(None)
    out = s.execute("/skill list").output
    assert "Skills:" in out
    assert "- python" in out or "- shell_scripting" in out


def test_agent_session_execute_skill_and_prompt_template_help(monkeypatch):
    d = importlib.import_module("agent")
    s = d.AgentSession.from_prefs(None)
    out1 = s.execute("/skill help").output.lower()
    assert "skill" in out1 and "prompt_template" in out1
    out2 = s.execute("/settings prompt_template help").output.lower()
    assert "prompt_template" in out2 and "skill" in out2


def test_agent_session_execute_help_is_top_level(monkeypatch):
    d = importlib.import_module("agent")
    s = d.AgentSession.from_prefs(None)
    out = s.execute("/help").output.lower()
    assert "/settings" in out and "try /settings help" in out
    assert "/skill" in out and "try /skill help" in out


