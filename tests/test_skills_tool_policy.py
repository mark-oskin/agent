"""Skill trigger matching and per-turn tool allowlists."""

from agentlib.skills.selection import match_skill_detail


def test_short_skill_trigger_does_not_match_inside_mcp():
    skills = {
        "python": {
            "triggers": ["py", "python"],
            "tools": ["call_python"],
        }
    }
    sid, _ = match_skill_detail("use the MCP get_headers tool", skills)
    assert sid is None
    sid2, _ = match_skill_detail("write a py script", skills)
    assert sid2 == "python"


def test_grounded_cli_trigger_matches_how_do_i_use():
    skills = {
        "grounded_cli": {
            "triggers": ["how do i use"],
            "tools": ["run_command", "read_file"],
        }
    }
    sid, tr = match_skill_detail("How do I use get_headers for my inbox?", skills)
    assert sid == "grounded_cli"
    assert tr == "how do i use"


def test_match_skill_for_turn_off_by_default(monkeypatch):
    from tests.harness import build_test_session

    _app, sess = build_test_session(monkeypatch)
    sid, tr = sess._match_skill_for_turn("How do I use get_headers for my inbox?")
    assert sid is None and tr is None


def test_match_skill_for_turn_when_enabled(monkeypatch):
    from tests.harness import build_test_session

    _app, sess = build_test_session(monkeypatch)
    sess._match_skill_detail = match_skill_detail
    sess.settings.set(("agent", "skill_auto_match_triggers"), True)
    sid, tr = sess._match_skill_for_turn("How do I use get_headers for my inbox?")
    assert sid == "grounded_cli"
    assert tr == "how do i use"
