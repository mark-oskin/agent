"""Tests for ``/load`` / ``/unload`` REPL extension modules."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from tests.harness import build_test_session, run_session_lines

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_repl_load_unload_extension_commands(tmp_path, monkeypatch):
    simple = FIXTURES / "repl_ext_simple.py"
    lines = [
        f"/load {simple}",
        "/ext_ok hello",
        "/unload",
        "/ext_ok should_fail",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "ext_ok:hello" in out
    assert "Unknown command '/ext_ok'" in out


def test_repl_extension_dispatch_with_apostrophe_in_rest(monkeypatch):
    """Regression: shlex on full line broke /ext_ok when rest contained ``doesn't``."""
    simple = FIXTURES / "repl_ext_simple.py"
    lines = [
        f"/load {simple}",
        "/ext_ok modify so it doesn't ask twice",
        "/unload",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "ext_ok:modify so it doesn't ask twice" in out
    assert "Unknown command '/ext_ok'" not in out


def test_repl_load_invoke_call_python_bridge(tmp_path, monkeypatch):
    bridge = FIXTURES / "repl_ext_bridge.py"
    lines = [
        f"/load {bridge}",
        "/pingx hi there",
        "/unload",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "pong:hi there" in out


def test_repl_unload_when_empty(monkeypatch):
    lines = ["/unload", "/quit"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    assert "No REPL extensions loaded" in buf.getvalue()


def test_repl_extension_register_help_shown_in_slash_help(monkeypatch):
    simple = FIXTURES / "repl_ext_simple.py"
    lines = [
        f"/load {simple}",
        "/help",
        "/unload",
        "/help",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "Loaded extensions (/load):" in out
    assert "/ext_ok" in out


def test_repl_load_unknown_option_warns(monkeypatch):
    simple = FIXTURES / "repl_ext_simple.py"
    lines = [
        f"/load {simple} --not_a_real_flag_ever",
        "/ext_ok hi",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "unknown /load option" in out
    assert "ext_ok:hi" in out


def test_repl_load_options_passed_to_registry(monkeypatch):
    probe = FIXTURES / "repl_ext_flag_probe.py"
    lines = [
        f"/load {probe} --single_lane",
        "/probe_load_flags",
        "/extensions",
        "/unload",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "single_lane" in out.replace(" ", "")
    assert "[single_lane]" in out


def test_repl_load_single_hyphen_option_normalized(monkeypatch):
    probe = FIXTURES / "repl_ext_flag_probe.py"
    lines = [
        f"/load {probe} --single-lane",
        "/probe_load_flags",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    assert "single_lane" in buf.getvalue().replace(" ", "")


def test_repl_load_flags_before_path_errors(monkeypatch):
    probe = FIXTURES / "repl_ext_flag_probe.py"
    lines = [
        f"/load --single_lane {probe}",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    assert "no file path before options" in buf.getvalue()


def test_repl_load_same_file_twice_replaces_help(monkeypatch):
    """Re-/load same path should not duplicate extension lines in /help."""
    simple = FIXTURES / "repl_ext_simple.py"
    lines = [
        f"/load {simple}",
        f"/load {simple}",
        "/help",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert out.count("test fixture extension") == 1
    assert "replacing earlier registration" in out


_CODE_EXT = Path(__file__).resolve().parent.parent / "extensions" / "code.py"


def test_repl_load_bare_help_is_generic(monkeypatch):
    lines = ["/load", "/quit"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "/load FILE.py --help" in out or "/load FILE.py -h" in out
    assert "/load info FILE.py" in out
    assert "extensions/code.py" not in out
    assert "--single_lane" not in out


def test_repl_load_info_and_help_describe_without_registering(monkeypatch):
    fx = FIXTURES / "repl_ext_with_describe.py"
    lines = [
        f"/load info {fx}",
        "/ext_with_desc_z x",
        f"/load {fx} --help",
        "/ext_with_desc_z y",
        f"/load {fx} -h",
        "/ext_with_desc_z z",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert out.count("TEST_EXT_LOAD_OPTIONS_LINE") == 3
    assert out.count("Unknown command '/ext_with_desc_z'") == 3


def test_repl_load_info_usage_when_missing_path(monkeypatch):
    lines = ["/load info", "/quit"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    assert "Usage: /load info FILE.py" in buf.getvalue()


def test_repl_load_help_trailing_tokens_warn(monkeypatch):
    fx = FIXTURES / "repl_ext_with_describe.py"
    lines = [
        f"/load {fx} --help --ignored",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "tokens after --help are ignored" in out
    assert "TEST_EXT_LOAD_OPTIONS_LINE" in out


def test_load_code_py_single_lane_updates_session_flag(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    session.execute_line(f"/load {_CODE_EXT} --single_lane")
    assert session.repl_code_extension_single_lane is True
    session.execute_line(f"/load {_CODE_EXT}")
    assert session.repl_code_extension_single_lane is False
    session.execute_line("/unload")
    assert session.repl_code_extension_single_lane is False
