"""CLI ``--debug_log`` parsing."""

from __future__ import annotations

from agentlib.cli import parse_main_argv
from agentlib.llm.profile import default_primary_llm_profile


def _parse(argv: list[str]):
    return parse_main_argv(
        argv,
        verbose=0,
        second_opinion_enabled=False,
        cloud_ai_enabled=False,
        save_context_path=None,
        enabled_tools=set(),
        primary_profile=default_primary_llm_profile(),
        reviewer_hosted_profile=None,
        reviewer_ollama_model=None,
        strip_leading_dashes_flag=lambda s: (s or "").lstrip("-").lower().replace("_", "-"),
        print_cli_help=lambda: None,
        apply_cli_primary_model=lambda name, prof: prof,
        normalize_tool_name=lambda s: None,
        format_unknown_tool_hint=lambda s: s,
        format_settings_tools_list=lambda tools: "",
    )


def test_debug_log_equals_form():
    p = _parse(["--debug_log=/tmp/llm.log", "hello"])
    assert p.debug_llm_log_path == "/tmp/llm.log"
    assert p.query_parts == ["hello"]


def test_debug_log_two_token_form():
    p = _parse(["--debug_log", "out/debug.txt", "q"])
    assert p.debug_llm_log_path == "out/debug.txt"
    assert p.query_parts == ["q"]


def test_debug_log_hyphen_alias():
    p = _parse(["--debug-log", "/var/tmp/x.log"])
    assert p.debug_llm_log_path == "/var/tmp/x.log"
