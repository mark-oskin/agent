from __future__ import annotations

from agentlib.cli import parse_main_argv
from agentlib.llm.profile import default_primary_llm_profile


def test_cli_verbose_3_consumes_level_and_starts_repl_mode():
    # When invoked as: agent.py --verbose 3
    # the "3" should be consumed as the verbose level, not treated as a query part.
    parsed = parse_main_argv(
        ["--verbose", "3"],
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
    assert parsed.verbose == 3
    assert parsed.query_parts == []

