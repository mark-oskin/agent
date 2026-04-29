import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set

from .prefs import set_agent_prefs_path_override


def parse_and_apply_cli_config_flag(argv: List[str]) -> List[str]:
    """
    Extract --config <file> or --config=<file> from argv, apply override, and return remaining args.

    This must run before loading prefs.
    """
    out: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--config":
            if i + 1 >= len(argv) or not str(argv[i + 1]).strip():
                print("Error: --config requires a file path.", file=sys.stderr)
                sys.exit(2)
            set_agent_prefs_path_override(argv[i + 1])
            i += 2
            continue
        if isinstance(a, str) and a.startswith("--config="):
            p = a.split("=", 1)[1]
            if not str(p).strip():
                print("Error: --config=<file> requires a non-empty file path.", file=sys.stderr)
                sys.exit(2)
            set_agent_prefs_path_override(p)
            i += 1
            continue
        out.append(a)
        i += 1
    return out


@dataclass
class CliParseResult:
    verbose: int
    verbose_flag_set: bool
    second_opinion_enabled: bool
    second_opinion_flag_set: bool
    cloud_ai_enabled: bool
    cloud_ai_flag_set: bool
    load_context_path: Optional[str]
    save_context_path: Optional[str]
    enabled_tools: Set[str]
    primary_profile: Any
    reviewer_hosted_profile: Any
    reviewer_ollama_model: Optional[str]
    prompt_template_selected: Optional[str]
    query_parts: List[str]
    help_requested: bool = False


def parse_main_argv(
    argv: List[str],
    *,
    verbose: int,
    second_opinion_enabled: bool,
    cloud_ai_enabled: bool,
    save_context_path: Optional[str],
    enabled_tools: Set[str],
    primary_profile: Any,
    reviewer_hosted_profile: Any,
    reviewer_ollama_model: Optional[str],
    strip_leading_dashes_flag: Callable[[str], str],
    print_cli_help: Callable[[], None],
    apply_cli_primary_model: Callable[[str, Any], Any],
    normalize_tool_name: Callable[[str], Optional[str]],
    format_unknown_tool_hint: Callable[[str], str],
    format_settings_tools_list: Callable[[Set[str]], str],
) -> CliParseResult:
    def stop_with_error(msg: str) -> CliParseResult:
        # Preserve historical behavior: most CLI parse errors are printed to stdout
        # so tests (and scripting) see them in the captured output.
        print(msg)
        return CliParseResult(
            verbose=verbose,
            verbose_flag_set=verbose_flag_set,
            second_opinion_enabled=second_opinion_enabled,
            second_opinion_flag_set=second_opinion_flag_set,
            cloud_ai_enabled=cloud_ai_enabled,
            cloud_ai_flag_set=cloud_ai_flag_set,
            load_context_path=load_context_path,
            save_context_path=save_context_path,
            enabled_tools=enabled_tools,
            primary_profile=primary_profile,
            reviewer_hosted_profile=reviewer_hosted_profile,
            reviewer_ollama_model=reviewer_ollama_model,
            prompt_template_selected=prompt_template_selected,
            query_parts=query_parts,
            help_requested=True,
        )

    verbose_flag_set = False
    second_opinion_flag_set = False
    cloud_ai_flag_set = False
    query_parts: List[str] = []
    load_context_path: Optional[str] = None
    prompt_template_selected: Optional[str] = None

    i = 0
    while i < len(argv):
        a = argv[i]
        fa = strip_leading_dashes_flag(a)
        if (a or "").startswith("-") and fa in ("help", "h", "?"):
            print_cli_help()
            return CliParseResult(
                verbose=verbose,
                verbose_flag_set=verbose_flag_set,
                second_opinion_enabled=second_opinion_enabled,
                second_opinion_flag_set=second_opinion_flag_set,
                cloud_ai_enabled=cloud_ai_enabled,
                cloud_ai_flag_set=cloud_ai_flag_set,
                load_context_path=load_context_path,
                save_context_path=save_context_path,
                enabled_tools=enabled_tools,
                primary_profile=primary_profile,
                reviewer_hosted_profile=reviewer_hosted_profile,
                reviewer_ollama_model=reviewer_ollama_model,
                prompt_template_selected=prompt_template_selected,
                query_parts=query_parts,
                help_requested=True,
            )
        if (a or "").startswith("-") and (fa == "model" or fa.startswith("model=")):
            if fa == "model":
                if i + 1 >= len(argv):
                    return stop_with_error("Error: --model requires a model name.")
                mname = str(argv[i + 1]).strip()
                if not mname:
                    return stop_with_error("Error: --model name must be non-empty.")
                i += 2
            else:
                _eq = str(a).split("=", 1)
                if len(_eq) < 2 or not _eq[1].strip():
                    return stop_with_error("Error: --model=<name> requires a non-empty name.")
                mname = _eq[1].strip()
                i += 1
            primary_profile = apply_cli_primary_model(mname, primary_profile)
            continue
        if fa in ("enable-tool",):
            if i + 1 >= len(argv):
                return stop_with_error("Error: -enable_tool requires a tool name.")
            t = normalize_tool_name(str(argv[i + 1]))
            if not t:
                return stop_with_error("Error: " + format_unknown_tool_hint(str(argv[i + 1])))
            enabled_tools.add(t)
            i += 2
            continue
        if fa in ("disable-tool",):
            if i + 1 >= len(argv):
                return stop_with_error("Error: -disable_tool requires a tool name.")
            t = normalize_tool_name(str(argv[i + 1]))
            if not t:
                return stop_with_error("Error: " + format_unknown_tool_hint(str(argv[i + 1])))
            enabled_tools.discard(t)
            i += 2
            continue
        if fa in ("list-tools",):
            print(format_settings_tools_list(enabled_tools))
            return CliParseResult(
                verbose=verbose,
                verbose_flag_set=verbose_flag_set,
                second_opinion_enabled=second_opinion_enabled,
                second_opinion_flag_set=second_opinion_flag_set,
                cloud_ai_enabled=cloud_ai_enabled,
                cloud_ai_flag_set=cloud_ai_flag_set,
                load_context_path=load_context_path,
                save_context_path=save_context_path,
                enabled_tools=enabled_tools,
                primary_profile=primary_profile,
                reviewer_hosted_profile=reviewer_hosted_profile,
                reviewer_ollama_model=reviewer_ollama_model,
                prompt_template_selected=prompt_template_selected,
                query_parts=query_parts,
                help_requested=True,
            )
        if strip_leading_dashes_flag(a) == "verbose":
            if i + 1 < len(argv) and argv[i + 1] in ("0", "1", "2"):
                verbose = int(argv[i + 1])
                i += 2
            else:
                verbose = 2
                i += 1
            verbose_flag_set = True
            continue
        if a in ("--second-opinion", "--second_opinion"):
            second_opinion_enabled = True
            second_opinion_flag_set = True
            i += 1
            continue
        if a in ("--cloud-ai", "--cloud_ai"):
            cloud_ai_enabled = True
            cloud_ai_flag_set = True
            i += 1
            continue
        if a in ("--load-context", "--load_context"):
            if i + 1 >= len(argv):
                return stop_with_error("Error: --load_context requires a file path.")
            load_context_path = str(argv[i + 1])
            i += 2
            continue
        if a in ("--save-context", "--save_context"):
            if i + 1 >= len(argv):
                return stop_with_error("Error: --save_context requires a file path.")
            save_context_path = str(argv[i + 1])
            i += 2
            continue
        if a in ("--prompt-template", "--prompt_template"):
            if i + 1 >= len(argv):
                return stop_with_error("Error: --prompt-template requires a template name.")
            prompt_template_selected = str(argv[i + 1]).strip()
            i += 2
            continue
        query_parts.append(str(a))
        i += 1

    return CliParseResult(
        verbose=verbose,
        verbose_flag_set=verbose_flag_set,
        second_opinion_enabled=second_opinion_enabled,
        second_opinion_flag_set=second_opinion_flag_set,
        cloud_ai_enabled=cloud_ai_enabled,
        cloud_ai_flag_set=cloud_ai_flag_set,
        load_context_path=load_context_path,
        save_context_path=save_context_path,
        enabled_tools=enabled_tools,
        primary_profile=primary_profile,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        prompt_template_selected=prompt_template_selected,
        query_parts=query_parts,
        help_requested=False,
    )

