"""REPL slash-command registry: dispatch, /help, and completion metadata."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Callable, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agentlib.session import AgentSession, SessionLineResult

MatchKind = Literal["exact_lower", "first_token", "prefix_lower", "prefix"]
HandlerFn = Callable[["AgentSession", str], "SessionLineResult"]
VisibleFn = Callable[["AgentSession"], bool]
CompleteFn = Callable[["AgentSession", list[str], str, "ReplCompletionContext"], list[str]]


@dataclass(frozen=True)
class ReplCompletionContext:
    """Host-provided hints (e.g. TUI lane labels)."""

    agent_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplCommandSpec:
    name: str
    aliases: tuple[str, ...] = ()
    match: MatchKind = "prefix_lower"
    priority: int = 0
    help_line: str = ""
    help_order: int = 100
    handler: Optional[HandlerFn] = None
    visible: VisibleFn = lambda _s: True
    complete: Optional[CompleteFn] = None

    def all_names(self) -> tuple[str, ...]:
        return (self.name,) + self.aliases

    def matches_line(self, line: str) -> bool:
        low = line.lower().strip()
        if not low:
            return False
        cmd = low.split(None, 1)[0]
        names_low = [n.lower() for n in self.all_names()]
        if self.match in ("exact_lower", "first_token"):
            if self.match == "exact_lower":
                return low in names_low
            return cmd in names_low
        if self.match == "prefix":
            return any(line.startswith(n) for n in self.all_names())
        return any(low.startswith(n.lower()) for n in self.all_names())

    def matches_token(self, token: str) -> bool:
        low = token.lower()
        names_low = [n.lower() for n in self.all_names()]
        if self.match in ("exact_lower", "first_token"):
            return low in names_low
        if self.match == "prefix":
            return any(token.startswith(n) for n in self.all_names())
        return any(low.startswith(n.lower()) for n in self.all_names())


def _visible_fork(_s: "AgentSession") -> bool:
    return _s.python_fork_agent is not None or _s.python_fork_background_agent is not None


def _visible_kill(s: "AgentSession") -> bool:
    return (
        s.python_fork_agent is not None
        or s.python_fork_background_agent is not None
        or s.python_host_command is not None
        or s.python_enqueue_line is not None
    )


def _visible_host_list(s: "AgentSession") -> bool:
    return s.python_host_command is not None


def _visible_delegate(s: "AgentSession") -> bool:
    return s.python_enqueue_line is not None or s.python_delegate_line is not None


def _handle_quit(_session: "AgentSession", _line: str) -> "SessionLineResult":
    from agentlib.session import SessionLineResult

    return SessionLineResult(quit=True)


def _handle_clear(session: "AgentSession", _line: str) -> "SessionLineResult":
    from agentlib.session import SessionLineResult

    session.messages.clear()
    session.last_reuse_skill_id = None
    session.repl_last_user_query = None
    session.repl_last_assistant_answer = None
    return session._emit_repl_command_text(
        "Context cleared (including stored skill for /skill reuse)."
    )


def _handle_usage(session: "AgentSession", _line: str) -> "SessionLineResult":
    text = session._format_last_ollama_usage_for_repl()
    return session._emit_repl_command_text(text.strip() if text.strip() else "No data available.")


def _handle_list(session: "AgentSession", _line: str) -> "SessionLineResult":
    return session._sink_host_ctl_result(session.host_ctl("list_agents"))


def _handle_switch(session: "AgentSession", line: str) -> "SessionLineResult":
    from agentlib.session import SessionLineResult
    from agentlib.sink import sink_print_compat

    try:
        parts = shlex.split(line)
    except ValueError as e:
        sink_print_compat(f"/switch: {e}")
        return SessionLineResult()
    if len(parts) < 2:
        sink_print_compat("Usage: /switch AGENT_LABEL")
        return SessionLineResult()
    return session._sink_host_ctl_result(session.host_ctl("switch", parts[1]))


def _handle_last_answer(session: "AgentSession", line: str) -> "SessionLineResult":
    from agentlib.session import SessionLineResult
    from agentlib.sink import sink_print_compat

    try:
        parts = shlex.split(line)
    except ValueError as e:
        sink_print_compat(f"/last_answer: {e}")
        return SessionLineResult()
    arg = parts[1] if len(parts) > 1 else None
    return session._sink_host_ctl_result(session.host_ctl("last_answer", arg))


def _handle_last_question(session: "AgentSession", line: str) -> "SessionLineResult":
    from agentlib.session import SessionLineResult
    from agentlib.sink import sink_print_compat

    try:
        parts = shlex.split(line)
    except ValueError as e:
        sink_print_compat(f"/last_question: {e}")
        return SessionLineResult()
    arg = parts[1] if len(parts) > 1 else None
    return session._sink_host_ctl_result(session.host_ctl("last_question", arg))


def _handle_help(session: "AgentSession", _line: str) -> "SessionLineResult":
    from agentlib.sink import sink_print_compat
    from agentlib.session import SessionLineResult

    sink_print_compat(format_repl_help(session))
    return SessionLineResult()


def _complete_prefix(options: tuple[str, ...]) -> CompleteFn:
    def _fn(
        _session: "AgentSession",
        _tokens: list[str],
        partial: str,
        _ctx: ReplCompletionContext,
    ) -> list[str]:
        p = partial.lower()
        return [o for o in options if o.lower().startswith(p)]

    return _fn


def _complete_agent_target(
    session: "AgentSession",
    tokens: list[str],
    partial: str,
    ctx: ReplCompletionContext,
) -> list[str]:
    if len(tokens) != 1:
        return []
    cmd = tokens[0].lower()
    if cmd not in ("/switch", "/send", "/turn", "/kill"):
        return []
    if ctx.agent_labels:
        labels = list(ctx.agent_labels)
    elif session.python_enqueue_line is None and session.python_delegate_line is None:
        labels = ["self"]
    else:
        labels = []
    if cmd in ("/send", "/turn") and "self" not in [x.lower() for x in labels]:
        labels = ["self"] + labels
    p = partial.lower()
    return [lab for lab in labels if lab.lower().startswith(p)]


def _complete_set(
    session: "AgentSession",
    tokens: list[str],
    partial: str,
    _ctx: ReplCompletionContext,
) -> list[str]:
    topics = (
        "save",
        "model",
        "enable",
        "disable",
        "tools",
        "system_prompt",
        "prompt_template",
        "context",
        "thinking",
        "ollama",
        "openai",
        "agent",
        "primary",
        "extensions",
        "lock",
        "unlock",
        "help",
    )
    if len(tokens) == 1:
        return _complete_prefix(topics)(session, tokens, partial, _ctx)
    if len(tokens) >= 2 and tokens[1].lower() in ("tools",):
        if len(tokens) == 2:
            return _complete_prefix(("list", "reload", "describe", "enable", "disable"))(
                session, tokens, partial, _ctx
            )
    if len(tokens) >= 2 and tokens[1].lower() == "context":
        if len(tokens) == 2:
            return _complete_prefix(("show", "help"))(session, tokens, partial, _ctx)
    if len(tokens) >= 2 and tokens[1].lower() == "thinking":
        if len(tokens) == 2:
            return _complete_prefix(("show", "on", "off"))(session, tokens, partial, _ctx)
    if len(tokens) >= 3 and tokens[1].lower() == "primary" and tokens[2].lower() == "request_options":
        if len(tokens) == 3:
            return _complete_prefix(("show", "clear", "set", "unset", "merge", "replace", "help"))(
                session, tokens, partial, _ctx
            )
    return []


def _complete_mcp(
    _session: "AgentSession",
    tokens: list[str],
    partial: str,
    ctx: ReplCompletionContext,
) -> list[str]:
    if len(tokens) == 1:
        return _complete_prefix(
            ("help", "list", "status", "enable", "disable", "session", "reload", "remove", "add")
        )(_session, tokens, partial, ctx)
    if len(tokens) == 2 and tokens[1].lower() == "session":
        return _complete_prefix(("on", "off"))(_session, tokens, partial, ctx)
    if len(tokens) == 2 and tokens[1].lower() == "add":
        return _complete_prefix(("stdio", "http"))(_session, tokens, partial, ctx)
    return []


def _complete_last(
    _session: "AgentSession",
    tokens: list[str],
    partial: str,
    ctx: ReplCompletionContext,
) -> list[str]:
    if len(tokens) == 1:
        return _complete_prefix(("answer", "question"))(_session, tokens, partial, ctx)
    if len(tokens) == 2 and ctx.agent_labels:
        p = partial.lower()
        return [lab for lab in ctx.agent_labels if lab.lower().startswith(p)]
    return []


def _complete_clipboard(
    _session: "AgentSession",
    tokens: list[str],
    partial: str,
    ctx: ReplCompletionContext,
) -> list[str]:
    if len(tokens) == 1:
        return _complete_prefix(("copy", "paste"))(_session, tokens, partial, ctx)
    if len(tokens) == 2 and tokens[1].lower() == "copy":
        return _complete_prefix(("all",))(_session, tokens, partial, ctx)
    return []


def _complete_load(
    _session: "AgentSession",
    tokens: list[str],
    partial: str,
    ctx: ReplCompletionContext,
) -> list[str]:
    if len(tokens) == 1:
        return _complete_prefix(("info",))(_session, tokens, partial, ctx)
    return []


def _build_core_specs() -> tuple[ReplCommandSpec, ...]:
    S = ReplCommandSpec
    return (
        S(
            "/quit",
            aliases=("/exit", "/q"),
            match="exact_lower",
            help_line="  /quit · /exit",
            help_order=10,
            handler=_handle_quit,
        ),
        S("/clear", match="exact_lower", help_line="  /clear", help_order=20, handler=_handle_clear),
        S(
            "/compact",
            match="first_token",
            priority=50,
            help_line="  /compact [N% | WORDS]   (default 10%; LLM compresses history, replaces messages)",
            help_order=30,
            handler=lambda s, ln: s._cmd_compact(ln),
        ),
        S(
            "/help",
            aliases=("/?",),
            match="exact_lower",
            help_line="  /help · /?",
            help_order=40,
            handler=_handle_help,
        ),
        S(
            "/usage",
            aliases=("/tokens",),
            match="exact_lower",
            help_line="  /usage · /tokens",
            help_order=50,
            handler=_handle_usage,
        ),
        S(
            "/show",
            match="prefix",
            priority=50,
            help_line="  /show model · /show models · /show reviewer  (see /show help)",
            help_order=60,
            handler=lambda s, ln: s._cmd_show(ln),
            complete=_complete_prefix(("model", "models", "reviewer", "help")),
        ),
        S(
            "/while",
            match="prefix_lower",
            priority=50,
            help_line="  /while …   (agent loop — run manually, not via session_command)",
            help_order=200,
            handler=lambda s, ln: s._cmd_while(ln),
        ),
        S(
            "/use-skills",
            match="prefix_lower",
            priority=80,
            help_line="  /use-skills …   (alias family for /skill)",
            help_order=210,
            handler=lambda s, ln: s._cmd_skill_backcompat(ln),
        ),
        S(
            "/use-skill",
            match="prefix_lower",
            priority=80,
            help_line="",
            help_order=999,
            handler=lambda s, ln: s._cmd_skill_backcompat(ln),
        ),
        S(
            "/reuse-skill",
            match="prefix_lower",
            priority=80,
            help_line="",
            help_order=999,
            handler=lambda s, ln: s._cmd_skill_backcompat(ln),
        ),
        S(
            "/skill",
            match="prefix_lower",
            priority=50,
            help_line="  /skill …   (run manually, not via session_command)",
            help_order=205,
            handler=lambda s, ln: s._cmd_skill(ln),
        ),
        S(
            "/set",
            aliases=("/settings",),
            match="first_token",
            priority=50,
            help_line="  /set …   (try /set help)",
            help_order=70,
            handler=lambda s, ln: s._cmd_settings(ln),
            complete=_complete_set,
        ),
        S(
            "/source",
            match="prefix_lower",
            priority=50,
            help_line="  /source FILE",
            help_order=80,
            handler=lambda s, ln: s._cmd_source(ln),
        ),
        S(
            "/import",
            match="prefix_lower",
            priority=50,
            help_line="  /import FILE",
            help_order=85,
            handler=None,
        ),
        S(
            "/load_context",
            match="prefix_lower",
            priority=90,
            help_line="",
            help_order=999,
            handler=lambda s, ln: s._cmd_load_context(ln),
        ),
        S(
            "/save_context",
            match="prefix_lower",
            priority=90,
            help_line="",
            help_order=999,
            handler=lambda s, ln: s._cmd_save_context(ln),
        ),
        S(
            "/context",
            match="prefix_lower",
            priority=50,
            help_line="  /context load|save|start_log FILE   (aliases: /load_context, /save_context)",
            help_order=90,
            handler=lambda s, ln: s._cmd_context(ln),
            complete=_complete_prefix(("load", "save", "start_log")),
        ),
        S(
            "/unload",
            match="first_token",
            priority=50,
            help_line="  /load FILE.py  ·  /unload  ·  /extensions   (REPL extension modules)",
            help_order=100,
            handler=lambda s, ln: s._cmd_repl_unload(ln),
        ),
        S(
            "/load",
            match="first_token",
            priority=50,
            help_line="",
            help_order=999,
            handler=lambda s, ln: s._cmd_repl_load(ln),
            complete=_complete_load,
        ),
        S(
            "/extensions",
            match="first_token",
            priority=50,
            help_line="",
            help_order=999,
            handler=lambda s, ln: s._cmd_repl_extensions(ln),
        ),
        S(
            "/cd",
            aliases=("/chdir",),
            match="first_token",
            priority=50,
            help_line="  /cd DIR",
            help_order=110,
            handler=lambda s, ln: s._cmd_cd(ln),
        ),
        S(
            "/mcp",
            match="first_token",
            priority=50,
            help_line="  /mcp …          (Model Context Protocol servers — try /mcp help)",
            help_order=120,
            handler=lambda s, ln: s._cmd_mcp(ln),
            complete=_complete_mcp,
        ),
        S(
            "/call_python",
            match="prefix_lower",
            priority=50,
            help_line="  /call_python …",
            help_order=130,
            handler=lambda s, ln: s._cmd_call_python(ln),
        ),
        S(
            "/run_command",
            match="prefix_lower",
            priority=50,
            help_line="  /run_command …",
            help_order=140,
            handler=lambda s, ln: s._cmd_run_command(ln),
        ),
        S(
            "!",
            match="prefix",
            priority=10,
            help_line="  ! CMD",
            help_order=150,
            handler=None,
        ),
        S(
            "/fork_background",
            match="prefix_lower",
            priority=100,
            help_line='  /fork NAME ["cmds"]…  ·  /fork_background NAME ["cmds"]…',
            help_order=160,
            visible=_visible_fork,
            handler=lambda s, ln: s._cmd_fork_background(ln),
        ),
        S(
            "/fork",
            match="prefix_lower",
            priority=90,
            help_line="",
            help_order=999,
            visible=_visible_fork,
            handler=lambda s, ln: s._cmd_fork(ln),
        ),
        S(
            "/kill",
            match="prefix_lower",
            priority=50,
            help_line="  /kill NAME",
            help_order=170,
            visible=_visible_kill,
            handler=None,
            complete=_complete_agent_target,
        ),
        S(
            "/list",
            match="exact_lower",
            help_line="  /list",
            help_order=180,
            visible=_visible_host_list,
            handler=_handle_list,
        ),
        S(
            "/switch",
            match="prefix_lower",
            priority=50,
            help_line="  /switch NAME",
            help_order=190,
            visible=_visible_host_list,
            handler=_handle_switch,
            complete=_complete_agent_target,
        ),
        S(
            "/last_answer",
            match="prefix_lower",
            priority=100,
            help_line="  /last answer|question [NAME]   (aliases: /last_answer, /last_question)",
            help_order=200,
            handler=_handle_last_answer,
            complete=_complete_last,
        ),
        S(
            "/last_question",
            match="prefix_lower",
            priority=100,
            help_line="",
            help_order=999,
            handler=_handle_last_question,
            complete=_complete_last,
        ),
        S(
            "/last",
            match="prefix_lower",
            priority=80,
            help_line="",
            help_order=999,
            handler=lambda s, ln: s._cmd_last(ln),
            complete=_complete_last,
        ),
        S(
            "/clipboard",
            match="prefix_lower",
            priority=50,
            help_line="  /clipboard copy|copy all|paste",
            help_order=210,
            handler=lambda s, ln: s._cmd_clipboard(ln),
            complete=_complete_clipboard,
        ),
        S(
            "/turn",
            match="prefix",
            priority=50,
            help_line=(
                '  /send NAME CMD…  ·  /turn NAME CMD…  (async vs blocking; NAME may be self)\n'
                '  /send NAME "cmd1,cmd2,…"  ·  /turn NAME "cmd1,cmd2,…"'
            ),
            help_order=220,
            visible=_visible_delegate,
            handler=lambda s, ln: s._cmd_turn_to_agent(ln),
            complete=_complete_agent_target,
        ),
        S(
            "/send",
            match="prefix",
            priority=50,
            help_line="",
            help_order=999,
            visible=_visible_delegate,
            handler=lambda s, ln: s._cmd_send_to_agent(ln),
            complete=_complete_agent_target,
        ),
        S(
            "/send",
            match="prefix",
            priority=50,
            help_line="  /send self CMD…  ·  /turn self CMD…  (self only without multi-agent host)",
            help_order=225,
            visible=lambda s: not _visible_delegate(s),
            handler=lambda s, ln: s._cmd_send_to_agent(ln),
            complete=_complete_prefix(("self",)),
        ),
        S(
            "/turn",
            match="prefix",
            priority=50,
            help_line="",
            help_order=999,
            visible=lambda s: not _visible_delegate(s),
            handler=lambda s, ln: s._cmd_turn_to_agent(ln),
            complete=_complete_prefix(("self",)),
        ),
    )


_CORE_SPECS: tuple[ReplCommandSpec, ...] = _build_core_specs()
_DISPATCH_SPECS: tuple[ReplCommandSpec, ...] = tuple(
    sorted(
        (s for s in _CORE_SPECS if s.handler is not None),
        key=lambda s: (-s.priority, -len(s.name), s.name),
    )
)


def find_spec_for_token(token: str, session: "AgentSession") -> Optional[ReplCommandSpec]:
    for spec in _DISPATCH_SPECS:
        if not spec.visible(session):
            continue
        if spec.matches_token(token):
            return spec
    return None


def find_spec_for_line(line: str, session: "AgentSession") -> Optional[ReplCommandSpec]:
    for spec in _DISPATCH_SPECS:
        if not spec.visible(session):
            continue
        if spec.matches_line(line):
            return spec
    return None


def dispatch_repl_command(session: "AgentSession", line: str) -> Optional["SessionLineResult"]:
    """Run a registered slash command, or return ``None`` if not handled."""
    spec = find_spec_for_line(line.strip(), session)
    if spec is None or spec.handler is None:
        return None
    return spec.handler(session, line)


def iter_visible_command_names(session: "AgentSession") -> list[str]:
    names: set[str] = set()
    for spec in _CORE_SPECS:
        if spec.visible(session):
            names.update(spec.all_names())
    names.update(session._repl_extension_commands.keys())
    return sorted(names)


def iter_completion_specs(session: "AgentSession") -> list[ReplCommandSpec]:
    specs: list[ReplCommandSpec] = []
    seen: set[str] = set()
    for spec in _CORE_SPECS:
        if not spec.visible(session):
            continue
        key = spec.name.lower()
        if key in seen:
            continue
        seen.add(key)
        specs.append(spec)
    return specs


def format_repl_help(session: "AgentSession") -> str:
    lines: list[str] = []
    for spec in sorted(_CORE_SPECS, key=lambda s: (s.help_order, s.name)):
        if spec.help_line and spec.visible(session):
            lines.append(spec.help_line)
    ext = session._repl_extension_help_text()
    if ext:
        lines.append(ext.rstrip("\n"))
    return "\n".join(lines) + "\n"


def complete_command_token(
    session: "AgentSession",
    tokens: list[str],
    partial: str,
    ctx: ReplCompletionContext,
) -> list[str]:
    if not tokens:
        p = partial.lower()
        return [n for n in iter_visible_command_names(session) if n.lower().startswith(p)]
    spec = find_spec_for_token(tokens[0], session)
    if spec is None:
        key = tokens[0].lower()
        handler = session._repl_extension_completers.get(key)
        if handler is not None:
            return handler(session, tokens, partial, ctx)
        return []
    if spec.complete is not None:
        return spec.complete(session, tokens, partial, ctx)
    return []
