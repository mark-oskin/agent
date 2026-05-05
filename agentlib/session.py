from __future__ import annotations

import datetime
import json
import os
import shlex
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Callable, Iterable, Optional

from agentlib.sink import emit_sink_scope, sink_emit, sink_print_compat
from agentlib.tools.registry import ToolRegistry
from agentlib.tools.routing import preferred_web_search_tool
from agentlib import prompts as agent_prompts

from .runtime import ConversationTurnDeps, run_agent_conversation_turn
from .settings import AgentSettings


@dataclass
class SessionLineResult:
    output: str = ""
    quit: bool = False


def parse_send_command(line: str) -> Optional[tuple[str, str]]:
    """Parse ``/send AGENT COMMAND...`` → ``(agent_name, command_line)``. Returns ``None`` if invalid."""
    try:
        toks = shlex.split((line or "").strip())
    except ValueError:
        return None
    if len(toks) < 3 or toks[0].lower() != "/send":
        return None
    return toks[1], shlex.join(toks[2:])


class AgentSession:
    """Owns interactive session state; parses and executes REPL lines."""

    def __init__(
        self,
        *,
        settings: AgentSettings,
        verbose: int,
        second_opinion_enabled: bool,
        cloud_ai_enabled: bool,
        save_context_path: Optional[str],
        enabled_tools: AbstractSet[str],
        enabled_toolsets: AbstractSet[str],
        primary_profile,
        reviewer_hosted_profile,
        reviewer_ollama_model: Optional[str],
        skills_map: dict,
        prompt_templates: dict,
        prompt_template_default: str,
        prompt_templates_dir: str,
        skills_dir: str,
        tools_dir: str,
        context_cfg: Optional[dict],
        system_prompt_override: Optional[str],
        system_prompt_path: Optional[str],
        session_prompt_template: Optional[str],
        # injected helpers / callbacks (from agent.py)
        agent_progress: Callable[[str], None],
        fetch_ollama_local_model_names: Callable[[], list[str]],
        format_last_ollama_usage_for_repl: Callable[[], str],
        format_session_primary_llm_line: Callable[[object], str],
        format_session_reviewer_line: Callable[[object, Optional[str]], str],
        print_skill_usage_verbose: Callable[..., None],
        match_skill_detail: Callable[[str, dict], tuple[Optional[str], Optional[str]]],
        ml_select_skill_id: Callable[..., tuple[Optional[str], str]],
        skill_plan_steps: Callable[..., tuple[Optional[list], str]],
        effective_enabled_tools_for_skill: Callable[[AbstractSet[str], dict, Optional[str]], AbstractSet[str]],
        effective_enabled_tools_for_turn: Callable[..., AbstractSet[str]],
        route_requires_websearch: Callable[..., Optional[str]],
        deliverable_skip_mandatory_web: Callable[[str], bool],
        user_wants_written_deliverable: Callable[[str], bool],
        interactive_turn_user_message: Callable[..., str],
        conversation_turn_deps: ConversationTurnDeps,
        save_context_bundle: Callable[..., None],
        load_context_messages: Callable[[str], list],
        registry: ToolRegistry,
        # prefs / persistence
        build_agent_prefs_payload: Callable[..., dict],
        write_agent_prefs_file: Callable[[dict], None],
        agent_prefs_path: Callable[[], str],
        settings_group_keys_lines: Callable[[str], str],
        settings_group_show: Callable[[str], str],
        settings_group_set: Callable[[str, str, str], str],
        settings_group_unset: Callable[[str, str], str],
        settings_get: Callable[[tuple, object], object],
        settings_set: Callable[[tuple, object], None],
        # llm profile helpers for /settings primary/second_opinion llm ...
        LlmProfile_cls,
        default_primary_llm_profile: Callable[[], object],
        describe_llm_profile_short: Callable[[object], str],
        ollama_second_opinion_model: Callable[[], str],
        # thinking helpers
        ollama_request_think_value: Callable[[], object],
        agent_thinking_level: Callable[[], str],
        agent_thinking_enabled_default_false: Callable[[], bool],
        agent_stream_thinking_enabled: Callable[[], bool],
        verbose_ack_message: Callable[[int], str],
        parse_while_repl_tokens: Callable[[list[str]], tuple[int, str, list[str]]],
        call_while_condition_judge: Callable[..., int],
        python_fork_agent: Optional[Callable[..., dict]] = None,
        python_delegate_line: Optional[Callable[..., dict]] = None,
        python_host_command: Optional[Callable[[dict], dict]] = None,
        python_enqueue_line: Optional[Callable[[str, str], dict]] = None,
    ):
        self.settings = settings
        self.verbose = int(verbose)
        self.second_opinion_on = bool(second_opinion_enabled)
        self.cloud_ai_enabled = bool(cloud_ai_enabled)
        self.session_save_path = save_context_path
        self.enabled_tools = set(enabled_tools)
        self.enabled_toolsets = set(enabled_toolsets)
        self.primary_profile = primary_profile
        self.reviewer_hosted_profile = reviewer_hosted_profile
        self.reviewer_ollama_model = reviewer_ollama_model
        self.skills_map = skills_map if isinstance(skills_map, dict) else {}
        self.prompt_templates = prompt_templates if isinstance(prompt_templates, dict) else {}
        self.template_default = (prompt_template_default or "").strip() or "coding"
        self.prompt_templates_dir = prompt_templates_dir
        self.skills_dir = skills_dir
        self.tools_dir = tools_dir
        self.context_cfg = context_cfg if isinstance(context_cfg, dict) else {}
        self.session_system_prompt = system_prompt_override
        self.session_system_prompt_path = system_prompt_path
        self.session_prompt_template = session_prompt_template
        # Persist system_prompt only when the user explicitly overrides/pins/loads it.
        self._system_prompt_explicit = bool(
            (isinstance(self.session_system_prompt_path, str) and self.session_system_prompt_path.strip())
            or (self.session_system_prompt is not None and str(self.session_system_prompt).strip())
        )
        # If the prompt is coming from a selected/default prompt template, treat it as non-explicit.
        if self.session_prompt_template and not (self.session_system_prompt_path or "").strip():
            self._system_prompt_explicit = False

        # Persist prompt templates/default only when explicitly changed in-session.
        self._prompt_templates_explicit = False
        self._prompt_template_default_explicit = False

        self.messages: list = []
        self.last_reuse_skill_id: Optional[str] = None
        # Last normal (non-slash) user line and last structured model answer for /last_* .
        self.repl_last_user_query: Optional[str] = None
        self.repl_last_assistant_answer: Optional[str] = None

        # injected helpers / callbacks
        self._agent_progress = agent_progress
        self._fetch_ollama_local_model_names = fetch_ollama_local_model_names
        self._format_last_ollama_usage_for_repl = format_last_ollama_usage_for_repl
        self._format_session_primary_llm_line = format_session_primary_llm_line
        self._format_session_reviewer_line = format_session_reviewer_line
        self._print_skill_usage_verbose = print_skill_usage_verbose
        self._match_skill_detail = match_skill_detail
        self._ml_select_skill_id = ml_select_skill_id
        self._skill_plan_steps = skill_plan_steps
        self._effective_enabled_tools_for_skill = effective_enabled_tools_for_skill
        self._effective_enabled_tools_for_turn = effective_enabled_tools_for_turn
        self._route_requires_websearch = route_requires_websearch
        self._deliverable_skip_mandatory_web = deliverable_skip_mandatory_web
        self._user_wants_written_deliverable = user_wants_written_deliverable
        self._interactive_turn_user_message = interactive_turn_user_message
        self._conversation_turn_deps = conversation_turn_deps
        self._save_context_bundle = save_context_bundle
        self._load_context_messages = load_context_messages
        self._registry = registry

        self._build_agent_prefs_payload = build_agent_prefs_payload
        self._write_agent_prefs_file = write_agent_prefs_file
        self._agent_prefs_path = agent_prefs_path
        self._settings_group_keys_lines = settings_group_keys_lines
        self._settings_group_show = settings_group_show
        self._settings_group_set = settings_group_set
        self._settings_group_unset = settings_group_unset
        self._settings_get = settings_get
        self._settings_set = settings_set

        self._LlmProfile = LlmProfile_cls
        self._default_primary_llm_profile = default_primary_llm_profile
        self._describe_llm_profile_short = describe_llm_profile_short
        self._ollama_second_opinion_model = ollama_second_opinion_model

        self._ollama_request_think_value = ollama_request_think_value
        self._agent_thinking_level = agent_thinking_level
        self._agent_thinking_enabled_default_false = agent_thinking_enabled_default_false
        self._agent_stream_thinking_enabled = agent_stream_thinking_enabled
        self._verbose_ack_message = verbose_ack_message
        self._parse_while_repl_tokens = parse_while_repl_tokens
        self._call_while_condition_judge = call_while_condition_judge
        # Multi-line /call_python support (buffer until closing quote).
        self._call_python_pending: Optional[str] = None

        self.python_fork_agent = python_fork_agent
        self.python_delegate_line = python_delegate_line
        self.python_host_command = python_host_command
        self.python_enqueue_line = python_enqueue_line

    def _agent_loop_budget(self) -> tuple[int, int, int, int]:
        s = self.settings
        return (
            max(1, s.get_int(("agent", "max_agent_steps"), 30)),
            max(1, s.get_int(("agent", "max_agent_steps_web"), 15)),
            max(1, s.get_int(("agent", "max_tool_calls_web"), 15)),
            max(1, s.get_int(("agent", "max_fetch_page_web"), 15)),
        )

    def host_ctl(self, op: str, arg: Optional[str] = None) -> dict:
        """
        Multi-agent host RPC (optional). Used by ``/list``, ``/switch``, ``/last_answer``,
        ``/last_question``, and ``/call_python`` helpers when a host (e.g. ``agent_tui``) wires
        ``python_host_command``.
        """
        h = self.python_host_command
        want_local_last = (
            h is None
            and op in ("last_answer", "last_question")
            and (arg is None or not str(arg).strip())
        )
        if want_local_last:
            if op == "last_answer":
                v = self.repl_last_assistant_answer
                hint = "(no last assistant answer yet)"
            else:
                v = self.repl_last_user_query
                hint = "(no last user question yet)"
            text = v.strip() if isinstance(v, str) and v.strip() else hint
            return {"ok": True, "text": text}
        if h is None:
            return {
                "ok": False,
                "error": "Multi-agent host not available (e.g. run agent_tui.py for /list and /switch).",
            }
        try:
            return h({"op": op, "arg": arg, "session": self})
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _sink_host_ctl_result(self, r: dict) -> SessionLineResult:
        if r.get("ok"):
            sink_print_compat(r.get("text") or "")
        else:
            sink_print_compat(r.get("error") or "failed")
        return SessionLineResult()

    def _cmd_send_to_agent(self, s: str) -> SessionLineResult:
        """Forward one line to another agent (async enqueue when ``python_enqueue_line`` is set)."""
        parsed = parse_send_command(s)
        if parsed is None:
            sink_print_compat('Usage: /send AGENT COMMAND...')
            return SessionLineResult()
        agent_name, cmd = parsed
        eq = self.python_enqueue_line
        if eq is not None:
            try:
                r = eq(agent_name, cmd)
            except BaseException as e:
                sink_print_compat(f"/send: {type(e).__name__}: {e}")
                return SessionLineResult()
            if isinstance(r, dict) and r.get("ok"):
                lab = str(r.get("label") or agent_name)
                if r.get("queued"):
                    sink_print_compat(f"[queued → {lab}] {cmd}")
                else:
                    sink_print_compat(f"[started → {lab}] {cmd}")
            elif isinstance(r, dict):
                sink_print_compat(str(r.get("error") or "/send failed"))
            else:
                sink_print_compat("[send] scheduled.")
            return SessionLineResult()
        dl = self.python_delegate_line
        if dl is None:
            sink_print_compat(
                "/send requires a multi-agent host (e.g. agent_tui.py); enqueue/delegate hook not configured."
            )
            return SessionLineResult()
        try:
            res = dl(agent_name, cmd)
        except BaseException as e:
            sink_print_compat(f"/send: {type(e).__name__}: {e}")
            return SessionLineResult()
        if isinstance(res, dict):
            if res.get("type") == "command":
                out = res.get("output") or ""
                if isinstance(out, str) and out.strip():
                    sink_print_compat(out.strip())
                else:
                    sink_print_compat(f"[sent → {agent_name}] command finished.")
            elif res.get("type") == "turn":
                if res.get("answered"):
                    ans = res.get("answer")
                    n = len(ans) if isinstance(ans, str) else 0
                    sink_print_compat(f"[sent → {agent_name}] turn answered ({n} chars).")
                else:
                    sink_print_compat(f"[sent → {agent_name}] turn finished.")
            else:
                sink_print_compat(f"[sent → {agent_name}] done.")
        else:
            sink_print_compat(f"[sent → {agent_name}] done.")
        return SessionLineResult()

    def execute_line(self, line: str, *, emit: Optional[Callable[[dict], None]] = None) -> dict:
        """
        Execute one REPL line.

        - When `emit` is provided, an emit sink is installed for this line so thinking, progress,
          tool output, and REPL command text stream incrementally via typed emit events (same schema as before).
        - Returns a structured dict for embedding callers (CLI can ignore most fields).
        """
        raw_line = (line or "").rstrip("\n")
        s0 = raw_line.strip()
        if not s0:
            return {"type": "noop", "quit": False}

        # Multi-line /call_python -c: allow pasting code with literal newlines by buffering
        # until shlex sees a closing quote.
        if self._call_python_pending is not None:
            combined = self._call_python_pending + "\n" + raw_line
            kind, payload = self._split_call_python_rest(combined.strip())
            if kind == "error" and isinstance(payload, str) and "No closing quotation" in payload:
                self._call_python_pending = combined
                return {"type": "command", "quit": False, "output": ""}
            self._call_python_pending = None
            s0 = combined.strip()
        else:
            # Start buffering if the user opened a quote but didn't close it yet.
            if s0.lower().startswith("/call_python"):
                kind, payload = self._split_call_python_rest(s0)
                if kind == "error" and isinstance(payload, str) and "No closing quotation" in payload:
                    self._call_python_pending = raw_line
                    return {"type": "command", "quit": False, "output": ""}

        if emit is None:
            # Preserve legacy behavior (prints inside handlers).
            if s0.startswith("/"):
                res = self._execute_command_line(s0)
                return {"type": "command", "quit": bool(res.quit), "output": res.output}
            if s0.startswith("!"):
                res = self._cmd_run_shell_bang(s0)
                return {"type": "command", "quit": bool(res.quit), "output": res.output}
            answered, final_answer = self._execute_user_request(s0)
            self.repl_last_user_query = s0
            self.repl_last_assistant_answer = (
                final_answer.strip()
                if isinstance(final_answer, str) and final_answer.strip()
                else None
            )
            return {
                "type": "turn",
                "quit": False,
                "answered": bool(answered),
                "answer": final_answer,
            }

        with emit_sink_scope(emit):
            if s0.startswith("/"):
                res = self._execute_command_line(s0)
                payload = {"type": "command", "quit": bool(res.quit), "output": res.output}
            elif s0.startswith("!"):
                res = self._cmd_run_shell_bang(s0)
                payload = {"type": "command", "quit": bool(res.quit), "output": res.output}
            else:
                answered, final_answer = self._execute_user_request(s0)
                self.repl_last_user_query = s0
                self.repl_last_assistant_answer = (
                    final_answer.strip()
                    if isinstance(final_answer, str) and final_answer.strip()
                    else None
                )
                payload = {
                    "type": "turn",
                    "quit": False,
                    "answered": bool(answered),
                    "answer": final_answer,
                }
            return payload

    def _today_str(self) -> str:
        return datetime.date.today().strftime("%Y-%m-%d (%A)")

    def _execute_user_request(self, user_query: str) -> tuple[bool, Optional[str]]:
        """One normal REPL turn: append messages and run the agent loop."""
        today_str = self._today_str()
        deliverable_wanted = self._user_wants_written_deliverable(user_query)
        sid0, tr0 = self._match_skill_detail(user_query, self.skills_map)
        et_turn0 = self._effective_enabled_tools_for_skill(
            frozenset(self.enabled_tools), self.skills_map, sid0
        )
        et_turn = self._effective_enabled_tools_for_turn(
            base_enabled_tools=et_turn0,
            enabled_toolsets=self.enabled_toolsets,
            user_query=user_query,
        )
        if self.verbose >= 1:
            d0 = (
                f"trigger match: longest substring {tr0!r} (skill {sid0!r})"
                if sid0 and tr0
                else "trigger match: no skill (no trigger substring matched)"
            )
            self._print_skill_usage_verbose(
                self.verbose,
                source="repl",
                skill_id=sid0,
                base_tools=self.enabled_tools,
                effective_tools=et_turn,
                detail=d0,
            )
        sprompt0 = (self.skills_map.get(sid0) or {}).get("prompt") if sid0 else None
        router_query = self._route_requires_websearch(
            user_query,
            today_str,
            self.primary_profile,
            et_turn,
            transcript_messages=self.messages,
        )
        if self._deliverable_skip_mandatory_web(user_query):
            router_query = None
        web_required = bool(router_query)
        turn_msg = self._interactive_turn_user_message(
            user_query,
            today_str,
            self.second_opinion_on,
            self.cloud_ai_enabled,
            primary_profile=self.primary_profile,
            reviewer_ollama_model=self.reviewer_ollama_model,
            reviewer_hosted_profile=self.reviewer_hosted_profile,
            enabled_tools=et_turn,
            system_instruction_override=self.session_system_prompt,
            skill_suffix=sprompt0,
        )
        self.messages.append({"role": "user", "content": turn_msg})
        _mw = preferred_web_search_tool(et_turn)
        if router_query and _mw:
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Before answering, you MUST call the tool {_mw}.\n"
                        "Respond with JSON only in tool_call form.\n"
                        f'Suggested query: "{router_query}"'
                    ),
                }
            )
        ms, msw, mtcw, mfpw = self._agent_loop_budget()
        answered, final_answer = run_agent_conversation_turn(
            self.messages,
            user_query,
            today_str,
            self._conversation_turn_deps,
            web_required=web_required,
            deliverable_wanted=deliverable_wanted,
            verbose=self.verbose,
            second_opinion_enabled=self.second_opinion_on,
            cloud_ai_enabled=self.cloud_ai_enabled,
            primary_profile=self.primary_profile,
            reviewer_hosted_profile=self.reviewer_hosted_profile,
            reviewer_ollama_model=self.reviewer_ollama_model,
            enabled_tools=et_turn,
            interactive_tool_recovery=True,
            context_cfg=self.context_cfg,
            print_answer=False,
            max_agent_steps=ms,
            max_agent_steps_web=msw,
            max_tool_calls_web=mtcw,
            max_fetch_page_web=mfpw,
        )
        if self.session_save_path:
            try:
                self._save_context_bundle(
                    self.session_save_path,
                    self.messages,
                    user_query,
                    final_answer,
                    answered,
                )
            except OSError as e:
                sink_emit({"type": "warning", "text": f"Warning: could not save context: {e}"})
        return bool(answered), final_answer

    def _run_with_selected_skill(
        self, req: str, sid: str, *, source: str, selection_rationale: str
    ) -> None:
        self.last_reuse_skill_id = sid
        src = (source or "").strip().lower()
        if src == "reuse":
            self._agent_progress("/skill reuse: using stored skill; starting…")
        elif src == "explicit":
            self._agent_progress("/skill: explicit skill selected; starting…")
        else:
            self._agent_progress("/skill auto: skill selected; starting…")
        et_turn0 = self._effective_enabled_tools_for_skill(
            frozenset(self.enabled_tools), self.skills_map, sid
        )
        et_turn = self._effective_enabled_tools_for_turn(
            base_enabled_tools=et_turn0,
            enabled_toolsets=self.enabled_toolsets,
            user_query=req,
        )
        rec = self.skills_map.get(sid) or {}
        skill_prompt = (rec.get("prompt") or "").strip() if isinstance(rec, dict) else ""
        if src == "reuse":
            sink_print_compat(
                f"/skill reuse: using skill {sid!r} (model skill selection skipped). "
                f"{selection_rationale}".strip()
            )
        elif src == "explicit":
            sink_print_compat(f"/skill: using skill {sid!r}. {selection_rationale}".strip())
        else:
            sink_print_compat(f"/skill auto selected {sid!r}. {selection_rationale}".strip())
        if self.verbose >= 1:
            self._print_skill_usage_verbose(
                self.verbose,
                source=f"skill_{src or 'auto'}",
                skill_id=sid,
                base_tools=self.enabled_tools,
                effective_tools=et_turn,
                detail=(
                    "reuse: same skill id as last /skill auto|reuse|<id>"
                    if src == "reuse"
                    else (
                        f"explicit skill id: {sid!r}"
                        if src == "explicit"
                        else f"model skill_id (not trigger): rationale={selection_rationale!r}"
                    )
                ),
            )
        today_str = self._today_str()
        deliverable_wanted = self._user_wants_written_deliverable(req)
        router_query = self._route_requires_websearch(
            req,
            today_str,
            self.primary_profile,
            et_turn,
            transcript_messages=self.messages,
        )
        if self._deliverable_skip_mandatory_web(req):
            router_query = None
        web_required = bool(router_query)
        steps, raw_plan = self._skill_plan_steps(
            user_request=req,
            today_str=today_str,
            skill_id=sid,
            skills_map=self.skills_map,
            primary_profile=self.primary_profile,
            _enabled_tools=et_turn,
            verbose=self.verbose,
            _system_prompt_override=self.session_system_prompt,
        )
        if steps:
            wf = ((rec.get("workflow") or {}) if isinstance(rec, dict) else {}) or {}
            step_prompt = (wf.get("step_prompt") or "").strip()
            sink_print_compat(f"Skill workflow: executing {len(steps)} step(s).", flush=True)
            self._agent_progress(f"Running {len(steps)}-step skill workflow…")
            if self.verbose >= 1:
                rp = raw_plan or ""
                cap = 1200
                preview = rp if len(rp) <= cap else rp[:cap] + "…"
                sink_print_compat(f"[*] [skills:planner] raw ({len(rp)} chars): {preview}")
            step_answers: list[str] = []
            for i, st in enumerate(steps, start=1):
                title = st.get("title") or f"step {i}"
                details = st.get("details") or ""
                success = st.get("success") or ""
                step_user = (
                    f"{req}\n\n"
                    f"Step {i}/{len(steps)}: {title}\n"
                    + (f"Details: {details}\n" if details else "")
                    + (f"Success: {success}\n" if success else "")
                    + ("\n" + step_prompt if step_prompt else "")
                )
                et_step = self._effective_enabled_tools_for_skill(
                    frozenset(self.enabled_tools), self.skills_map, sid
                )
                tit_one = (title or "")[:120]
                if len(title or "") > 120:
                    tit_one += "…"
                self._agent_progress(f"Workflow step {i}/{len(steps)}: {tit_one}")
                sprompt0 = skill_prompt
                turn_msg = self._interactive_turn_user_message(
                    step_user,
                    today_str,
                    self.second_opinion_on,
                    self.cloud_ai_enabled,
                    primary_profile=self.primary_profile,
                    reviewer_ollama_model=self.reviewer_ollama_model,
                    reviewer_hosted_profile=self.reviewer_hosted_profile,
                    enabled_tools=et_step,
                    system_instruction_override=self.session_system_prompt,
                    skill_suffix=sprompt0,
                )
                self.messages.append({"role": "user", "content": turn_msg})
                _mw_step = preferred_web_search_tool(et_step)
                if router_query and _mw_step and i == 1:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Before answering, you MUST call the tool {_mw_step}.\n"
                                "Respond with JSON only in tool_call form.\n"
                                f'Suggested query: "{router_query}"'
                            ),
                        }
                    )
                ms, msw, mtcw, mfpw = self._agent_loop_budget()
                answered, final_answer = run_agent_conversation_turn(
                    self.messages,
                    step_user,
                    today_str,
                    self._conversation_turn_deps,
                    web_required=web_required if i == 1 else False,
                    deliverable_wanted=deliverable_wanted,
                    verbose=self.verbose,
                    second_opinion_enabled=self.second_opinion_on,
                    cloud_ai_enabled=self.cloud_ai_enabled,
                    primary_profile=self.primary_profile,
                    reviewer_hosted_profile=self.reviewer_hosted_profile,
                    reviewer_ollama_model=self.reviewer_ollama_model,
                    enabled_tools=et_step,
                    interactive_tool_recovery=True,
                    context_cfg=self.context_cfg,
                    print_answer=False,
                    max_agent_steps=ms,
                    max_agent_steps_web=msw,
                    max_tool_calls_web=mtcw,
                    max_fetch_page_web=mfpw,
                )
                self._agent_progress(f"Step {i}/{len(steps)} finished.")
                if final_answer:
                    step_answers.append(final_answer)
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps({"action": "answer", "answer": final_answer}),
                        }
                    )
            if step_answers:
                sink_print_compat(step_answers[-1])
            return

        self._agent_progress("Running a single agent turn with the selected skill…")
        turn_msg = self._interactive_turn_user_message(
            req,
            today_str,
            self.second_opinion_on,
            self.cloud_ai_enabled,
            primary_profile=self.primary_profile,
            reviewer_ollama_model=self.reviewer_ollama_model,
            reviewer_hosted_profile=self.reviewer_hosted_profile,
            enabled_tools=et_turn,
            system_instruction_override=self.session_system_prompt,
            skill_suffix=skill_prompt,
        )
        self.messages.append({"role": "user", "content": turn_msg})
        _mw2 = preferred_web_search_tool(et_turn)
        if router_query and _mw2:
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Before answering, you MUST call the tool {_mw2}.\n"
                        "Respond with JSON only in tool_call form.\n"
                        f'Suggested query: "{router_query}"'
                    ),
                }
            )
        ms, msw, mtcw, mfpw = self._agent_loop_budget()
        run_agent_conversation_turn(
            self.messages,
            req,
            today_str,
            self._conversation_turn_deps,
            web_required=web_required,
            deliverable_wanted=deliverable_wanted,
            verbose=self.verbose,
            second_opinion_enabled=self.second_opinion_on,
            cloud_ai_enabled=self.cloud_ai_enabled,
            primary_profile=self.primary_profile,
            reviewer_hosted_profile=self.reviewer_hosted_profile,
            reviewer_ollama_model=self.reviewer_ollama_model,
            enabled_tools=et_turn,
            interactive_tool_recovery=True,
            context_cfg=self.context_cfg,
            max_agent_steps=ms,
            max_agent_steps_web=msw,
            max_tool_calls_web=mtcw,
            max_fetch_page_web=mfpw,
        )

    def _execute_command_line(self, s: str) -> SessionLineResult:
        low = s.lower()
        cmd = (low.split(None, 1)[0] if low.strip() else "")
        if low in ("/quit", "/exit", "/q"):
            return SessionLineResult(quit=True)
        if low == "/clear":
            self.messages.clear()
            self.last_reuse_skill_id = None
            self.repl_last_user_query = None
            self.repl_last_assistant_answer = None
            sink_print_compat("Context cleared (including stored skill for /skill reuse).")
            return SessionLineResult()
        if low in ("/usage", "/tokens"):
            sink_print_compat(self._format_last_ollama_usage_for_repl())
            return SessionLineResult()
        if s.startswith("/show"):
            return self._cmd_show(s)
        if s.startswith("/while"):
            return self._cmd_while(s)
        if low.startswith("/skill"):
            return self._cmd_skill(s)
        if low.startswith("/use-skills") or low.startswith("/use-skill") or low.startswith("/reuse-skill"):
            return self._cmd_skill_backcompat(s)
        if cmd in ("/set", "/settings"):
            return self._cmd_settings(s)
        if low.startswith("/source"):
            return self._cmd_source(s)
        if low.startswith("/load_context"):
            return self._cmd_load_context(s)
        if low.startswith("/save_context"):
            return self._cmd_save_context(s)
        if low.startswith("/run_command"):
            return self._cmd_run_command(s)
        if low.startswith("/call_python"):
            return self._cmd_call_python(s)
        if low == "/list":
            return self._sink_host_ctl_result(self.host_ctl("list_agents"))
        if low.startswith("/switch"):
            try:
                parts = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/switch: {e}")
                return SessionLineResult()
            if len(parts) < 2:
                sink_print_compat("Usage: /switch AGENT_LABEL")
                return SessionLineResult()
            return self._sink_host_ctl_result(self.host_ctl("switch", parts[1]))
        if low.startswith("/last_answer"):
            try:
                parts = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/last_answer: {e}")
                return SessionLineResult()
            arg = parts[1] if len(parts) > 1 else None
            return self._sink_host_ctl_result(self.host_ctl("last_answer", arg))
        if low.startswith("/last_question"):
            try:
                parts = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/last_question: {e}")
                return SessionLineResult()
            arg = parts[1] if len(parts) > 1 else None
            return self._sink_host_ctl_result(self.host_ctl("last_question", arg))
        if s.startswith("/send"):
            return self._cmd_send_to_agent(s)
        if low in ("/help", "/?"):
            ma = ""
            if self.python_fork_agent is not None:
                ma = (
                    "  /fork NAME [\"cmd1,cmd2\"]           Fork lane from history; switch to new lane\n"
                    "  /fork_background NAME [\"cmd1,cmd2\"]   Fork lane without switching sidebar focus\n"
                )
            tui_kill = ""
            if (
                self.python_fork_agent is not None
                or self.python_host_command is not None
                or self.python_enqueue_line is not None
            ):
                tui_kill = "  /kill NAME                 Close an agent lane by label\n"
            host_extras = ""
            if self.python_host_command is not None:
                host_extras = (
                    "  /list                       Active agents (* = focused)\n"
                    "  /switch NAME                Focus agent by label\n"
                )
            snap_extras = (
                "  /last_answer [NAME]         Last model answer (this agent or NAME)\n"
                "  /last_question [NAME]       Last user question sent to the model\n"
            )
            delegate_extras = ""
            if self.python_enqueue_line is not None or self.python_delegate_line is not None:
                delegate_extras = (
                    "  /send NAME CMD...           Run CMD on another agent without blocking this one\n"
                )
            sink_print_compat(
                "Commands:\n"
                "  /quit                    Exit\n"
                "  /clear                   Clear in-memory conversation\n"
                "  /help                    Help\n"
                "  /usage                   Last local Ollama usage\n"
                "  /show ...                Show current state (try /show help)\n"
                "  /skill ...               Skills (try /skill help)\n"
                "  /while ...               Loops (try /while help)\n"
                "  /set ...                 Configuration (try /set help)\n"
                "  /source <file>           Read commands/prompts from file\n"
                "  /load_context <file>     Replace session messages from JSON\n"
                "  /save_context <file>     Write session JSON; set auto-save path\n"
                "  /call_python ...         Run Python in-process (try /call_python help)\n"
                "  /run_command ...        Run shell command (try /run_command help); shorthand: ! CMD\n"
                + ma
                + tui_kill
                + host_extras
                + snap_extras
                + delegate_extras
            )
            return SessionLineResult()
        sink_print_compat(f"Unknown command {s.split()[0]!r}. Try /help.")
        return SessionLineResult()

    def _cmd_source(self, s: str) -> SessionLineResult:
        """
        Read a file of commands/prompts and execute them line-by-line.

        Similar to bash `source`: each non-empty line is processed as if typed into the REPL.
        """
        try:
            toks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/source: {e}")
            return SessionLineResult()
        if len(toks) < 2:
            sink_print_compat("Usage: /source <file>")
            return SessionLineResult()
        path = os.path.expanduser(" ".join(toks[1:]).strip())
        if not path:
            sink_print_compat("Usage: /source <file>")
            return SessionLineResult()
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError as e:
            sink_print_compat(f"/source error: {e}")
            return SessionLineResult()

        executed = 0
        for raw in lines:
            line = (raw or "").rstrip("\n")
            if not line.strip():
                continue
            executed += 1
            res = self.execute_line(line)
            if bool((res or {}).get("quit", False)):
                return SessionLineResult(quit=True)
        sink_print_compat(f"Sourced {executed} line(s) from {path!r}.")
        return SessionLineResult()

    def _repl_shell_run(self, cmd: str) -> SessionLineResult:
        cmd = (cmd or "").strip()
        if not cmd:
            sink_print_compat("/run_command: missing command.")
            return SessionLineResult()
        from agentlib.tools import builtins as tool_builtins

        sink_print_compat(tool_builtins.run_command(cmd))
        return SessionLineResult()

    def _cmd_run_shell_bang(self, s: str) -> SessionLineResult:
        """``! COMMAND`` → same shell execution as ``/run_command COMMAND``."""
        t = (s or "").strip()
        if not t.startswith("!"):
            sink_print_compat("Internal error: expected line starting with '!'.")
            return SessionLineResult()
        cmd = t[1:].lstrip()
        if not cmd:
            sink_print_compat(
                "Usage: ! <shell command>\n"
                "(same as /run_command). Try /run_command help."
            )
            return SessionLineResult()
        return self._repl_shell_run(cmd)

    def _cmd_run_command(self, s: str) -> SessionLineResult:
        """Run a local shell command (``run_command`` tool backend; shell=True)."""
        t = (s or "").strip()
        low = t.lower()
        prefix = "/run_command"
        if not low.startswith(prefix):
            sink_print_compat("/run_command: invalid invocation.")
            return SessionLineResult()
        rest = t[len(prefix) :].lstrip()
        if not rest or rest.lower() in ("help", "-h", "--help", "-?"):
            sink_print_compat(
                "/run_command — run a shell command on this machine (subprocess, shell=True)\n\n"
                "Usage:\n"
                "  /run_command help\n"
                "  /run_command COMMAND       Everything after the command name is passed to your shell\n\n"
                "Shorthand:\n"
                "  ! COMMAND                  Same as /run_command COMMAND\n\n"
                "Uses the same backend as the agent run_command tool; local/trusted use only."
            )
            return SessionLineResult()
        return self._repl_shell_run(rest)

    def _split_call_python_rest(self, s: str) -> tuple[str, Optional[str]]:
        """Return (kind, payload): help | error | code | file."""
        prefix = "/call_python"
        t = (s or "").strip()
        low = t.lower()
        if not low.startswith(prefix):
            return ("bad", None)
        rest = t[len(prefix) :].strip()
        if not rest or rest.lower() in ("help", "-h", "--help", "-?"):
            return ("help", None)
        try:
            parts = shlex.split(rest)
        except ValueError as e:
            return ("error", str(e))
        if not parts:
            return ("help", None)
        if parts[0] == "-c":
            if len(parts) < 2:
                return ("error", "/call_python -c requires Python source")
            # Allow multi-line code to be passed as a single CLI argument by embedding
            # `\n` sequences inside quotes: /call_python -c "line1\nline2".
            # Decode common escapes so users don't need literal newlines (which the REPL
            # would interpret as separate commands).
            raw = " ".join(parts[1:])
            try:
                decoded = bytes(raw, "utf-8").decode("unicode_escape")
            except Exception:
                decoded = raw
            return ("code", decoded)
        return ("file", parts[0])

    def _cmd_call_python(self, s: str) -> SessionLineResult:
        """
        Execute Python in this interpreter (full ``__builtins__`` — trusted users only).

        Injected globals: ``ai``, ``fork_agent``, ``list_agents``, ``switch_agent``, ``last_answer``,
        ``last_question``, ``session`` (this AgentSession),
        ``print`` (routes through emit/sink like other REPL output).

        ``ai(line)`` runs ``execute_line(line)`` on this session (LLM turns and ``/`` commands).
        ``ai(line, agent_name)`` forwards to ``python_delegate_line`` when configured (multi-agent UIs).

        ``fork_agent(name[, commands])`` calls ``python_fork_agent`` when configured.

        ``send(agent_name, cmd)`` forwards ``cmd`` to another lane asynchronously when
        ``python_enqueue_line`` is wired; otherwise falls back to synchronous ``python_delegate_line``.

        ``list_agents()``, ``switch_agent(name)``, ``last_answer(name=None)``, ``last_question(name=None)``
        call ``session.host_ctl(...)`` when ``python_host_command`` is wired (e.g. ``agent_tui``).
        """
        kind, payload = self._split_call_python_rest(s)
        if kind == "help":
            sink_print_compat(
                "/call_python — run Python in the agent process\n\n"
                "Usage:\n"
                "  /call_python help\n"
                "  /call_python -c CODE          Python source (quote for spaces; supports \\n escapes)\n"
                "  /call_python PATH.py          UTF-8 script file\n\n"
                "Multi-line:\n"
                "  You can paste multi-line Python by opening a quote after -c and closing it on a later line.\n\n"
                "Globals:\n"
                "  ai(cmd)                       Same as typing ``cmd`` here (LLM or ``/command``).\n"
                "  ai(cmd, agent_name)           Target another agent when multi-agent hooks exist.\n"
                "  fork_agent(name [, cmds])     Fork a lane when ``python_fork_agent`` is wired.\n"
                "  send(name, cmd)               Forward cmd to another agent (async when host supports it).\n"
                "  list_agents()                 Snapshots lanes when ``python_host_command`` is wired.\n"
                "  switch_agent(name)            Focus lane by label (host).\n"
                "  last_answer([name])           Last model answer for this lane or NAME.\n"
                "  last_question([name])         Last user question for this lane or NAME.\n"
                "  session.host_ctl(op, arg)     Low-level host RPC (same ops as slash commands).\n"
                "  session                       This AgentSession.\n"
                "  print(...)                    Routed like REPL output (emit when streaming).\n"
            )
            return SessionLineResult()
        if kind == "bad":
            sink_print_compat("/call_python: invalid invocation.")
            return SessionLineResult()
        if kind == "error":
            sink_print_compat(f"/call_python: {payload}")
            return SessionLineResult()

        session = self

        def ai(cmd: str, agent_name: Optional[str] = None) -> dict:
            line = (cmd or "").strip()
            if not line:
                return {"type": "noop", "quit": False}
            sub = (agent_name or "").strip()
            if sub:
                dl = session.python_delegate_line
                if dl is None:
                    sink_print_compat(
                        "ai(..., agent_name) requires a multi-agent host "
                        "(e.g. agent_tui.py); delegate hook not configured."
                    )
                    return {"type": "command", "quit": False, "output": "delegate unavailable"}
                return dl(sub, line)
            return session.execute_line(line)

        def fork_agent(name: str, commands: Optional[Iterable[str]] = None) -> dict:
            hook = session.python_fork_agent
            cmds = None if commands is None else list(commands)
            if hook is None:
                sink_print_compat(
                    "fork_agent() requires a multi-agent host (e.g. agent_tui.py); hook not configured."
                )
                return {"type": "fork", "ok": False, "error": "fork unavailable"}
            return hook(str(name).strip(), cmds)

        def list_agents() -> dict:
            return session.host_ctl("list_agents")

        def switch_agent(name: str) -> dict:
            return session.host_ctl("switch", str(name).strip())

        def last_answer(agent_name: Optional[str] = None) -> dict:
            a = (agent_name or "").strip()
            return session.host_ctl("last_answer", a if a else None)

        def last_question(agent_name: Optional[str] = None) -> dict:
            a = (agent_name or "").strip()
            return session.host_ctl("last_question", a if a else None)

        def send(agent_name: str, cmd: str) -> dict:
            nm = str(agent_name or "").strip()
            line = (cmd or "").strip()
            if not nm:
                sink_print_compat("send() requires a non-empty agent name.")
                return {"type": "command", "quit": False, "output": "bad send"}
            if not line:
                sink_print_compat("send() requires a non-empty command.")
                return {"type": "command", "quit": False, "output": "bad send"}
            eq = session.python_enqueue_line
            if eq is not None:
                try:
                    return eq(nm, line)
                except BaseException as e:
                    return {"ok": False, "error": f"{type(e).__name__}: {e}"}
            dl = session.python_delegate_line
            if dl is None:
                sink_print_compat(
                    "send() requires a multi-agent host (e.g. agent_tui.py); enqueue/delegate not configured."
                )
                return {"type": "command", "quit": False, "output": "delegate unavailable"}
            return dl(nm, line)

        g = {
            "__builtins__": __builtins__,
            "__name__": "__call_python__",
            "ai": ai,
            "fork_agent": fork_agent,
            "send": send,
            "list_agents": list_agents,
            "switch_agent": switch_agent,
            "last_answer": last_answer,
            "last_question": last_question,
            "session": session,
            "print": sink_print_compat,
        }

        try:
            if kind == "code":
                assert payload is not None
                filename = "<call_python -c>"
                src = payload
            else:
                assert payload is not None
                path = Path(payload).expanduser()
                if not path.is_file():
                    sink_print_compat(f"/call_python: not a file: {path}")
                    return SessionLineResult()
                filename = str(path.resolve())
                src = path.read_text(encoding="utf-8")
            code = compile(src, filename, "exec")
            # Use the same mapping for globals and locals so imports and top-level defs
            # live in ``g``. With ``exec(code, g, {})``, CPython treats the code like a class
            # body and binds module-level imports into the empty locals dict, so functions
            # (whose __globals__ is ``g``) see NameError for stdlib/third-party names.
            exec(code, g, g)
        except BaseException:
            sink_print_compat(traceback.format_exc())
            return SessionLineResult()

        return SessionLineResult()

    def _cmd_show(self, s: str) -> SessionLineResult:
        try:
            toks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/show: {e}")
            return SessionLineResult()
        if len(toks) < 2 or toks[1].lower() in ("help", "-h", "--help"):
            sink_print_compat(
                "Usage:\n"
                "  /show model      Primary LLM in use (Ollama or hosted)\n"
                "  /show models     Local Ollama models available on this machine\n"
                "  /show reviewer   Second-opinion reviewer model\n"
                "\n"
                "Settings that already have a show line: /set tools, /set context show, "
                "/set thinking show, /set system_prompt show, /set prompt_template show, "
                "/set ollama|openai|agent show"
            )
            return SessionLineResult()
        sub = toks[1].lower().replace("-", "_")
        if sub in ("models", "local_models"):
            try:
                names = self._fetch_ollama_local_model_names()
                sink_print_compat("\n".join(names) if names else "(no models returned)")
            except Exception as e:
                sink_print_compat(f"/show models error: {e}")
            return SessionLineResult()
        if sub in ("model", "primary", "llm"):
            sink_print_compat(f"Primary LLM: {self._format_session_primary_llm_line(self.primary_profile)}")
            return SessionLineResult()
        if sub in ("reviewer", "second_opinion", "2nd"):
            sink_print_compat(
                "Second-opinion reviewer: "
                + self._format_session_reviewer_line(self.reviewer_hosted_profile, self.reviewer_ollama_model)
            )
            return SessionLineResult()
        sink_print_compat("Unknown /show topic. Try: /show model, /show models, or /show reviewer")
        return SessionLineResult()

    def _cmd_while(self, s: str) -> SessionLineResult:
        try:
            wtoks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/while: {e}")
            return SessionLineResult()
        if len(wtoks) == 1 or (len(wtoks) == 2 and wtoks[1].lower() in ("help", "-h", "--help")):
            sink_print_compat(
                "Usage:\n"
                "  /while [--max N] <condition> do <action>\n"
                "  <condition> and <action> are shlex-quoted; use double or single quotes (use the other kind for quotes inside).\n"
                "  Like C while (CONDITION) { … }: the judge returns whether CONDITION is TRUE or FALSE right now.\n"
                "    1 = TRUE — stay in the loop (run <action>, then re-check).\n"
                "    0 = FALSE — exit the loop (do not run <action>).\n"
                "  Default --max is 50 iterations (each iteration: one judge + at most one body).\n"
                "\n"
                "  Body: one or more comma-separated quoted prompts (shlex; add a space before each comma if your\n"
                "  shell glues commas, e.g.  \"a\" , \"b\"  or  \"a\", \"b\"  both work).\n"
                "  Examples:\n"
                '    /while "pytest is still failing" do "fix from output and run pytest"\n'
                "    /while 'work remains' do 'step A', 'step B', 'step C'\n"
                "    /while --max 10 'server not yet returning 200' do 'patch and curl until OK'\n"
            )
            return SessionLineResult()
        try:
            max_while, while_cond, body_prompts = self._parse_while_repl_tokens(wtoks)
        except ValueError as e:
            sink_print_compat(f"/while: {e}")
            return SessionLineResult()
        try:
            abort_while = False
            for wit in range(1, max_while + 1):
                try:
                    bit = self._call_while_condition_judge(
                        while_cond,
                        self.messages,
                        primary_profile=self.primary_profile,
                        verbose=self.verbose,
                    )
                except KeyboardInterrupt:
                    self._agent_progress("Cancelled /while (condition check).")
                    sink_print_compat("\n[Cancelled]\n")
                    break
                if bit == 0:
                    sink_print_compat(
                        f"/while: condition false (judge returned 0). Exiting after check {wit}/{max_while}."
                    )
                    break
                n_steps = len(body_prompts)
                for si, bp in enumerate(body_prompts, start=1):
                    uq = f"[ /while iteration {wit}/{max_while} step {si}/{n_steps} ]\n{bp}"
                    self._agent_progress(f"/while: iteration {wit}/{max_while} step {si}/{n_steps}")
                    try:
                        self._execute_user_request(uq)
                    except KeyboardInterrupt:
                        self._agent_progress("Cancelled /while (body).")
                        sink_print_compat("\n[Cancelled]\n")
                        abort_while = True
                        break
                if abort_while:
                    break
            else:
                sink_print_compat(f"/while: reached --max {max_while} without judge returning 0 (exit).")
        except Exception as e:
            sink_print_compat(f"/while error: {e}")
        return SessionLineResult()

    def _cmd_skill(self, s: str) -> SessionLineResult:
        try:
            toks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/skill: {e}")
            return SessionLineResult()
        if len(toks) < 2 or toks[1].lower() in ("help", "-h", "--help"):
            sink_print_compat(
                "Usage:\n"
                "  /skill list\n"
                "  /skill auto <request>\n"
                "  /skill reuse <request>\n"
                "  /skill <skill-id> <request>\n"
            )
            return SessionLineResult()
        sub = toks[1].strip()
        if sub.lower() in ("list", "ls"):
            if not self.skills_map:
                sink_print_compat("(no skills loaded)")
            else:
                sink_print_compat("Skills:")
                for sid in sorted(self.skills_map.keys()):
                    rec = self.skills_map.get(sid) or {}
                    desc = (rec.get("description") or "").strip() if isinstance(rec, dict) else ""
                    sink_print_compat(f"- {sid}" + (f": {desc}" if desc else ""))
            return SessionLineResult()
        if sub.lower() == "auto":
            req = " ".join(toks[2:]).strip()
            if not req:
                sink_print_compat("Usage: /skill auto <request>")
                return SessionLineResult()
            sid, why = self._ml_select_skill_id(
                req, self.skills_map, primary_profile=self.primary_profile, verbose=self.verbose
            )
            if not sid:
                sink_print_compat(f"/skill auto: no skill selected. {why}".strip())
                return SessionLineResult()
            self._run_with_selected_skill(req, sid, source="auto", selection_rationale=why)
            return SessionLineResult()
        if sub.lower() == "reuse":
            req = " ".join(toks[2:]).strip()
            if not req:
                sink_print_compat("Usage: /skill reuse <request>")
                return SessionLineResult()
            if not self.last_reuse_skill_id:
                sink_print_compat("/skill reuse: no stored skill. Run /skill auto <request> or /skill <id> <request> first.")
                return SessionLineResult()
            sid2 = self.last_reuse_skill_id
            if sid2 not in self.skills_map:
                sink_print_compat(
                    f"/skill reuse: stored skill {sid2!r} is not in the current skill set. "
                    "Run /skill auto again (check skills_dir / /set save)."
                )
                self.last_reuse_skill_id = None
                return SessionLineResult()
            self._run_with_selected_skill(
                req,
                sid2,
                source="reuse",
                selection_rationale="Follow-up; model skill selector skipped; same id as last skill run.",
            )
            return SessionLineResult()
        # explicit id
        sid = sub
        req = " ".join(toks[2:]).strip()
        if not sid or not req:
            sink_print_compat("Usage: /skill <skill> <request>")
            return SessionLineResult()
        if sid not in self.skills_map:
            sink_print_compat(
                f"/skill: unknown skill {sid!r}. "
                "Run /set save if you changed skills_dir, or check your skills directory."
            )
            return SessionLineResult()
        self._run_with_selected_skill(req, sid, source="explicit", selection_rationale="Explicit skill id; model skill selector skipped.")
        return SessionLineResult()

    def _cmd_skill_backcompat(self, s: str) -> SessionLineResult:
        low = s.lower()
        if low.startswith("/use-skills"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/use-skills: {e}")
                return SessionLineResult()
            if len(toks) < 2:
                sink_print_compat("Usage: /use-skills <user request>")
                return SessionLineResult()
            req = " ".join(toks[1:]).strip()
            if not req:
                sink_print_compat("Usage: /use-skills <user request>")
                return SessionLineResult()
            sid, why = self._ml_select_skill_id(
                req, self.skills_map, primary_profile=self.primary_profile, verbose=self.verbose
            )
            if not sid:
                sink_print_compat(f"/use-skills: no skill selected. {why}".strip())
                return SessionLineResult()
            self._run_with_selected_skill(req, sid, source="auto", selection_rationale=why)
            return SessionLineResult()
        if low.startswith("/use-skill"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/use-skill: {e}")
                return SessionLineResult()
            if len(toks) < 3:
                sink_print_compat("Usage: /use-skill <skill> <user request>")
                return SessionLineResult()
            sid = toks[1].strip()
            req = " ".join(toks[2:]).strip()
            if not sid or not req:
                sink_print_compat("Usage: /use-skill <skill> <user request>")
                return SessionLineResult()
            if sid not in self.skills_map:
                sink_print_compat(
                    f"/use-skill: unknown skill {sid!r}. "
                    "Run /set save if you changed skills_dir, or check your skills directory."
                )
                return SessionLineResult()
            self._run_with_selected_skill(req, sid, source="explicit", selection_rationale="Explicit skill id; model skill selector skipped.")
            return SessionLineResult()
        if low.startswith("/reuse-skill"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                sink_print_compat(f"/reuse-skill: {e}")
                return SessionLineResult()
            if len(toks) < 2:
                sink_print_compat("Usage: /reuse-skill <follow-up request (same skill as last /use-skills or /reuse-skill)>")
                return SessionLineResult()
            req = " ".join(toks[1:]).strip()
            if not req:
                sink_print_compat("Usage: /reuse-skill <follow-up request>")
                return SessionLineResult()
            if not self.last_reuse_skill_id:
                sink_print_compat(
                    "/reuse-skill: no stored skill. Run /use-skills <request> first, "
                    "or use a normal line for trigger-based skills."
                )
                return SessionLineResult()
            sid2 = self.last_reuse_skill_id
            if sid2 not in self.skills_map:
                sink_print_compat(
                    f"/reuse-skill: stored skill {sid2!r} is not in the current skill set. "
                    "Run /use-skills again (check skills_dir / /set save)."
                )
                self.last_reuse_skill_id = None
                return SessionLineResult()
            self._run_with_selected_skill(req, sid2, source="reuse", selection_rationale="Follow-up; model skill selector skipped; same id as last skill run.")
            return SessionLineResult()
        sink_print_compat(f"Unknown command {s.split()[0]!r}. Try /help.")
        return SessionLineResult()

    def _cmd_load_context(self, s: str) -> SessionLineResult:
        rest = s.split(None, 1)
        if len(rest) < 2:
            sink_print_compat("Usage: /load_context <file>")
            return SessionLineResult()
        path = rest[1].strip()
        if not path:
            sink_print_compat("Usage: /load_context <file>")
            return SessionLineResult()
        try:
            loaded = self._load_context_messages(path)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            sink_print_compat(f"/load_context error: {e}")
            return SessionLineResult()
        self.messages[:] = loaded
        sink_print_compat(f"Loaded {len(loaded)} message(s) from {path!r}.")
        return SessionLineResult()

    def _cmd_save_context(self, s: str) -> SessionLineResult:
        rest = s.split(None, 1)
        if len(rest) < 2:
            sink_print_compat("Usage: /save_context <file>")
            return SessionLineResult()
        path = rest[1].strip()
        if not path:
            sink_print_compat("Usage: /save_context <file>")
            return SessionLineResult()
        try:
            self._save_context_bundle(path, self.messages, "", None, False)
        except OSError as e:
            sink_print_compat(f"/save_context error: {e}")
            return SessionLineResult()
        self.session_save_path = path
        sink_print_compat(f"Wrote current session to {path!r}; further turns auto-save there.")
        return SessionLineResult()

    def _cmd_settings(self, s: str) -> SessionLineResult:
        try:
            toks = shlex.split(s)
        except ValueError as e:
            sink_print_compat(f"/set: {e}")
            return SessionLineResult()
        if len(toks) < 2:
            sink_print_compat("Usage: /set <topic> ...   (try: /set help)")
            return SessionLineResult()
        key = toks[1].lower().replace("-", "_")
        if key in ("help", "-h", "--help"):
            sink_print_compat(
                "Usage:\n"
                "  /set save\n"
                "  /set model <ollama-model>\n"
                "  /set enable|disable <feature/tool>\n"
                "  /set tools ...\n"
                "  /set system_prompt ...\n"
                "  /set prompt_template ...\n"
                "  /set context ...\n"
                "  /set thinking ...\n"
                "  /set ollama|openai|agent show|keys|set|unset\n"
            )
            return SessionLineResult()

        # group-backed settings
        if key in ("ollama", "openai", "agent"):
            if len(toks) < 3:
                sink_print_compat(
                    f"Usage: /set {key} show | keys | set <name> <value> | unset <name>\n"
                    "  Keys are lowercase (e.g. host, model, api_key). After changing, use /set save."
                )
                sink_print_compat(self._settings_group_keys_lines(key))
                return SessionLineResult()
            sub = toks[2].lower()
            if sub in ("show", "list"):
                try:
                    sink_print_compat(self._settings_group_show(key))
                except (ValueError, OSError) as e:
                    sink_print_compat(f"/set {key} show: {e}")
                return SessionLineResult()
            if sub in ("keys", "key", "help"):
                try:
                    sink_print_compat(self._settings_group_keys_lines(key))
                except (ValueError, OSError) as e:
                    sink_print_compat(f"/set {key} keys: {e}")
                return SessionLineResult()
            if sub == "set":
                if len(toks) < 4:
                    sink_print_compat(f"Usage: /set {key} set <name> <value (optional, quote spaces with shlex)>")
                    return SessionLineResult()
                raw_k = toks[3]
                value = " ".join(toks[4:]) if len(toks) > 4 else ""
                try:
                    msg = self._settings_group_set(key, raw_k, value)
                except ValueError as e:
                    sink_print_compat(f"/set {key} set: {e}")
                    return SessionLineResult()
                sink_print_compat(msg)
                return SessionLineResult()
            if sub in ("unset", "delete", "clear"):
                if len(toks) < 4:
                    sink_print_compat(f"Usage: /set {key} unset <name>")
                    return SessionLineResult()
                try:
                    msg = self._settings_group_unset(key, toks[3])
                except ValueError as e:
                    sink_print_compat(f"/set {key} unset: {e}")
                    return SessionLineResult()
                sink_print_compat(msg)
                return SessionLineResult()
            sink_print_compat(f"Unknown /set {key} subcommand. Try: /set {key} show | set | unset | keys")
            return SessionLineResult()

        if key == "verbose":
            if len(toks) != 3:
                sink_print_compat("Usage: /set verbose 0|1|2|on|off")
                return SessionLineResult()
            tok = toks[2].strip().lower()
            if tok == "on":
                self.verbose = 2
            elif tok == "off":
                self.verbose = 0
            elif tok in ("0", "1", "2"):
                self.verbose = int(tok)
            else:
                sink_print_compat("Usage: /set verbose 0|1|2|on|off")
                return SessionLineResult()
            sink_print_compat(self._verbose_ack_message(self.verbose))
            return SessionLineResult()

        if key == "tools":
            if len(toks) == 2 or (len(toks) >= 3 and toks[2].lower() in ("list", "ls", "show")):
                sink_print_compat(self._registry.format_settings_tools_list(self.enabled_tools))
                plugin_toolsets = self._registry.plugin_toolsets
                if plugin_toolsets:
                    lines = ["\nToolsets (plugins):"]
                    for nm in sorted(plugin_toolsets.keys()):
                        on = "on" if nm in self.enabled_toolsets else "off"
                        desc = str((plugin_toolsets.get(nm) or {}).get("description") or "").strip()
                        lines.append(f"  [{on}] {nm}" + (f" — {desc}" if desc else ""))
                        tnames = sorted(self._registry.plugin_tools_for_toolset(nm))
                        for tid in tnames:
                            td_on = (nm in self.enabled_toolsets) and (tid in self.enabled_tools)
                            reason = ""
                            if nm not in self.enabled_toolsets:
                                reason = " (toolset off)"
                            elif tid not in self.enabled_tools:
                                reason = " (tool disabled)"
                            lines.append(f"       - {'on' if td_on else 'off'} {tid}{reason}")
                    lines.append("Enable a toolset:  /set tools enable <toolset>")
                    lines.append("Disable a toolset: /set tools disable <toolset>")
                    lines.append("Reload plugins:    /set tools reload")
                    lines.append("Describe a tool:   /set tools describe <tool-id>")
                    sink_print_compat("\n".join(lines))
                return SessionLineResult()
            if len(toks) >= 4 and toks[2].lower() in ("enable", "on"):
                nm = toks[3].strip().lower()
                plugin_toolsets = self._registry.plugin_toolsets
                if nm in plugin_toolsets:
                    self.enabled_toolsets.add(nm)
                    for tid in self._registry.plugin_tools_for_toolset(nm):
                        self.enabled_tools.add(tid)
                    sink_print_compat(
                        f"Toolset enabled: {nm!r} (tools may be routed per request). Use /set save to persist."
                    )
                else:
                    sink_print_compat(f"Unknown toolset {nm!r}. Try: /set tools")
                return SessionLineResult()
            if len(toks) >= 4 and toks[2].lower() in ("disable", "off"):
                nm = toks[3].strip().lower()
                plugin_toolsets = self._registry.plugin_toolsets
                if nm in plugin_toolsets:
                    self.enabled_toolsets.discard(nm)
                    for tid in self._registry.plugin_tools_for_toolset(nm):
                        self.enabled_tools.discard(tid)
                    sink_print_compat(f"Toolset disabled: {nm!r}. Use /set save to persist.")
                else:
                    sink_print_compat(f"Unknown toolset {nm!r}. Try: /set tools")
                return SessionLineResult()
            if len(toks) >= 3 and toks[2].lower() in ("reload", "refresh"):
                self._registry.load_plugin_toolsets(self.tools_dir)
                self._registry.register_aliases()
                sink_print_compat(f"Reloaded plugin toolsets from {self.tools_dir!r}.")
                return SessionLineResult()
            if len(toks) >= 4 and toks[2].lower() in ("describe", "desc", "help"):
                tid = toks[3].strip()
                if not tid:
                    sink_print_compat("Usage: /set tools describe <tool-id>")
                    return SessionLineResult()
                nm = tid.strip().lower()
                plugin_toolsets = self._registry.plugin_toolsets
                if nm in plugin_toolsets:
                    rec = plugin_toolsets.get(nm) or {}
                    desc = str(rec.get("description") or "").strip()
                    sink_print_compat(f"Toolset: {nm}\nDescription: {desc if desc else '(none)'}")
                    sink_print_compat("Tools:")
                    for one in sorted(self._registry.plugin_tools_for_toolset(nm)):
                        sink_print_compat("  - " + one)
                    return SessionLineResult()
                sink_print_compat(self._registry.describe_tool_call_contract(tid))
                return SessionLineResult()
            sink_print_compat("Usage: /set tools [list] | enable <toolset> | disable <toolset>")
            return SessionLineResult()

        if key == "system_prompt":
            if len(toks) < 3:
                sink_print_compat(
                    "Usage:\n"
                    "  /set system_prompt show\n"
                    "  /set system_prompt reset\n"
                    "  /set system_prompt pin\n"
                    "  /set system_prompt file <path>\n"
                    "  /set system_prompt save <path>\n"
                    "  /set system_prompt <text>\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub == "show":
                body = agent_prompts.effective_system_instruction_text_for_tools(
                    self.session_system_prompt, frozenset(self.enabled_tools)
                )
                sink_print_compat(f"Effective system prompt ({len(body)} chars):\n{body}")
                if self.session_system_prompt_path:
                    sink_print_compat(f"(File-backed: {self.session_system_prompt_path!r})")
                elif self.session_system_prompt is not None:
                    sink_print_compat("(Session inline override.)")
                else:
                    sink_print_compat("(Built-in default.)")
                return SessionLineResult()
            if sub in ("pin", "snapshot"):
                body = agent_prompts.effective_system_instruction_text_for_tools(
                    self.session_system_prompt, frozenset(self.enabled_tools)
                )
                self.session_system_prompt = body
                self.session_system_prompt_path = None
                self.session_prompt_template = None
                self._system_prompt_explicit = True
                sink_print_compat(
                    f"System prompt pinned for this session ({len(body)} chars). "
                    "Use /set save to persist to ~/.agent.json."
                )
                return SessionLineResult()
            if sub in ("reset", "default"):
                self.session_system_prompt = None
                self.session_system_prompt_path = None
                self._system_prompt_explicit = False
                sink_print_compat("System prompt reset to built-in default for this session.")
                return SessionLineResult()
            if sub == "file":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set system_prompt file <path>")
                    return SessionLineResult()
                path = os.path.expanduser(" ".join(toks[3:]).strip())
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        body = f.read()
                except OSError as e:
                    sink_print_compat(f"/set system_prompt file: {e}")
                    return SessionLineResult()
                if not body.strip():
                    sink_print_compat("File is empty.")
                    return SessionLineResult()
                self.session_system_prompt = body
                self.session_system_prompt_path = os.path.abspath(path)
                self.session_prompt_template = None
                self._system_prompt_explicit = True
                sink_print_compat(
                    f"System prompt loaded from {path!r} ({len(body)} chars). "
                    "/set save will store this path in ~/.agent.json."
                )
                return SessionLineResult()
            if sub == "save":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set system_prompt save <path>")
                    return SessionLineResult()
                path = os.path.expanduser(" ".join(toks[3:]).strip())
                body = agent_prompts.effective_system_instruction_text_for_tools(
                    self.session_system_prompt, frozenset(self.enabled_tools)
                )
                try:
                    parent = os.path.dirname(path)
                    if parent and not os.path.isdir(parent):
                        os.makedirs(parent, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(body)
                except OSError as e:
                    sink_print_compat(f"/set system_prompt save: {e}")
                    return SessionLineResult()
                sink_print_compat(f"Wrote system prompt ({len(body)} chars) to {path!r}.")
                return SessionLineResult()
            phrase = " ".join(toks[2:])
            if not phrase.strip():
                sink_print_compat("Usage: /set system_prompt <non-empty one-line text>")
                return SessionLineResult()
            self.session_system_prompt = phrase
            self.session_system_prompt_path = None
            self.session_prompt_template = None
            self._system_prompt_explicit = True
            sink_print_compat(
                f"System prompt set inline ({len(phrase)} chars). "
                "/set save will store the text in ~/.agent.json."
            )
            return SessionLineResult()

        if key in ("prompt_template", "prompt_templates", "prompt"):
            if len(toks) < 3:
                sink_print_compat(
                    "Usage:\n"
                    "  /set prompt_template list\n"
                    "  /set prompt_template show\n"
                    "  /set prompt_template use <name>\n"
                    "  /set prompt_template default <name>\n"
                    "  /set prompt_template set <name> <text>\n"
                    "  /set prompt_template delete <name>\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub in ("help", "-h", "--help", "explain"):
                sink_print_compat("Try: /set prompt_template list")
                return SessionLineResult()
            if sub == "list":
                names = sorted(self.prompt_templates.keys())
                if not names:
                    sink_print_compat("(no prompt templates)")
                    return SessionLineResult()
                for nm in names:
                    obj = self.prompt_templates.get(nm) or {}
                    desc = str(obj.get("description") or "").strip() if isinstance(obj, dict) else ""
                    mark = ""
                    if self.session_prompt_template == nm:
                        mark = " *active*"
                    elif self.template_default == nm:
                        mark = " (default)"
                    line = f"- {nm}{mark}"
                    if desc:
                        line += f": {desc}"
                    sink_print_compat(line)
                return SessionLineResult()
            if sub == "show":
                active = self.session_prompt_template or self.template_default
                body = agent_prompts.resolve_prompt_template_text(active, self.prompt_templates) or ""
                sink_print_compat(f"Active template: {active!r}\nPrompt ({len(body)} chars):\n{body}")
                return SessionLineResult()
            if sub in ("use", "select"):
                if len(toks) < 4:
                    sink_print_compat("Usage: /set prompt_template use <name>")
                    return SessionLineResult()
                nm = toks[3].strip()
                if nm not in self.prompt_templates:
                    sink_print_compat(f"Unknown template {nm!r}. Try: /set prompt_template list")
                    return SessionLineResult()
                resolved = agent_prompts.resolve_prompt_template_text(nm, self.prompt_templates)
                if not resolved:
                    sink_print_compat(f"Template {nm!r} has no usable text/path.")
                    return SessionLineResult()
                self.session_system_prompt = resolved
                self.session_system_prompt_path = None
                self.session_prompt_template = nm
                # Selecting a prompt template is not a system_prompt override; it should not be persisted
                # as a system_prompt snapshot unless the user explicitly pins/sets it.
                self._system_prompt_explicit = False
                sink_print_compat(f"Using prompt template {nm!r} for this session.")
                return SessionLineResult()
            if sub == "default":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set prompt_template default <name>")
                    return SessionLineResult()
                nm = toks[3].strip()
                if nm not in self.prompt_templates:
                    sink_print_compat(f"Unknown template {nm!r}. Try: /set prompt_template list")
                    return SessionLineResult()
                self.template_default = nm
                self._prompt_template_default_explicit = True
                sink_print_compat(f"Default prompt template set to {nm!r} (use /set save to persist).")
                return SessionLineResult()
            if sub == "set":
                if len(toks) < 5:
                    sink_print_compat("Usage: /set prompt_template set <name> <text>")
                    return SessionLineResult()
                nm = toks[3].strip()
                text = " ".join(toks[4:]).strip()
                if not nm:
                    sink_print_compat("Template name must be non-empty.")
                    return SessionLineResult()
                if not text:
                    sink_print_compat("Template text must be non-empty.")
                    return SessionLineResult()
                cur = self.prompt_templates.get(nm) or {}
                desc = str(cur.get("description") or "") if isinstance(cur, dict) else ""
                self.prompt_templates[nm] = {"kind": "overlay", "description": desc, "text": text}
                self._prompt_templates_explicit = True
                sink_print_compat(f"Template {nm!r} set/updated (overlay). Use /set save to persist.")
                return SessionLineResult()
            if sub in ("delete", "del", "rm", "remove"):
                if len(toks) < 4:
                    sink_print_compat("Usage: /set prompt_template delete <name>")
                    return SessionLineResult()
                nm = toks[3].strip()
                on_disk = os.path.join(self.prompt_templates_dir, f"{nm}.json")
                if os.path.isfile(on_disk):
                    sink_print_compat(
                        "Refusing to delete a template that exists as a file on disk in "
                        f"the configured prompt_templates_dir ({self.prompt_templates_dir!r}). "
                        "You can override it in ~/.agent.json with a same-named entry."
                    )
                    return SessionLineResult()
                if nm not in self.prompt_templates:
                    sink_print_compat(f"Unknown template {nm!r}.")
                    return SessionLineResult()
                self.prompt_templates.pop(nm, None)
                if self.session_prompt_template == nm:
                    self.session_prompt_template = None
                self._prompt_templates_explicit = True
                sink_print_compat(f"Deleted template {nm!r}. Use /set save to persist.")
                return SessionLineResult()
            sink_print_compat("Unknown subcommand. Try: /set prompt_template list")
            return SessionLineResult()

        if key in ("context", "context_manager", "context_window"):
            if len(toks) < 3:
                sink_print_compat(
                    "Usage:\n"
                    "  /set context show\n"
                    "  /set context on|off\n"
                    "  /set context tokens <n>\n"
                    "  /set context trigger <0..1>\n"
                    "  /set context target <0..1>\n"
                    "  /set context keep_tail <n>\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub == "show":
                sink_print_compat(
                    "Context manager (prefs; env vars may override):\n"
                    f"  enabled: {bool(self.context_cfg.get('enabled', True))}\n"
                    f"  tokens: {self.context_cfg.get('tokens', 0)}  (0 = auto per backend)\n"
                    f"  trigger_frac: {self.context_cfg.get('trigger_frac', 0.75)}\n"
                    f"  target_frac: {self.context_cfg.get('target_frac', 0.55)}\n"
                    f"  keep_tail_messages: {self.context_cfg.get('keep_tail_messages', 12)}\n"
                )
                return SessionLineResult()
            if sub in ("on", "enable", "enabled", "true"):
                self.context_cfg["enabled"] = True
                sink_print_compat("Context manager enabled for this session. Use /set save to persist.")
                return SessionLineResult()
            if sub in ("off", "disable", "disabled", "false"):
                self.context_cfg["enabled"] = False
                sink_print_compat("Context manager disabled for this session. Use /set save to persist.")
                return SessionLineResult()
            if sub == "tokens":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set context tokens <n>")
                    return SessionLineResult()
                try:
                    n = int(toks[3], 10)
                except ValueError:
                    sink_print_compat("tokens must be an integer.")
                    return SessionLineResult()
                if n < 0:
                    n = 0
                self.context_cfg["tokens"] = n
                sink_print_compat(f"context tokens set to {n} (0 = auto). Use /set save to persist.")
                return SessionLineResult()
            if sub == "trigger":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set context trigger <0..1>")
                    return SessionLineResult()
                try:
                    x = float(toks[3])
                except ValueError:
                    sink_print_compat("trigger must be a number.")
                    return SessionLineResult()
                self.context_cfg["trigger_frac"] = max(0.05, min(0.95, x))
                sink_print_compat(f"trigger_frac set to {self.context_cfg['trigger_frac']}. Use /set save to persist.")
                return SessionLineResult()
            if sub == "target":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set context target <0..1>")
                    return SessionLineResult()
                try:
                    x = float(toks[3])
                except ValueError:
                    sink_print_compat("target must be a number.")
                    return SessionLineResult()
                cur_tr = float(self.context_cfg.get("trigger_frac", 0.75))
                self.context_cfg["target_frac"] = max(0.05, min(cur_tr, x))
                sink_print_compat(f"target_frac set to {self.context_cfg['target_frac']}. Use /set save to persist.")
                return SessionLineResult()
            if sub in ("keep_tail", "keep", "tail"):
                if len(toks) < 4:
                    sink_print_compat("Usage: /set context keep_tail <n>")
                    return SessionLineResult()
                try:
                    n = int(toks[3], 10)
                except ValueError:
                    sink_print_compat("keep_tail must be an integer.")
                    return SessionLineResult()
                self.context_cfg["keep_tail_messages"] = max(4, n)
                sink_print_compat(
                    f"keep_tail_messages set to {self.context_cfg['keep_tail_messages']}. Use /set save to persist."
                )
                return SessionLineResult()
            sink_print_compat("Unknown subcommand. Try: /set context show")
            return SessionLineResult()

        if key == "save":
            full_snapshot = False
            if len(toks) == 3 and toks[2].strip().lower() == "full":
                full_snapshot = True
            elif any(t.strip().lower() == "--full" for t in toks[2:]):
                full_snapshot = True
            elif len(toks) != 2:
                sink_print_compat("Usage: /set save [full|--full]")
                return SessionLineResult()
            try:
                payload = self._build_agent_prefs_payload(
                    primary_profile=self.primary_profile,
                    second_opinion_on=self.second_opinion_on,
                    cloud_ai_enabled=self.cloud_ai_enabled,
                    enabled_tools=self.enabled_tools,
                    enabled_toolsets=self.enabled_toolsets,
                    reviewer_hosted_profile=self.reviewer_hosted_profile,
                    reviewer_ollama_model=self.reviewer_ollama_model,
                    session_save_path=self.session_save_path,
                    system_prompt_override=(
                        self.session_system_prompt if self._system_prompt_explicit else None
                    ),
                    system_prompt_path_override=(
                        self.session_system_prompt_path if self._system_prompt_explicit else None
                    ),
                    prompt_templates=self.prompt_templates if self._prompt_templates_explicit else None,
                    prompt_template_default=self.template_default if self._prompt_template_default_explicit else None,
                    context_manager=self.context_cfg,
                    verbose_level=self.verbose,
                    full_snapshot=full_snapshot,
                )
                self._write_agent_prefs_file(payload)
            except OSError as e:
                sink_print_compat(f"/set save error: {e}")
                return SessionLineResult()
            sink_print_compat(f"Saved settings to {self._agent_prefs_path()!r}.")
            return SessionLineResult()

        if key == "model":
            if len(toks) < 3:
                sink_print_compat("Usage: /set model <ollama-model-name>")
                return SessionLineResult()
            name = toks[2].strip()
            if not name:
                sink_print_compat("Usage: /set model <ollama-model-name>")
                return SessionLineResult()
            self._settings_set(("ollama", "model"), name)
            sink_print_compat(f"ollama.model set to {name!r}. Use /set save to persist.")
            return SessionLineResult()

        if key == "enable":
            if len(toks) < 3:
                sink_print_compat(
                    "Usage: /set enable second_opinion|<tool or phrase>\n"
                    "  Examples: /set enable web search   /set enable shell   /set enable stream_thinking\n"
                    "  See: /set tools"
                )
                return SessionLineResult()
            phrase = " ".join(toks[2:])
            feat = self._registry.canonicalize_user_tool_phrase(phrase)
            if feat == "second_opinion":
                self.second_opinion_on = True
                sink_print_compat("second_opinion enabled for this session.")
                return SessionLineResult()
            if feat in ("stream_thinking", "streamthinking", "stream_think", "thinking_stream", "showthinking", "show_thinking"):
                self._settings_set(("agent", "stream_thinking"), True)
                sink_print_compat(
                    "stream_thinking enabled for this session (streams model thinking when available). Use /set save to persist."
                )
                return SessionLineResult()
            if feat == "verbose":
                self.verbose = 2
                sink_print_compat(self._verbose_ack_message(self.verbose))
                return SessionLineResult()
            tn = self._registry.normalize_tool_name(phrase)
            if tn:
                self.enabled_tools.add(tn)
                sink_print_compat(f"Tool enabled: {tn}")
                return SessionLineResult()
            sink_print_compat(self._registry.format_unknown_tool_hint(phrase))
            return SessionLineResult()

        if key == "disable":
            if len(toks) < 3:
                sink_print_compat(
                    "Usage: /set disable second_opinion|<tool or phrase>\n"
                    "  Examples: /set disable web search   /set disable shell   /set disable stream_thinking\n"
                    "  See: /set tools"
                )
                return SessionLineResult()
            phrase = " ".join(toks[2:])
            feat = self._registry.canonicalize_user_tool_phrase(phrase)
            if feat == "second_opinion":
                self.second_opinion_on = False
                sink_print_compat("second_opinion disabled for this session.")
                return SessionLineResult()
            if feat in ("stream_thinking", "streamthinking", "stream_think", "thinking_stream", "showthinking", "show_thinking"):
                self._settings_set(("agent", "stream_thinking"), False)
                sink_print_compat("stream_thinking disabled for this session. Use /set save to persist.")
                return SessionLineResult()
            if feat == "verbose":
                self.verbose = 0
                sink_print_compat(self._verbose_ack_message(self.verbose))
                return SessionLineResult()
            tn = self._registry.normalize_tool_name(phrase)
            if tn:
                self.enabled_tools.discard(tn)
                sink_print_compat(f"Tool disabled: {tn}")
                return SessionLineResult()
            sink_print_compat(self._registry.format_unknown_tool_hint(phrase))
            return SessionLineResult()

        if key == "thinking":
            if len(toks) < 3:
                sink_print_compat(
                    "Usage:\n"
                    "  /set thinking show\n"
                    "  /set thinking on|off\n"
                    "  /set thinking level low|medium|high\n"
                    "Notes:\n"
                    "  - This controls the Ollama request `think` field (bool or level string).\n"
                    "  - Some models ignore booleans and require levels; others support both.\n"
                    "  - thinking on/level also enables stream_thinking automatically (use /set disable stream_thinking to hide).\n"
                    "  - Use /set save to persist.\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub == "show":
                think_v = self._ollama_request_think_value()
                lvl = self._agent_thinking_level()
                on = self._agent_thinking_enabled_default_false()
                st = "on" if on else "off"
                sink_print_compat(
                    f"thinking: {st}; level: {lvl or '(none)'}; ollama think value: {think_v!r}; stream_thinking: {self._agent_stream_thinking_enabled()}"
                )
                return SessionLineResult()
            if sub in ("on", "enable", "enabled", "true"):
                self._settings_set(("agent", "thinking"), True)
                self._settings_set(("agent", "stream_thinking"), True)
                sink_print_compat(
                    "thinking enabled for this session (and stream_thinking enabled). Use /set save to persist."
                )
                return SessionLineResult()
            if sub in ("off", "disable", "disabled", "false"):
                self._settings_set(("agent", "thinking"), False)
                self._settings_set(("agent", "thinking_level"), "")
                self._settings_set(("agent", "stream_thinking"), False)
                sink_print_compat(
                    "thinking disabled for this session (and stream_thinking disabled). Use /set save to persist."
                )
                return SessionLineResult()
            if sub == "level":
                if len(toks) < 4:
                    sink_print_compat("Usage: /set thinking level low|medium|high")
                    return SessionLineResult()
                lvl = toks[3].strip().lower()
                if lvl not in ("low", "medium", "high"):
                    sink_print_compat("thinking level must be one of: low, medium, high")
                    return SessionLineResult()
                self._settings_set(("agent", "thinking_level"), lvl)
                self._settings_set(("agent", "thinking"), True)
                self._settings_set(("agent", "stream_thinking"), True)
                sink_print_compat(
                    f"thinking level set to {lvl!r} for this session (and stream_thinking enabled). Use /set save to persist."
                )
                return SessionLineResult()
            sink_print_compat("Unknown /set thinking subcommand. Try: /set thinking show | on | off | level …")
            return SessionLineResult()

        if key == "primary" and len(toks) >= 4 and toks[2].lower() == "llm":
            sub = toks[3].lower()
            if sub == "ollama":
                self.primary_profile = self._default_primary_llm_profile()
                sink_print_compat("Primary LLM: local Ollama.")
            elif sub == "hosted":
                if len(toks) < 6:
                    sink_print_compat("Usage: /set primary llm hosted <base_url> <model> [api_key]")
                    return SessionLineResult()
                bu, mod = toks[4], toks[5]
                if not bu.startswith(("http://", "https://")):
                    sink_print_compat("base_url must start with http:// or https://")
                    return SessionLineResult()
                keyval = toks[6] if len(toks) > 6 else ""
                self.primary_profile = self._LlmProfile(
                    backend="hosted",
                    base_url=bu,
                    model=mod,
                    api_key=keyval,
                )
                if not (keyval or "").strip():
                    sink_print_compat("Note: api_key is not set; hosted primary calls will fail until it is.")
                sink_print_compat(
                    "Primary LLM: hosted OpenAI-compatible API "
                    f"({self._describe_llm_profile_short(self.primary_profile)})."
                )
            else:
                sink_print_compat("Usage: /set primary llm ollama|hosted …")
            return SessionLineResult()

        if toks[1].replace("-", "_").lower() == "second_opinion" and len(toks) >= 4 and toks[2].lower() == "llm":
            sub = toks[3].lower()
            if sub == "ollama":
                self.reviewer_hosted_profile = None
                self.reviewer_ollama_model = toks[4] if len(toks) > 4 else None
                om = self.reviewer_ollama_model or self._ollama_second_opinion_model()
                sink_print_compat(f"Second-opinion reviewer: local Ollama, model {om!r}.")
            elif sub == "hosted":
                if len(toks) < 6:
                    sink_print_compat("Usage: /set second_opinion llm hosted <base_url> <model> [api_key]")
                    return SessionLineResult()
                bu, mod = toks[4], toks[5]
                if not bu.startswith(("http://", "https://")):
                    sink_print_compat("base_url must start with http:// or https://")
                    return SessionLineResult()
                keyval = toks[6] if len(toks) > 6 else ""
                self.reviewer_hosted_profile = self._LlmProfile(
                    backend="hosted",
                    base_url=bu,
                    model=mod,
                    api_key=keyval,
                )
                self.reviewer_ollama_model = None
                if not (keyval or "").strip():
                    sink_print_compat("Note: api_key is not set; hosted second opinion will fail until it is.")
                sink_print_compat(
                    "Second-opinion reviewer: hosted "
                    f"({self._describe_llm_profile_short(self.reviewer_hosted_profile)})."
                )
            else:
                sink_print_compat("Usage: /set second_opinion llm ollama|hosted …")
            return SessionLineResult()

        sink_print_compat("Unknown /set subcommand. Try /help.")
        return SessionLineResult()

