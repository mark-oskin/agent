from __future__ import annotations

import datetime
import contextlib
import json
import os
import shlex
from dataclasses import dataclass
from io import StringIO
from typing import AbstractSet, Callable, Optional

from agentlib.tools.registry import ToolRegistry
from agentlib import prompts as agent_prompts

from .runtime import ConversationTurnDeps, run_agent_conversation_turn
from .settings import AgentSettings


@dataclass
class SessionLineResult:
    output: str = ""
    quit: bool = False


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

        self.messages: list = []
        self.last_reuse_skill_id: Optional[str] = None

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

    def execute_line(self, line: str, *, emit: Optional[Callable[[dict], None]] = None) -> dict:
        """
        Execute one REPL line.

        - When `emit` is provided, all stdout/stderr output produced during handling is captured and
          forwarded as typed emit events.
        - Returns a structured dict for embedding callers (CLI can ignore most fields).
        """
        s = (line or "").strip()
        if not s:
            return {"type": "noop", "quit": False}

        if emit is None:
            # Preserve legacy behavior (prints inside handlers).
            if s.startswith("/"):
                res = self._execute_command_line(s)
                return {"type": "command", "quit": bool(res.quit), "output": res.output}
            answered, final_answer = self._execute_user_request(s)
            return {
                "type": "turn",
                "quit": False,
                "answered": bool(answered),
                "answer": final_answer,
            }

        def _emit_captured(kind: str, text: str) -> None:
            if not text:
                return
            for raw_line in text.splitlines():
                ln = raw_line.rstrip("\n")
                if not ln.strip():
                    continue
                et = kind
                if ln.startswith("→ "):
                    et = "progress"
                elif ln.startswith("[Thinking]") or ln.startswith("[Done thinking]"):
                    et = "thinking"
                elif ln.startswith("Warning:"):
                    et = "warning"
                emit({"type": et, "text": ln})

        out_buf = StringIO()
        err_buf = StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            if s.startswith("/"):
                res = self._execute_command_line(s)
                payload = {"type": "command", "quit": bool(res.quit), "output": res.output}
            else:
                answered, final_answer = self._execute_user_request(s)
                payload = {
                    "type": "turn",
                    "quit": False,
                    "answered": bool(answered),
                    "answer": final_answer,
                }
        _emit_captured("output", out_buf.getvalue())
        _emit_captured("stderr", err_buf.getvalue())
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
        if router_query and "search_web" in et_turn:
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "Before answering, you MUST call the tool search_web.\n"
                        "Respond with JSON only in tool_call form.\n"
                        f'Suggested query: "{router_query}"'
                    ),
                }
            )
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
                print(f"Warning: could not save context: {e}")
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
            print(
                f"/skill reuse: using skill {sid!r} (model skill selection skipped). "
                f"{selection_rationale}".strip()
            )
        elif src == "explicit":
            print(f"/skill: using skill {sid!r}. {selection_rationale}".strip())
        else:
            print(f"/skill auto selected {sid!r}. {selection_rationale}".strip())
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
            print(f"Skill workflow: executing {len(steps)} step(s).", flush=True)
            self._agent_progress(f"Running {len(steps)}-step skill workflow…")
            if self.verbose >= 1:
                rp = raw_plan or ""
                cap = 1200
                preview = rp if len(rp) <= cap else rp[:cap] + "…"
                print(f"[*] [skills:planner] raw ({len(rp)} chars): {preview}")
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
                if router_query and "search_web" in et_step and i == 1:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Before answering, you MUST call the tool search_web.\n"
                                "Respond with JSON only in tool_call form.\n"
                                f'Suggested query: "{router_query}"'
                            ),
                        }
                    )
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
                print(step_answers[-1])
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
        if router_query and "search_web" in et_turn:
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "Before answering, you MUST call the tool search_web.\n"
                        "Respond with JSON only in tool_call form.\n"
                        f'Suggested query: "{router_query}"'
                    ),
                }
            )
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
        )

    def _execute_command_line(self, s: str) -> SessionLineResult:
        low = s.lower()
        if low in ("/quit", "/exit", "/q"):
            return SessionLineResult(quit=True)
        if low == "/clear":
            self.messages.clear()
            self.last_reuse_skill_id = None
            print("Context cleared (including stored skill for /skill reuse).")
            return SessionLineResult()
        if low == "/models":
            try:
                names = self._fetch_ollama_local_model_names()
                print("\n".join(names) if names else "(no models returned)")
            except Exception as e:
                print(f"/models error: {e}")
            return SessionLineResult()
        if low in ("/usage", "/tokens"):
            print(self._format_last_ollama_usage_for_repl())
            return SessionLineResult()
        if s.startswith("/show"):
            return self._cmd_show(s)
        if s.startswith("/while"):
            return self._cmd_while(s)
        if low.startswith("/skill"):
            return self._cmd_skill(s)
        if low.startswith("/use-skills") or low.startswith("/use-skill") or low.startswith("/reuse-skill"):
            return self._cmd_skill_backcompat(s)
        if low.startswith("/settings"):
            return self._cmd_settings(s)
        if low.startswith("/load_context"):
            return self._cmd_load_context(s)
        if low.startswith("/save_context"):
            return self._cmd_save_context(s)
        if low in ("/help", "/?"):
            print(
                "Commands:\n"
                "  /quit                    Exit\n"
                "  /clear                   Clear in-memory conversation\n"
                "  /help                    Help\n"
                "  /models                  List local Ollama models\n"
                "  /usage                   Last local Ollama usage\n"
                "  /show ...                Show current state (try /show help)\n"
                "  /skill ...               Skills (try /skill help)\n"
                "  /while ...               Loops (try /while help)\n"
                "  /settings ...            Configuration (try /settings help)\n"
                "  /load_context <file>     Replace session messages from JSON\n"
                "  /save_context <file>     Write session JSON; set auto-save path\n"
            )
            return SessionLineResult()
        print(f"Unknown command {s.split()[0]!r}. Try /help.")
        return SessionLineResult()

    def _cmd_show(self, s: str) -> SessionLineResult:
        try:
            toks = shlex.split(s)
        except ValueError as e:
            print(f"/show: {e}")
            return SessionLineResult()
        if len(toks) < 2 or toks[1].lower() in ("help", "-h", "--help"):
            print(
                "Usage:\n"
                "  /show model      Primary LLM in use (Ollama or hosted)\n"
                "  /show reviewer   Second-opinion reviewer model\n"
                "\n"
                "Settings that already have a show line: /settings tools, /settings context show, "
                "/settings thinking show, /settings system_prompt show, /settings prompt_template show, "
                "/settings ollama|openai|agent show"
            )
            return SessionLineResult()
        sub = toks[1].lower().replace("-", "_")
        if sub in ("model", "primary", "llm"):
            print(f"Primary LLM: {self._format_session_primary_llm_line(self.primary_profile)}")
            return SessionLineResult()
        if sub in ("reviewer", "second_opinion", "2nd"):
            print(
                "Second-opinion reviewer: "
                + self._format_session_reviewer_line(self.reviewer_hosted_profile, self.reviewer_ollama_model)
            )
            return SessionLineResult()
        print("Unknown /show topic. Try: /show model   or   /show reviewer")
        return SessionLineResult()

    def _cmd_while(self, s: str) -> SessionLineResult:
        try:
            wtoks = shlex.split(s)
        except ValueError as e:
            print(f"/while: {e}")
            return SessionLineResult()
        if len(wtoks) == 1 or (len(wtoks) == 2 and wtoks[1].lower() in ("help", "-h", "--help")):
            print(
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
            print(f"/while: {e}")
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
                    print("\n[Cancelled]\n")
                    break
                if bit == 0:
                    print(
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
                        print("\n[Cancelled]\n")
                        abort_while = True
                        break
                if abort_while:
                    break
            else:
                print(f"/while: reached --max {max_while} without judge returning 0 (exit).")
        except Exception as e:
            print(f"/while error: {e}")
        return SessionLineResult()

    def _cmd_skill(self, s: str) -> SessionLineResult:
        try:
            toks = shlex.split(s)
        except ValueError as e:
            print(f"/skill: {e}")
            return SessionLineResult()
        if len(toks) < 2 or toks[1].lower() in ("help", "-h", "--help"):
            print(
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
                print("(no skills loaded)")
            else:
                print("Skills:")
                for sid in sorted(self.skills_map.keys()):
                    rec = self.skills_map.get(sid) or {}
                    desc = (rec.get("description") or "").strip() if isinstance(rec, dict) else ""
                    print(f"- {sid}" + (f": {desc}" if desc else ""))
            return SessionLineResult()
        if sub.lower() == "auto":
            req = " ".join(toks[2:]).strip()
            if not req:
                print("Usage: /skill auto <request>")
                return SessionLineResult()
            sid, why = self._ml_select_skill_id(
                req, self.skills_map, primary_profile=self.primary_profile, verbose=self.verbose
            )
            if not sid:
                print(f"/skill auto: no skill selected. {why}".strip())
                return SessionLineResult()
            self._run_with_selected_skill(req, sid, source="auto", selection_rationale=why)
            return SessionLineResult()
        if sub.lower() == "reuse":
            req = " ".join(toks[2:]).strip()
            if not req:
                print("Usage: /skill reuse <request>")
                return SessionLineResult()
            if not self.last_reuse_skill_id:
                print("/skill reuse: no stored skill. Run /skill auto <request> or /skill <id> <request> first.")
                return SessionLineResult()
            sid2 = self.last_reuse_skill_id
            if sid2 not in self.skills_map:
                print(
                    f"/skill reuse: stored skill {sid2!r} is not in the current skill set. "
                    "Run /skill auto again (check skills_dir / /settings save)."
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
            print("Usage: /skill <skill> <request>")
            return SessionLineResult()
        if sid not in self.skills_map:
            print(
                f"/skill: unknown skill {sid!r}. "
                "Run /settings save if you changed skills_dir, or check your skills directory."
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
                print(f"/use-skills: {e}")
                return SessionLineResult()
            if len(toks) < 2:
                print("Usage: /use-skills <user request>")
                return SessionLineResult()
            req = " ".join(toks[1:]).strip()
            if not req:
                print("Usage: /use-skills <user request>")
                return SessionLineResult()
            sid, why = self._ml_select_skill_id(
                req, self.skills_map, primary_profile=self.primary_profile, verbose=self.verbose
            )
            if not sid:
                print(f"/use-skills: no skill selected. {why}".strip())
                return SessionLineResult()
            self._run_with_selected_skill(req, sid, source="auto", selection_rationale=why)
            return SessionLineResult()
        if low.startswith("/use-skill"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                print(f"/use-skill: {e}")
                return SessionLineResult()
            if len(toks) < 3:
                print("Usage: /use-skill <skill> <user request>")
                return SessionLineResult()
            sid = toks[1].strip()
            req = " ".join(toks[2:]).strip()
            if not sid or not req:
                print("Usage: /use-skill <skill> <user request>")
                return SessionLineResult()
            if sid not in self.skills_map:
                print(
                    f"/use-skill: unknown skill {sid!r}. "
                    "Run /settings save if you changed skills_dir, or check your skills directory."
                )
                return SessionLineResult()
            self._run_with_selected_skill(req, sid, source="explicit", selection_rationale="Explicit skill id; model skill selector skipped.")
            return SessionLineResult()
        if low.startswith("/reuse-skill"):
            try:
                toks = shlex.split(s)
            except ValueError as e:
                print(f"/reuse-skill: {e}")
                return SessionLineResult()
            if len(toks) < 2:
                print("Usage: /reuse-skill <follow-up request (same skill as last /use-skills or /reuse-skill)>")
                return SessionLineResult()
            req = " ".join(toks[1:]).strip()
            if not req:
                print("Usage: /reuse-skill <follow-up request>")
                return SessionLineResult()
            if not self.last_reuse_skill_id:
                print(
                    "/reuse-skill: no stored skill. Run /use-skills <request> first, "
                    "or use a normal line for trigger-based skills."
                )
                return SessionLineResult()
            sid2 = self.last_reuse_skill_id
            if sid2 not in self.skills_map:
                print(
                    f"/reuse-skill: stored skill {sid2!r} is not in the current skill set. "
                    "Run /use-skills again (check skills_dir / /settings save)."
                )
                self.last_reuse_skill_id = None
                return SessionLineResult()
            self._run_with_selected_skill(req, sid2, source="reuse", selection_rationale="Follow-up; model skill selector skipped; same id as last skill run.")
            return SessionLineResult()
        print(f"Unknown command {s.split()[0]!r}. Try /help.")
        return SessionLineResult()

    def _cmd_load_context(self, s: str) -> SessionLineResult:
        rest = s.split(None, 1)
        if len(rest) < 2:
            print("Usage: /load_context <file>")
            return SessionLineResult()
        path = rest[1].strip()
        if not path:
            print("Usage: /load_context <file>")
            return SessionLineResult()
        try:
            loaded = self._load_context_messages(path)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"/load_context error: {e}")
            return SessionLineResult()
        self.messages[:] = loaded
        print(f"Loaded {len(loaded)} message(s) from {path!r}.")
        return SessionLineResult()

    def _cmd_save_context(self, s: str) -> SessionLineResult:
        rest = s.split(None, 1)
        if len(rest) < 2:
            print("Usage: /save_context <file>")
            return SessionLineResult()
        path = rest[1].strip()
        if not path:
            print("Usage: /save_context <file>")
            return SessionLineResult()
        try:
            self._save_context_bundle(path, self.messages, "", None, False)
        except OSError as e:
            print(f"/save_context error: {e}")
            return SessionLineResult()
        self.session_save_path = path
        print(f"Wrote current session to {path!r}; further turns auto-save there.")
        return SessionLineResult()

    def _cmd_settings(self, s: str) -> SessionLineResult:
        try:
            toks = shlex.split(s)
        except ValueError as e:
            print(f"/settings: {e}")
            return SessionLineResult()
        if len(toks) < 2:
            print("Usage: /settings <topic> ...   (try: /settings help)")
            return SessionLineResult()
        key = toks[1].lower().replace("-", "_")
        if key in ("help", "-h", "--help"):
            print(
                "Usage:\n"
                "  /settings save\n"
                "  /settings model <ollama-model>\n"
                "  /settings enable|disable <feature/tool>\n"
                "  /settings tools ...\n"
                "  /settings system_prompt ...\n"
                "  /settings prompt_template ...\n"
                "  /settings context ...\n"
                "  /settings thinking ...\n"
                "  /settings ollama|openai|agent show|keys|set|unset\n"
            )
            return SessionLineResult()

        # group-backed settings
        if key in ("ollama", "openai", "agent"):
            if len(toks) < 3:
                print(
                    f"Usage: /settings {key} show | keys | set <name> <value> | unset <name>\n"
                    "  Keys are lowercase (e.g. host, model, api_key). After changing, use /settings save."
                )
                print(self._settings_group_keys_lines(key))
                return SessionLineResult()
            sub = toks[2].lower()
            if sub in ("show", "list"):
                try:
                    print(self._settings_group_show(key))
                except (ValueError, OSError) as e:
                    print(f"/settings {key} show: {e}")
                return SessionLineResult()
            if sub in ("keys", "key", "help"):
                try:
                    print(self._settings_group_keys_lines(key))
                except (ValueError, OSError) as e:
                    print(f"/settings {key} keys: {e}")
                return SessionLineResult()
            if sub == "set":
                if len(toks) < 4:
                    print(f"Usage: /settings {key} set <name> <value (optional, quote spaces with shlex)>")
                    return SessionLineResult()
                raw_k = toks[3]
                value = " ".join(toks[4:]) if len(toks) > 4 else ""
                try:
                    msg = self._settings_group_set(key, raw_k, value)
                except ValueError as e:
                    print(f"/settings {key} set: {e}")
                    return SessionLineResult()
                print(msg)
                return SessionLineResult()
            if sub in ("unset", "delete", "clear"):
                if len(toks) < 4:
                    print(f"Usage: /settings {key} unset <name>")
                    return SessionLineResult()
                try:
                    msg = self._settings_group_unset(key, toks[3])
                except ValueError as e:
                    print(f"/settings {key} unset: {e}")
                    return SessionLineResult()
                print(msg)
                return SessionLineResult()
            print(f"Unknown /settings {key} subcommand. Try: /settings {key} show | set | unset | keys")
            return SessionLineResult()

        if key == "verbose":
            if len(toks) != 3:
                print("Usage: /settings verbose 0|1|2|on|off")
                return SessionLineResult()
            tok = toks[2].strip().lower()
            if tok == "on":
                self.verbose = 2
            elif tok == "off":
                self.verbose = 0
            elif tok in ("0", "1", "2"):
                self.verbose = int(tok)
            else:
                print("Usage: /settings verbose 0|1|2|on|off")
                return SessionLineResult()
            print(self._verbose_ack_message(self.verbose))
            return SessionLineResult()

        if key == "tools":
            if len(toks) == 2 or (len(toks) >= 3 and toks[2].lower() in ("list", "ls", "show")):
                print(self._registry.format_settings_tools_list(self.enabled_tools))
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
                    lines.append("Enable a toolset:  /settings tools enable <toolset>")
                    lines.append("Disable a toolset: /settings tools disable <toolset>")
                    lines.append("Reload plugins:    /settings tools reload")
                    lines.append("Describe a tool:   /settings tools describe <tool-id>")
                    print("\n".join(lines))
                return SessionLineResult()
            if len(toks) >= 4 and toks[2].lower() in ("enable", "on"):
                nm = toks[3].strip().lower()
                plugin_toolsets = self._registry.plugin_toolsets
                if nm in plugin_toolsets:
                    self.enabled_toolsets.add(nm)
                    for tid in self._registry.plugin_tools_for_toolset(nm):
                        self.enabled_tools.add(tid)
                    print(
                        f"Toolset enabled: {nm!r} (tools may be routed per request). Use /settings save to persist."
                    )
                else:
                    print(f"Unknown toolset {nm!r}. Try: /settings tools")
                return SessionLineResult()
            if len(toks) >= 4 and toks[2].lower() in ("disable", "off"):
                nm = toks[3].strip().lower()
                plugin_toolsets = self._registry.plugin_toolsets
                if nm in plugin_toolsets:
                    self.enabled_toolsets.discard(nm)
                    for tid in self._registry.plugin_tools_for_toolset(nm):
                        self.enabled_tools.discard(tid)
                    print(f"Toolset disabled: {nm!r}. Use /settings save to persist.")
                else:
                    print(f"Unknown toolset {nm!r}. Try: /settings tools")
                return SessionLineResult()
            if len(toks) >= 3 and toks[2].lower() in ("reload", "refresh"):
                self._registry.load_plugin_toolsets(self.tools_dir)
                self._registry.register_aliases()
                print(f"Reloaded plugin toolsets from {self.tools_dir!r}.")
                return SessionLineResult()
            if len(toks) >= 4 and toks[2].lower() in ("describe", "desc", "help"):
                tid = toks[3].strip()
                if not tid:
                    print("Usage: /settings tools describe <tool-id>")
                    return SessionLineResult()
                nm = tid.strip().lower()
                plugin_toolsets = self._registry.plugin_toolsets
                if nm in plugin_toolsets:
                    rec = plugin_toolsets.get(nm) or {}
                    desc = str(rec.get("description") or "").strip()
                    print(f"Toolset: {nm}\nDescription: {desc if desc else '(none)'}")
                    print("Tools:")
                    for one in sorted(self._registry.plugin_tools_for_toolset(nm)):
                        print("  - " + one)
                    return SessionLineResult()
                print(self._registry.describe_tool_call_contract(tid))
                return SessionLineResult()
            print("Usage: /settings tools [list] | enable <toolset> | disable <toolset>")
            return SessionLineResult()

        if key == "system_prompt":
            if len(toks) < 3:
                print(
                    "Usage:\n"
                    "  /settings system_prompt show\n"
                    "  /settings system_prompt reset\n"
                    "  /settings system_prompt file <path>\n"
                    "  /settings system_prompt save <path>\n"
                    "  /settings system_prompt <text>\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub == "show":
                body = agent_prompts.effective_system_instruction_text(self.session_system_prompt)
                print(f"Effective system prompt ({len(body)} chars):\n{body}")
                if self.session_system_prompt_path:
                    print(f"(File-backed: {self.session_system_prompt_path!r})")
                elif self.session_system_prompt is not None:
                    print("(Session inline override.)")
                else:
                    print("(Built-in default.)")
                return SessionLineResult()
            if sub in ("reset", "default"):
                self.session_system_prompt = None
                self.session_system_prompt_path = None
                print("System prompt reset to built-in default for this session.")
                return SessionLineResult()
            if sub == "file":
                if len(toks) < 4:
                    print("Usage: /settings system_prompt file <path>")
                    return SessionLineResult()
                path = os.path.expanduser(" ".join(toks[3:]).strip())
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        body = f.read()
                except OSError as e:
                    print(f"/settings system_prompt file: {e}")
                    return SessionLineResult()
                if not body.strip():
                    print("File is empty.")
                    return SessionLineResult()
                self.session_system_prompt = body
                self.session_system_prompt_path = os.path.abspath(path)
                print(
                    f"System prompt loaded from {path!r} ({len(body)} chars). "
                    "/settings save will store this path in ~/.agent.json."
                )
                return SessionLineResult()
            if sub == "save":
                if len(toks) < 4:
                    print("Usage: /settings system_prompt save <path>")
                    return SessionLineResult()
                path = os.path.expanduser(" ".join(toks[3:]).strip())
                body = agent_prompts.effective_system_instruction_text(self.session_system_prompt)
                try:
                    parent = os.path.dirname(path)
                    if parent and not os.path.isdir(parent):
                        os.makedirs(parent, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(body)
                except OSError as e:
                    print(f"/settings system_prompt save: {e}")
                    return SessionLineResult()
                print(f"Wrote system prompt ({len(body)} chars) to {path!r}.")
                return SessionLineResult()
            phrase = " ".join(toks[2:])
            if not phrase.strip():
                print("Usage: /settings system_prompt <non-empty one-line text>")
                return SessionLineResult()
            self.session_system_prompt = phrase
            self.session_system_prompt_path = None
            print(
                f"System prompt set inline ({len(phrase)} chars). "
                "/settings save will store the text in ~/.agent.json."
            )
            return SessionLineResult()

        if key in ("prompt_template", "prompt_templates", "prompt"):
            if len(toks) < 3:
                print(
                    "Usage:\n"
                    "  /settings prompt_template list\n"
                    "  /settings prompt_template show\n"
                    "  /settings prompt_template use <name>\n"
                    "  /settings prompt_template default <name>\n"
                    "  /settings prompt_template set <name> <text>\n"
                    "  /settings prompt_template delete <name>\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub in ("help", "-h", "--help", "explain"):
                print("Try: /settings prompt_template list")
                return SessionLineResult()
            if sub == "list":
                names = sorted(self.prompt_templates.keys())
                if not names:
                    print("(no prompt templates)")
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
                    print(line)
                return SessionLineResult()
            if sub == "show":
                active = self.session_prompt_template or self.template_default
                body = agent_prompts.resolve_prompt_template_text(active, self.prompt_templates) or ""
                print(f"Active template: {active!r}\nPrompt ({len(body)} chars):\n{body}")
                return SessionLineResult()
            if sub in ("use", "select"):
                if len(toks) < 4:
                    print("Usage: /settings prompt_template use <name>")
                    return SessionLineResult()
                nm = toks[3].strip()
                if nm not in self.prompt_templates:
                    print(f"Unknown template {nm!r}. Try: /settings prompt_template list")
                    return SessionLineResult()
                resolved = agent_prompts.resolve_prompt_template_text(nm, self.prompt_templates)
                if not resolved:
                    print(f"Template {nm!r} has no usable text/path.")
                    return SessionLineResult()
                self.session_system_prompt = resolved
                self.session_system_prompt_path = None
                self.session_prompt_template = nm
                print(f"Using prompt template {nm!r} for this session.")
                return SessionLineResult()
            if sub == "default":
                if len(toks) < 4:
                    print("Usage: /settings prompt_template default <name>")
                    return SessionLineResult()
                nm = toks[3].strip()
                if nm not in self.prompt_templates:
                    print(f"Unknown template {nm!r}. Try: /settings prompt_template list")
                    return SessionLineResult()
                self.template_default = nm
                print(f"Default prompt template set to {nm!r} (use /settings save to persist).")
                return SessionLineResult()
            if sub == "set":
                if len(toks) < 5:
                    print("Usage: /settings prompt_template set <name> <text>")
                    return SessionLineResult()
                nm = toks[3].strip()
                text = " ".join(toks[4:]).strip()
                if not nm:
                    print("Template name must be non-empty.")
                    return SessionLineResult()
                if not text:
                    print("Template text must be non-empty.")
                    return SessionLineResult()
                cur = self.prompt_templates.get(nm) or {}
                desc = str(cur.get("description") or "") if isinstance(cur, dict) else ""
                self.prompt_templates[nm] = {"kind": "overlay", "description": desc, "text": text}
                print(f"Template {nm!r} set/updated (overlay). Use /settings save to persist.")
                return SessionLineResult()
            if sub in ("delete", "del", "rm", "remove"):
                if len(toks) < 4:
                    print("Usage: /settings prompt_template delete <name>")
                    return SessionLineResult()
                nm = toks[3].strip()
                on_disk = os.path.join(self.prompt_templates_dir, f"{nm}.json")
                if os.path.isfile(on_disk):
                    print(
                        "Refusing to delete a template that exists as a file on disk in "
                        f"the configured prompt_templates_dir ({self.prompt_templates_dir!r}). "
                        "You can override it in ~/.agent.json with a same-named entry."
                    )
                    return SessionLineResult()
                if nm not in self.prompt_templates:
                    print(f"Unknown template {nm!r}.")
                    return SessionLineResult()
                self.prompt_templates.pop(nm, None)
                if self.session_prompt_template == nm:
                    self.session_prompt_template = None
                print(f"Deleted template {nm!r}. Use /settings save to persist.")
                return SessionLineResult()
            print("Unknown subcommand. Try: /settings prompt_template list")
            return SessionLineResult()

        if key in ("context", "context_manager", "context_window"):
            if len(toks) < 3:
                print(
                    "Usage:\n"
                    "  /settings context show\n"
                    "  /settings context on|off\n"
                    "  /settings context tokens <n>\n"
                    "  /settings context trigger <0..1>\n"
                    "  /settings context target <0..1>\n"
                    "  /settings context keep_tail <n>\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub == "show":
                print(
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
                print("Context manager enabled for this session. Use /settings save to persist.")
                return SessionLineResult()
            if sub in ("off", "disable", "disabled", "false"):
                self.context_cfg["enabled"] = False
                print("Context manager disabled for this session. Use /settings save to persist.")
                return SessionLineResult()
            if sub == "tokens":
                if len(toks) < 4:
                    print("Usage: /settings context tokens <n>")
                    return SessionLineResult()
                try:
                    n = int(toks[3], 10)
                except ValueError:
                    print("tokens must be an integer.")
                    return SessionLineResult()
                if n < 0:
                    n = 0
                self.context_cfg["tokens"] = n
                print(f"context tokens set to {n} (0 = auto). Use /settings save to persist.")
                return SessionLineResult()
            if sub == "trigger":
                if len(toks) < 4:
                    print("Usage: /settings context trigger <0..1>")
                    return SessionLineResult()
                try:
                    x = float(toks[3])
                except ValueError:
                    print("trigger must be a number.")
                    return SessionLineResult()
                self.context_cfg["trigger_frac"] = max(0.05, min(0.95, x))
                print(f"trigger_frac set to {self.context_cfg['trigger_frac']}. Use /settings save to persist.")
                return SessionLineResult()
            if sub == "target":
                if len(toks) < 4:
                    print("Usage: /settings context target <0..1>")
                    return SessionLineResult()
                try:
                    x = float(toks[3])
                except ValueError:
                    print("target must be a number.")
                    return SessionLineResult()
                cur_tr = float(self.context_cfg.get("trigger_frac", 0.75))
                self.context_cfg["target_frac"] = max(0.05, min(cur_tr, x))
                print(f"target_frac set to {self.context_cfg['target_frac']}. Use /settings save to persist.")
                return SessionLineResult()
            if sub in ("keep_tail", "keep", "tail"):
                if len(toks) < 4:
                    print("Usage: /settings context keep_tail <n>")
                    return SessionLineResult()
                try:
                    n = int(toks[3], 10)
                except ValueError:
                    print("keep_tail must be an integer.")
                    return SessionLineResult()
                self.context_cfg["keep_tail_messages"] = max(4, n)
                print(
                    f"keep_tail_messages set to {self.context_cfg['keep_tail_messages']}. Use /settings save to persist."
                )
                return SessionLineResult()
            print("Unknown subcommand. Try: /settings context show")
            return SessionLineResult()

        if key == "save":
            if len(toks) != 2:
                print("Usage: /settings save")
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
                    system_prompt_override=self.session_system_prompt,
                    system_prompt_path_override=self.session_system_prompt_path,
                    prompt_templates=self.prompt_templates,
                    prompt_template_default=self.template_default,
                    prompt_templates_dir=self.prompt_templates_dir,
                    skills_dir=self.skills_dir,
                    tools_dir=self.tools_dir,
                    context_manager=self.context_cfg,
                    verbose_level=self.verbose,
                )
                self._write_agent_prefs_file(payload)
            except OSError as e:
                print(f"/settings save error: {e}")
                return SessionLineResult()
            print(f"Saved settings to {self._agent_prefs_path()!r}.")
            return SessionLineResult()

        if key == "model":
            if len(toks) < 3:
                print("Usage: /settings model <ollama-model-name>")
                return SessionLineResult()
            name = toks[2].strip()
            if not name:
                print("Usage: /settings model <ollama-model-name>")
                return SessionLineResult()
            self._settings_set(("ollama", "model"), name)
            print(f"ollama.model set to {name!r}. Use /settings save to persist.")
            return SessionLineResult()

        if key == "enable":
            if len(toks) < 3:
                print(
                    "Usage: /settings enable second_opinion|<tool or phrase>\n"
                    "  Examples: /settings enable web search   /settings enable shell   /settings enable stream_thinking\n"
                    "  See: /settings tools"
                )
                return SessionLineResult()
            phrase = " ".join(toks[2:])
            feat = self._registry.canonicalize_user_tool_phrase(phrase)
            if feat == "second_opinion":
                self.second_opinion_on = True
                print("second_opinion enabled for this session.")
                return SessionLineResult()
            if feat in ("stream_thinking", "streamthinking", "stream_think", "thinking_stream", "showthinking", "show_thinking"):
                self._settings_set(("agent", "stream_thinking"), True)
                print(
                    "stream_thinking enabled for this session (streams model thinking when available). Use /settings save to persist."
                )
                return SessionLineResult()
            if feat == "verbose":
                self.verbose = 2
                print(self._verbose_ack_message(self.verbose))
                return SessionLineResult()
            tn = self._registry.normalize_tool_name(phrase)
            if tn:
                self.enabled_tools.add(tn)
                print(f"Tool enabled: {tn}")
                return SessionLineResult()
            print(self._registry.format_unknown_tool_hint(phrase))
            return SessionLineResult()

        if key == "disable":
            if len(toks) < 3:
                print(
                    "Usage: /settings disable second_opinion|<tool or phrase>\n"
                    "  Examples: /settings disable web search   /settings disable shell   /settings disable stream_thinking\n"
                    "  See: /settings tools"
                )
                return SessionLineResult()
            phrase = " ".join(toks[2:])
            feat = self._registry.canonicalize_user_tool_phrase(phrase)
            if feat == "second_opinion":
                self.second_opinion_on = False
                print("second_opinion disabled for this session.")
                return SessionLineResult()
            if feat in ("stream_thinking", "streamthinking", "stream_think", "thinking_stream", "showthinking", "show_thinking"):
                self._settings_set(("agent", "stream_thinking"), False)
                print("stream_thinking disabled for this session. Use /settings save to persist.")
                return SessionLineResult()
            if feat == "verbose":
                self.verbose = 0
                print(self._verbose_ack_message(self.verbose))
                return SessionLineResult()
            tn = self._registry.normalize_tool_name(phrase)
            if tn:
                self.enabled_tools.discard(tn)
                print(f"Tool disabled: {tn}")
                return SessionLineResult()
            print(self._registry.format_unknown_tool_hint(phrase))
            return SessionLineResult()

        if key == "thinking":
            if len(toks) < 3:
                print(
                    "Usage:\n"
                    "  /settings thinking show\n"
                    "  /settings thinking on|off\n"
                    "  /settings thinking level low|medium|high\n"
                    "Notes:\n"
                    "  - This controls the Ollama request `think` field (bool or level string).\n"
                    "  - Some models ignore booleans and require levels; others support both.\n"
                    "  - thinking on/level also enables stream_thinking automatically (use /settings disable stream_thinking to hide).\n"
                    "  - Use /settings save to persist.\n"
                )
                return SessionLineResult()
            sub = toks[2].lower()
            if sub == "show":
                think_v = self._ollama_request_think_value()
                lvl = self._agent_thinking_level()
                on = self._agent_thinking_enabled_default_false()
                st = "on" if on else "off"
                print(
                    f"thinking: {st}; level: {lvl or '(none)'}; ollama think value: {think_v!r}; stream_thinking: {self._agent_stream_thinking_enabled()}"
                )
                return SessionLineResult()
            if sub in ("on", "enable", "enabled", "true"):
                self._settings_set(("agent", "thinking"), True)
                self._settings_set(("agent", "stream_thinking"), True)
                print(
                    "thinking enabled for this session (and stream_thinking enabled). Use /settings save to persist."
                )
                return SessionLineResult()
            if sub in ("off", "disable", "disabled", "false"):
                self._settings_set(("agent", "thinking"), False)
                self._settings_set(("agent", "thinking_level"), "")
                self._settings_set(("agent", "stream_thinking"), False)
                print(
                    "thinking disabled for this session (and stream_thinking disabled). Use /settings save to persist."
                )
                return SessionLineResult()
            if sub == "level":
                if len(toks) < 4:
                    print("Usage: /settings thinking level low|medium|high")
                    return SessionLineResult()
                lvl = toks[3].strip().lower()
                if lvl not in ("low", "medium", "high"):
                    print("thinking level must be one of: low, medium, high")
                    return SessionLineResult()
                self._settings_set(("agent", "thinking_level"), lvl)
                self._settings_set(("agent", "thinking"), True)
                self._settings_set(("agent", "stream_thinking"), True)
                print(
                    f"thinking level set to {lvl!r} for this session (and stream_thinking enabled). Use /settings save to persist."
                )
                return SessionLineResult()
            print("Unknown /settings thinking subcommand. Try: /settings thinking show | on | off | level …")
            return SessionLineResult()

        if key == "primary" and len(toks) >= 4 and toks[2].lower() == "llm":
            sub = toks[3].lower()
            if sub == "ollama":
                self.primary_profile = self._default_primary_llm_profile()
                print("Primary LLM: local Ollama.")
            elif sub == "hosted":
                if len(toks) < 6:
                    print("Usage: /settings primary llm hosted <base_url> <model> [api_key]")
                    return SessionLineResult()
                bu, mod = toks[4], toks[5]
                if not bu.startswith(("http://", "https://")):
                    print("base_url must start with http:// or https://")
                    return SessionLineResult()
                keyval = toks[6] if len(toks) > 6 else ""
                self.primary_profile = self._LlmProfile(
                    backend="hosted",
                    base_url=bu,
                    model=mod,
                    api_key=keyval,
                )
                if not (keyval or "").strip():
                    print("Note: api_key is not set; hosted primary calls will fail until it is.")
                print(
                    "Primary LLM: hosted OpenAI-compatible API "
                    f"({self._describe_llm_profile_short(self.primary_profile)})."
                )
            else:
                print("Usage: /settings primary llm ollama|hosted …")
            return SessionLineResult()

        if toks[1].replace("-", "_").lower() == "second_opinion" and len(toks) >= 4 and toks[2].lower() == "llm":
            sub = toks[3].lower()
            if sub == "ollama":
                self.reviewer_hosted_profile = None
                self.reviewer_ollama_model = toks[4] if len(toks) > 4 else None
                om = self.reviewer_ollama_model or self._ollama_second_opinion_model()
                print(f"Second-opinion reviewer: local Ollama, model {om!r}.")
            elif sub == "hosted":
                if len(toks) < 6:
                    print("Usage: /settings second_opinion llm hosted <base_url> <model> [api_key]")
                    return SessionLineResult()
                bu, mod = toks[4], toks[5]
                if not bu.startswith(("http://", "https://")):
                    print("base_url must start with http:// or https://")
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
                    print("Note: api_key is not set; hosted second opinion will fail until it is.")
                print(
                    "Second-opinion reviewer: hosted "
                    f"({self._describe_llm_profile_short(self.reviewer_hosted_profile)})."
                )
            else:
                print("Usage: /settings second_opinion llm ollama|hosted …")
            return SessionLineResult()

        print("Unknown /settings subcommand. Try /help.")
        return SessionLineResult()

