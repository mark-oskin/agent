"""
Bridge Telegram updates (channels, bot DMs, and optionally groups) into ``ai(...)``.

Run inside an agent session so ``ai`` / ``print`` are injected by ``/call_python``::

    export TELEGRAM_BOT_TOKEN='123456:ABC...'
    export TELEGRAM_ALLOWED_CHAT_IDS='-1001234567890'   # strongly recommended
    /call_python telegram_channel_ai.py

Use an absolute path if your cwd is not the repo root.

Setup
-----
1. Create a bot with `@BotFather`, copy the token.
2. Add the bot to your **channel** as an administrator (it only receives ``channel_post``
   updates when it can read the channel).
3. Discover the channel numeric id (negative, often ``-100...``); many bots log
   ``chat.id`` on first post, or use `@userinfobot` / forwarded-message tricks.

Environment
-----------
``TELEGRAM_BOT_TOKEN`` (required)
    Bot API token.

``TELEGRAM_ALLOWED_CHAT_IDS`` (optional but recommended)
    Comma-separated integer chat ids (channels use negative ids like ``-100…``; **your own
    user id** for DMs is positive). If unset, every matching update is forwarded (risky).

``TELEGRAM_POLL_TIMEOUT`` (optional)
    Long-polling timeout in seconds (default ``30``).

``TELEGRAM_INCLUDE_GROUPS`` (optional, default **on**)
    When **on** (default), handles ``message`` / ``edited_message`` as well as channel posts —
    required for **direct messages to the bot** (those arrive as ``message``, not
    ``channel_post``). Set to ``0`` / ``false`` / ``no`` for **channels only** (ignore DMs
    and group chats).

``TELEGRAM_STRIP_BOT_MENTION`` (optional)
    Set to ``1`` to remove a leading ``@YourBot`` token from text.

``TELEGRAM_DEBUG`` (optional)
    Set to ``1`` for verbose traces: every poll result count, update shapes, skip reasons,
    and webhook status. Use when posts seem missing.

``TELEGRAM_MIRROR`` (optional, default **on**)
    Reply on Telegram with ``sendMessage`` after each handled line (slash commands + model
    answers). Uses the injected ``session`` object so assistant replies come from the return
    payload (not only streamed prints). Set to ``0`` / ``false`` to disable.

If nothing arrives
~~~~~~~~~~~~~~~~~~
On startup we call ``deleteWebhook`` — if a webhook was set, ``getUpdates`` would otherwise stay
empty. Still stuck? Try ``TELEGRAM_DEBUG=1``, confirm ``TELEGRAM_ALLOWED_CHAT_IDS`` matches the
real ``chat_id`` (debug prints it). For **channels-only** mode set ``TELEGRAM_INCLUDE_GROUPS=0``.
Ensure no second script polls the same bot token (Telegram returns a Conflict error).

Stopping
--------
This loop runs until the host stops the session (or the OS ends the process). In the
multi-agent TUI, use a dedicated lane: the turn stays *busy* while this script runs.
"""

from __future__ import annotations

import builtins
import os
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

import requests

PrintFn = Callable[..., None]


def _env_bool(key: str, *, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("0", "false", "no", "off"):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    return default


def _parse_allowed_chat_ids(raw: Optional[str]) -> Optional[Set[int]]:
    if raw is None or not str(raw).strip():
        return None
    out: Set[int] = set()
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        out.add(int(p))
    return out


def _posts_from_update(update: Dict[str, Any], include_groups: bool) -> List[Dict[str, Any]]:
    posts: List[Dict[str, Any]] = []
    for key in ("channel_post", "edited_channel_post"):
        p = update.get(key)
        if isinstance(p, dict):
            posts.append(p)
    if include_groups:
        for key in ("message", "edited_message"):
            p = update.get(key)
            if isinstance(p, dict):
                posts.append(p)
    return posts


def _post_text(post: Dict[str, Any]) -> str:
    t = post.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    cap = post.get("caption")
    if isinstance(cap, str) and cap.strip():
        return cap.strip()
    return ""


def _strip_bot_username(text: str, username: Optional[str]) -> str:
    if not username:
        return text
    pat = re.compile(rf"^@{re.escape(username)}\s+", re.I)
    return pat.sub("", text).strip()


def _update_summary(update: Dict[str, Any]) -> str:
    keys = sorted(k for k in update.keys() if k != "update_id")
    return ",".join(keys) if keys else "(empty)"


def _prepare_long_poll(token: str, http: requests.Session, print_fn: PrintFn, debug: bool) -> None:
    """
    Drop webhook if configured — otherwise ``getUpdates`` often stays empty forever.

    See `<https://core.telegram.org/bots/api#getupdates>`_.
    """
    base = f"https://api.telegram.org/bot{token}"
    try:
        wi = http.get(f"{base}/getWebhookInfo", timeout=(10, 30))
        winfo = wi.json()
        if debug:
            print_fn(f"telegram_channel_ai [debug]: getWebhookInfo → {winfo}")
        res = winfo.get("result") if isinstance(winfo, dict) else {}
        url = (res or {}).get("url") or ""
        if isinstance(url, str) and url.strip():
            url_disp = url[:48] + "…" if len(url) > 48 else url
            print_fn(
                "telegram_channel_ai: webhook was active — clearing so long-polling works "
                f"(url={url_disp})"
            )
        dw = http.post(
            f"{base}/deleteWebhook",
            json={"drop_pending_updates": False},
            timeout=(10, 30),
        )
        dj = dw.json()
        if debug:
            print_fn(f"telegram_channel_ai [debug]: deleteWebhook → {dj}")
        if isinstance(dj, dict) and not dj.get("ok"):
            print_fn(f"telegram_channel_ai: deleteWebhook failed: {dj!r}")
    except requests.RequestException as e:
        print_fn(f"telegram_channel_ai: webhook cleanup request failed: {e}")
    except ValueError:
        print_fn("telegram_channel_ai: webhook cleanup returned non-JSON")


def _get_me_username(token: str, print_fn: PrintFn) -> Optional[str]:
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{token}/getMe",
            timeout=(10, 30),
        )
        r.raise_for_status()
        body = r.json()
        if not body.get("ok"):
            print_fn(f"telegram_channel_ai: getMe not ok: {body!r}")
            return None
        u = body.get("result") or {}
        un = u.get("username")
        return str(un) if un else None
    except Exception as e:
        print_fn(f"telegram_channel_ai: getMe failed: {e}")
        return None


def _fallback_stdio_emit(ev: dict) -> None:
    """Echo emit events like ``sink_emit`` does when no sink is installed (stdout/stderr).

    Must use ``builtins.print``: ``/call_python`` installs ``print`` = ``sink_print_compat`` in
    this module's globals, which would recurse back into the active emit tee.
    """
    typ = ev.get("type") or "output"
    text = ev.get("text")
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    end = ev.get("end", "\n")
    flush = bool(ev.get("flush", True))
    fil = (
        sys.stderr
        if typ in ("progress", "stderr", "warning", "debug", "error")
        else sys.stdout
    )
    builtins.print(text, end=end, file=fil, flush=flush)


def _telegram_chunk_text(text: str, limit: int = 3900) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    rest = text
    while rest:
        chunks.append(rest[:limit])
        rest = rest[limit:]
    return chunks


def _telegram_send_reply(
    http: requests.Session,
    token: str,
    chat_id: int,
    text: str,
    *,
    reply_to_message_id: Optional[int],
    print_fn: PrintFn,
    debug: bool,
) -> None:
    """Best-effort ``sendMessage`` (plain text); splits near Telegram's length limit."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = _telegram_chunk_text(text.strip()) if text.strip() else []
    if not chunks:
        chunks = ["(no reply text)"]
    for i, chunk in enumerate(chunks):
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": chunk}
        if reply_to_message_id is not None and i == 0:
            payload["reply_to_message_id"] = reply_to_message_id
        try:
            r = http.post(url, json=payload, timeout=(10, 60))
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            print_fn(f"telegram_channel_ai: sendMessage failed: {e}")
            return
        if isinstance(body, dict) and not body.get("ok"):
            print_fn(f"telegram_channel_ai: sendMessage error: {body!r}")
            return
        if debug:
            print_fn(f"telegram_channel_ai [debug]: sendMessage chunk {i + 1}/{len(chunks)} ok")


def _execute_line_mirror_to_telegram(
    agent_session: Any,
    line: str,
    *,
    chat_id: int,
    reply_mid: Optional[int],
    http: requests.Session,
    token: str,
    mirror: bool,
    print_fn: PrintFn,
    debug: bool,
) -> None:
    """
    Run ``session.execute_line`` with an emit tee so slash-command output is captured.
    Assistant turns use ``print_answer=False`` internally — final text comes from the returned
    payload ``answer`` field.
    """
    segments: List[str] = []

    def tee(ev: dict) -> None:
        _fallback_stdio_emit(ev)
        typ = ev.get("type") or "output"
        text = ev.get("text")
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        partial = bool(ev.get("partial"))
        if typ == "thinking":
            return
        if typ == "progress":
            return
        if typ == "answer":
            if text.strip():
                segments.append(text.strip())
            return
        if typ in ("stderr", "warning", "error", "debug"):
            if text.strip():
                segments.append(f"[{typ}] {text.strip()}")
            return
        if typ == "output":
            if partial:
                return
            if text.strip():
                segments.append(text.strip())

    payload = agent_session.execute_line(line, emit=tee)
    body = ""
    if isinstance(payload, dict):
        if payload.get("type") == "turn":
            ans = payload.get("answer")
            if isinstance(ans, str) and ans.strip():
                body = ans.strip()
        elif payload.get("type") == "command":
            cmd_out = payload.get("output")
            if isinstance(cmd_out, str) and cmd_out.strip():
                segments.insert(0, cmd_out.strip())

    if not body:
        body = "\n".join(segments).strip()

    # Assistant replies are not emitted when ``print_answer=False`` inside the agent turn;
    # echo the structured ``answer`` field once emit scope has ended so the host UI sees it.
    if isinstance(payload, dict) and payload.get("type") == "turn":
        ans = payload.get("answer")
        if isinstance(ans, str) and ans.strip():
            print_fn(ans.strip())
        elif payload.get("answered") and body.strip():
            print_fn(body.strip())

    if mirror:
        _telegram_send_reply(
            http,
            token,
            chat_id,
            body,
            reply_to_message_id=reply_mid,
            print_fn=print_fn,
            debug=debug,
        )
    elif debug and not body.strip():
        print_fn("telegram_channel_ai [debug]: empty mirror body (nothing to send)")


def run_listener(
    *,
    agent_session: Any,
    print_fn: PrintFn,
    bot_token: str,
    allowed_chat_ids: Optional[Set[int]] = None,
    poll_timeout: int = 30,
    include_groups: bool = True,
    strip_bot_mention: bool = False,
    stop_event: Optional[threading.Event] = None,
    debug: bool = False,
    mirror_replies: bool = True,
) -> None:
    """
    Long-poll Telegram and run each inbound text line via ``agent_session.execute_line``.

    ``stop_event``: when set, exits the loop after the current poll (for tests).
    """
    token = (bot_token or "").strip()
    if not token:
        print_fn("telegram_channel_ai: set TELEGRAM_BOT_TOKEN")
        return

    me_user: Optional[str] = None
    if strip_bot_mention:
        me_user = _get_me_username(token, print_fn)

    if allowed_chat_ids is None:
        print_fn(
            "telegram_channel_ai: TELEGRAM_ALLOWED_CHAT_IDS not set — "
            "accepting every matching update this bot receives."
        )
    elif debug:
        print_fn(f"telegram_channel_ai [debug]: TELEGRAM_ALLOWED_CHAT_IDS={sorted(allowed_chat_ids)}")

    offset: Optional[int] = None
    stop = stop_event or threading.Event()
    http = requests.Session()

    _prepare_long_poll(token, http, print_fn, debug)

    scope = (
        "channels + DM/group messages"
        if include_groups
        else "channels only (channel_post / edited_channel_post)"
    )
    print_fn(
        "telegram_channel_ai: polling getUpdates "
        f"({scope}; timeout={poll_timeout}s). Ctrl+C / stop session to exit."
    )

    polls = 0
    while not stop.is_set():
        params: Dict[str, Any] = {"timeout": poll_timeout}
        if offset is not None:
            params["offset"] = offset

        polls += 1
        try:
            r = http.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params=params,
                timeout=(10, poll_timeout + 15),
            )
        except requests.RequestException as e:
            print_fn(f"telegram_channel_ai: network error: {e}")
            time.sleep(2)
            continue

        if r.status_code != 200:
            print_fn(f"telegram_channel_ai: getUpdates HTTP {r.status_code}: {r.text[:500]}")

        try:
            body = r.json()
        except ValueError:
            print_fn(f"telegram_channel_ai: bad JSON HTTP {r.status_code}")
            time.sleep(2)
            continue

        if not body.get("ok"):
            desc = ""
            params_err = body.get("parameters") if isinstance(body, dict) else None
            if isinstance(params_err, dict):
                desc = str(params_err.get("retry_after") or "")
            err_msg = ""
            if isinstance(body, dict):
                err_msg = str(body.get("description") or "")
            print_fn(f"telegram_channel_ai: getUpdates error: {body!r}")
            if "Conflict" in err_msg or "terminated by other getUpdates" in err_msg:
                print_fn(
                    "telegram_channel_ai: hint — another process is polling this bot token "
                    "(stop other bots/scripts using the same TELEGRAM_BOT_TOKEN)."
                )
            if desc:
                print_fn(f"telegram_channel_ai: retry_after={desc}")
            time.sleep(2)
            continue

        results = body.get("result") or []
        if debug:
            print_fn(
                f"telegram_channel_ai [debug]: poll #{polls} → {len(results)} update(s); "
                f"offset={offset}"
            )

        for update in results:
            uid = update.get("update_id")
            if isinstance(uid, int):
                offset = uid + 1

            if debug:
                print_fn(
                    f"telegram_channel_ai [debug]: update_id={uid} kinds=[{_update_summary(update)}]"
                )

            posts = _posts_from_update(update, include_groups)
            if debug and not posts:
                hint = (
                    " (channels-only mode hides DM/group message updates)"
                    if not include_groups
                    else ""
                )
                print_fn(
                    "telegram_channel_ai [debug]: update had no channel/edited_channel_post"
                    + (" or message/edited_message" if include_groups else "")
                    + hint
                )

            for post in posts:
                chat = post.get("chat") or {}
                chat_id = chat.get("id")
                ctype = chat.get("type")
                if not isinstance(chat_id, int):
                    if debug:
                        print_fn("telegram_channel_ai [debug]: skip post — chat.id missing")
                    continue
                if debug:
                    print_fn(
                        f"telegram_channel_ai [debug]: post chat_id={chat_id} "
                        f"type={ctype!r} has_text={bool(post.get('text'))} "
                        f"has_caption={bool(post.get('caption'))}"
                    )
                if allowed_chat_ids is not None and chat_id not in allowed_chat_ids:
                    print_fn(
                        "telegram_channel_ai: skipped chat_id="
                        f"{chat_id} (not in TELEGRAM_ALLOWED_CHAT_IDS); "
                        "fix env or post TELEGRAM_DEBUG=1 and watch [debug] lines."
                    )
                    continue

                text = _post_text(post)
                if not text:
                    if debug:
                        keys = sorted(post.keys())
                        print_fn(
                            f"telegram_channel_ai [debug]: skip non-text post "
                            f"(chat_id={chat_id}); keys={keys}"
                        )
                    continue
                if strip_bot_mention and me_user:
                    text = _strip_bot_username(text, me_user)
                if not text:
                    continue

                preview = text if len(text) <= 120 else text[:117] + "..."
                print_fn(f"telegram_channel_ai: chat {chat_id} → execute_line({preview!r})")
                reply_mid = post.get("message_id")
                if not isinstance(reply_mid, int):
                    reply_mid = None
                try:
                    _execute_line_mirror_to_telegram(
                        agent_session,
                        text,
                        chat_id=chat_id,
                        reply_mid=reply_mid,
                        http=http,
                        token=token,
                        mirror=mirror_replies,
                        print_fn=print_fn,
                        debug=debug,
                    )
                except BaseException as e:
                    print_fn(f"telegram_channel_ai: execute_line raised {type(e).__name__}: {e}")


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    allowed = _parse_allowed_chat_ids(os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS"))
    poll_timeout = int(os.environ.get("TELEGRAM_POLL_TIMEOUT", "30"))
    include_groups = _env_bool("TELEGRAM_INCLUDE_GROUPS", default=True)
    strip_mention = os.environ.get("TELEGRAM_STRIP_BOT_MENTION", "").strip() in (
        "1",
        "true",
        "yes",
    )
    debug = os.environ.get("TELEGRAM_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    mirror = _env_bool("TELEGRAM_MIRROR", default=True)

    # Injected by AgentSession._cmd_call_python via exec globals.
    sess = session  # type: ignore[name-defined]
    print_fn = print  # type: ignore[name-defined]

    run_listener(
        agent_session=sess,
        print_fn=print_fn,
        bot_token=token,
        allowed_chat_ids=allowed,
        poll_timeout=poll_timeout,
        include_groups=include_groups,
        strip_bot_mention=strip_mention,
        debug=debug,
        mirror_replies=mirror,
    )


if __name__ == "__call_python__":
    main()
