"""Tests for ``/compact`` REPL command (LLM transcript compression)."""

from __future__ import annotations

from agentlib.context.compaction import count_words, llm_compress_transcript_for_repl
from agentlib.embedding import build_embedded_session
from agentlib.llm.profile import LlmProfile


def test_count_words_basic():
    assert count_words("one two three") == 3
    assert count_words("") == 0


def test_llm_compress_transcript_truncates_words():
    def fake_plain(msgs, model):  # noqa: ARG001
        return " ".join([f"w{i}" for i in range(20)])

    out = llm_compress_transcript_for_repl(
        profile=LlmProfile(backend="ollama", model="m"),
        transcript="user:\nhello\n\nassistant:\nworld\n",
        constraint_kind="max_words",
        constraint_value=5,
        call_hosted_chat_plain=lambda *_a, **_k: "",
        call_ollama_plaintext=fake_plain,
        ollama_model="m",
    )
    assert "w4" in out
    assert "w5" not in out


def test_compact_repl_replaces_messages(monkeypatch, capsys):
    _, sess = build_embedded_session(verbose=0)

    def fake_compress(**kwargs):  # noqa: ARG001
        return "COMPRESSED_BODY"

    monkeypatch.setattr("agentlib.session.llm_compress_transcript_for_repl", fake_compress)

    sess.messages.append({"role": "user", "content": "first question"})
    sess.messages.append({"role": "assistant", "content": "first answer"})
    sess.last_reuse_skill_id = "sid"
    sess.repl_last_user_query = "q"
    sess.repl_last_assistant_answer = "a"

    sess.execute_line("/compact 50")
    cap = capsys.readouterr()
    assert "replaced history" in cap.out
    assert len(sess.messages) == 1
    assert sess.messages[0].get("role") == "system"
    assert "COMPRESSED_BODY" in (sess.messages[0].get("content") or "")
    assert sess.last_reuse_skill_id is None
    assert sess.repl_last_user_query is None
    assert sess.repl_last_assistant_answer is None


def test_compact_repl_empty_history(capsys):
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line("/compact")
    out = capsys.readouterr().out
    assert "nothing to compress" in out.lower()


def test_llm_compress_transcript_uses_hosted_when_profile_hosted():
    def fake_hosted(msgs, profile):  # noqa: ARG001
        assert any("Transcript to compress" in (m.get("content") or "") for m in msgs if m.get("role") == "user")
        return "  HOSTED_OUT  "

    out = llm_compress_transcript_for_repl(
        profile=LlmProfile(backend="hosted", base_url="https://example.invalid/v1", model="m", api_key="k"),
        transcript="user:\nhi\n",
        constraint_kind="approx_tokens",
        constraint_value=200,
        call_hosted_chat_plain=fake_hosted,
        call_ollama_plaintext=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("ollama must not be called")),
        ollama_model="ignored",
    )
    assert out.strip() == "HOSTED_OUT"
