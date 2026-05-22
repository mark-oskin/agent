"""JSON string field extraction for streaming answers."""

from agentlib.agent_json import _extract_json_string_field


def test_extract_answer_use_last():
    raw = (
        '{"action":"answer","answer":"old"}'
        '{"action":"answer","answer":"new"}'
    )
    assert _extract_json_string_field(raw, "answer", use_last=True) == "new"
    assert _extract_json_string_field(raw, "answer", use_last=False) == "old"


def test_extract_answer_unterminated_stops_before_glued_object():
    raw = '{"action":"answer","answer":"body text"},{"action":"answer","answer":"x"}'
    assert (
        _extract_json_string_field(raw, "answer", allow_unterminated=True, use_last=False)
        == "body text"
    )
