from agentlib.sink import sink_delegate_capture_append


def test_sink_delegate_capture_append_basic():
    buf: list[str] = []
    sink_delegate_capture_append({"type": "output", "text": "Hello", "end": "\n"}, buf)
    assert buf == ["Hello\n"]


def test_sink_delegate_capture_append_skips_thinking_and_answer():
    buf: list[str] = []
    sink_delegate_capture_append({"type": "thinking", "text": "x"}, buf)
    sink_delegate_capture_append({"type": "answer", "text": "y"}, buf)
    assert buf == []


def test_sink_delegate_capture_append_stderr():
    buf: list[str] = []
    sink_delegate_capture_append({"type": "stderr", "text": "e", "end": ""}, buf)
    assert buf == ["e"]
