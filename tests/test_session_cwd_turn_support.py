"""Session cwd resolution for tool parameters (``deps.session_cwd`` / ``/cd``)."""

import os
from types import SimpleNamespace

from agentlib.coercion import scalar_to_str
from agentlib.tools import turn_support as ts


def test_apply_session_cwd_write_file_relative(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    deps = SimpleNamespace(
        session_cwd=str(root),
        scalar_to_str=lambda x, d="": scalar_to_str(x, d),
        plugin_tool_handlers={},
    )
    out = ts.apply_session_cwd_tool_params("write_file", {"path": "foo.txt", "content": "x"}, deps)
    assert out["path"] == os.path.normpath(str(root / "foo.txt"))


def test_apply_session_cwd_plugin_tool_resolves_path(tmp_path):
    deps = SimpleNamespace(
        session_cwd=str(tmp_path),
        scalar_to_str=lambda x, d="": scalar_to_str(x, d),
        plugin_tool_handlers={"custom_fs": lambda p: "ok"},
    )
    out = ts.apply_session_cwd_tool_params("custom_fs", {"path": "out.txt"}, deps)
    assert out["path"] == os.path.normpath(str(tmp_path / "out.txt"))


def test_run_command_with_session_cwd_passes_kwarg(tmp_path):
    calls: list[tuple[str, object]] = []

    def fake_run(cmd, cwd=None):
        calls.append((cmd, cwd))
        return "ok"

    root = os.path.abspath(os.path.expanduser(str(tmp_path)))
    deps = SimpleNamespace(session_cwd=str(tmp_path), run_command=fake_run)
    ts.run_command_with_session_cwd(deps, "echo hi")
    assert calls == [("echo hi", root)]
