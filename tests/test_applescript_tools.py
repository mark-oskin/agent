from __future__ import annotations

import shutil
import sys

import pytest

import agentlib.tools.plugins as plugins


def test_plugin_toolset_registers_applescript_tools(tmp_path):
    from pathlib import Path

    plugins.load_plugin_toolsets(tools_dir=str(tmp_path), default_tools_dir=str(tmp_path))
    assert "run_applescript" not in plugins.PLUGIN_TOOL_HANDLERS

    import tools as tools_pkg

    base = str(Path(tools_pkg.__file__).resolve().parent)
    plugins.load_plugin_toolsets(tools_dir=None, default_tools_dir=base)
    assert "run_applescript" in plugins.PLUGIN_TOOL_HANDLERS
    assert plugins.PLUGIN_TOOL_TO_TOOLSET.get("run_applescript") == "applescript"


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("osascript") is None,
    reason="requires macOS osascript",
)
def test_run_applescript_success():
    from tools import applescript as a

    out = a.run_applescript({"script": 'return "ok"'})
    assert "(exit 0)" in out
    assert "ok" in out
    out2 = a.run_applescript({"script": 'return "ok"', "echo_script": True})
    assert "SCRIPT:\n" in out2
    assert 'return "ok"' in out2


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("osascript") is None,
    reason="requires macOS osascript",
)
def test_run_applescript_error_exit_nonzero():
    from tools import applescript as a

    out = a.run_applescript({"script": 'error "boom" number 42'})
    assert "(exit " in out and "(exit 0)" not in out
    assert "boom" in out.lower()
    out2 = a.run_applescript({"script": 'error "boom" number 42', "use_temp_file": True})
    assert "COMMAND: osascript <tempfile.applescript>" in out2
    assert "(exit " in out2 and "(exit 0)" not in out2

