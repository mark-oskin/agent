from fork_parse import parse_fork_command, parse_kill_command


def test_parse_fork_name_only():
    assert parse_fork_command("/fork Experiment") == ("Experiment", [])
    assert parse_fork_command("  /fork  Agent2  ") == ("Agent2", [])


def test_parse_fork_single_unquoted_tail():
    assert parse_fork_command("/fork Bob hello world") == ("Bob", ["hello world"])


def test_parse_fork_quoted_csv():
    assert parse_fork_command('/fork R "one,two , three"') == ("R", ["one", "two", "three"])


def test_parse_fork_not_fork():
    assert parse_fork_command("/help") is None
    assert parse_fork_command("") is None


def test_parse_fork_invalid():
    assert parse_fork_command("/fork") is None
    assert parse_fork_command('/fork Name "unclosed') is None


def test_parse_kill_basic():
    assert parse_kill_command("/kill Coder") == "Coder"
    assert parse_kill_command("  /kill  x  ") == "x"


def test_parse_kill_quoted():
    assert parse_kill_command('/kill "Agent 2"') == "Agent 2"


def test_parse_kill_invalid():
    assert parse_kill_command("/kill") is None
    assert parse_kill_command('/kill "oops') is None
    assert parse_kill_command("/fork x") is None
