from agentlib.tui_parse import (
    format_fork_command_line,
    format_send_command_line,
    parse_fork_background_command,
    parse_fork_command,
    parse_kill_command,
    parse_send_command,
)


def test_parse_fork_name_only():
    assert parse_fork_command("/fork Experiment") == ("Experiment", [])
    assert parse_fork_command("  /fork  Agent2  ") == ("Agent2", [])


def test_parse_fork_single_unquoted_tail():
    assert parse_fork_command("/fork Bob hello world") == ("Bob", ["hello world"])


def test_parse_fork_quoted_csv():
    assert parse_fork_command('/fork R "one,two , three"') == ("R", ["one", "two", "three"])


def test_parse_fork_comma_inside_single_quotes():
    """Commas inside '...' are literal; commas outside still split."""
    assert parse_fork_command('/fork R "\'hello, world\', second"') == ("R", ["hello, world", "second"])


def test_parse_fork_backslash_comma_outside_quotes():
    r""" \, is a literal comma and does not start a new command."""
    assert parse_fork_command(r'/fork R "a\,b, c "') == ("R", ["a,b", "c"])


def test_parse_fork_roundtrip_cmds_with_commas():
    line = format_fork_command_line("Lane", ["x,y", "z"])
    assert parse_fork_command(line) == ("Lane", ["x,y", "z"])


def test_parse_fork_unclosed_single_quote_invalid():
    assert parse_fork_command('/fork X "start,\'no close"') is None


def test_parse_fork_not_fork():
    assert parse_fork_command("/help") is None
    assert parse_fork_command("") is None


def test_parse_fork_invalid():
    assert parse_fork_command("/fork") is None
    assert parse_fork_command('/fork Name "unclosed') is None


def test_parse_fork_shlex_quoted_name():
    assert parse_fork_command("/fork 'Bob Jr'") == ("Bob Jr", [])


def test_format_fork_roundtrip():
    assert parse_fork_command(format_fork_command_line("Reviewer", ["a", "b"])) == (
        "Reviewer",
        ["a", "b"],
    )
    line = format_fork_command_line("Bob Jr", [])
    assert parse_fork_command(line) == ("Bob Jr", [])


def test_parse_send_unquoted_multiword():
    assert parse_send_command("/send Bob hello world") == ("Bob", ["hello world"])


def test_parse_send_quoted_csv():
    assert parse_send_command('/send Planner "one,two , three"') == ("Planner", ["one", "two", "three"])


def test_parse_send_comma_inside_single_quotes():
    assert parse_send_command('/send R "\'hello, world\', second"') == ("R", ["hello, world", "second"])


def test_parse_send_empty_inner_invalid():
    assert parse_send_command('/send Bob ""') is None


def test_parse_send_not_send():
    assert parse_send_command("/fork Bob hi") is None


def test_format_send_roundtrip():
    line = format_send_command_line("Lane", ["x,y", "/help"])
    assert parse_send_command(line) == ("Lane", ["x,y", "/help"])


def test_parse_fork_background_matches_fork_body():
    assert parse_fork_background_command('/fork_background X "a,b"') == parse_fork_command(
        '/fork X "a,b"'
    )
    assert parse_fork_background_command("/fork_background Y hello") == parse_fork_command(
        "/fork Y hello"
    )


def test_parse_kill_basic():
    assert parse_kill_command("/kill Coder") == "Coder"
    assert parse_kill_command("  /kill  x  ") == "x"


def test_parse_kill_quoted():
    assert parse_kill_command('/kill "Agent 2"') == "Agent 2"


def test_parse_kill_invalid():
    assert parse_kill_command("/kill") is None
    assert parse_kill_command('/kill "oops') is None
    assert parse_kill_command("/fork x") is None
