import pytest

from mewcode.tools import ToolBox, ToolError, Workspace


@pytest.fixture()
def box(tmp_path):
    return ToolBox(Workspace(tmp_path), confirm=lambda cmd: False)


def test_write_then_read_roundtrip(box, tmp_path):
    msg = box.write_file("a/b.txt", "hello")
    assert "wrote" in msg
    assert box.read_file("a/b.txt") == "hello"


def test_read_missing_file_raises(box):
    with pytest.raises(ToolError):
        box.read_file("nope.txt")


def test_path_escape_is_rejected(box):
    with pytest.raises(ToolError):
        box.write_file("../evil.txt", "x")
    with pytest.raises(ToolError):
        box.read_file("../../etc/passwd")


def test_grep_finds_matches(box):
    box.write_file("x.py", "def foo():\n    return 1\n")
    box.write_file("y.py", "def bar():\n    return 2\n")
    out = box.grep_search(r"def \w+")
    assert "x.py:1" in out and "y.py:1" in out


def test_grep_invalid_regex_raises(box):
    with pytest.raises(ToolError):
        box.grep_search("([unclosed")


def test_shell_safe_command_runs(box):
    out = box.run_shell("echo hello-mew")
    assert "exit_code=0" in out and "hello-mew" in out


def test_shell_blocked_command(box):
    with pytest.raises(ToolError):
        box.run_shell("rm -rf / --no-preserve-root")


def test_shell_unconfirmed_command_rejected(box):
    # 夹具里的确认回调恒为“拒绝”
    with pytest.raises(ToolError):
        box.run_shell("python3 hello.py")
