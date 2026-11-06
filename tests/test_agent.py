import json

from mewcode.agent import Agent
from mewcode.llm import AssistantReply, FakeLLM, tool_call_reply
from mewcode.tools import ToolBox, Workspace


def make_agent(tmp_path, script, **kwargs):
    toolbox = ToolBox(Workspace(tmp_path), confirm=lambda cmd: True)
    return Agent(FakeLLM(script), toolbox, **kwargs)


def test_write_and_finish(tmp_path):
    agent = make_agent(tmp_path, [
        tool_call_reply("write_file", {"path": "note.txt", "content": "hi"}),
        AssistantReply(content="done"),
    ])
    result = agent.run("write a note")
    assert result.stopped_reason == "done"
    assert result.steps == 2
    assert (tmp_path / "note.txt").read_text() == "hi"


def test_tool_error_is_fed_back_and_model_recovers(tmp_path):
    # First turn: the fake model asks for a file that does not exist.
    # The loop must NOT crash — it feeds {"ok": false, ...} back, and the
    # model (second script item, a callable) checks that feedback, then
    # recovers by creating the file instead.
    def second_turn(messages):
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert tool_msgs, "expected a tool feedback message"
        payload = json.loads(tool_msgs[-1]["content"])
        assert payload["ok"] is False
        assert "not found" in payload["error"]
        return tool_call_reply("write_file", {"path": "missing.txt", "content": "created"})

    agent = make_agent(tmp_path, [
        tool_call_reply("read_file", {"path": "missing.txt"}),
        second_turn,
        AssistantReply(content="recovered"),
    ])
    result = agent.run("read missing.txt, create it if absent")
    assert result.stopped_reason == "done"
    assert result.final_text == "recovered"
    assert (tmp_path / "missing.txt").read_text() == "created"


def test_invalid_json_arguments_become_error_feedback(tmp_path):
    from mewcode.llm import ToolCall
    bad = AssistantReply(tool_calls=[ToolCall(id="c1", name="write_file", arguments="{oops")])
    agent = make_agent(tmp_path, [bad, AssistantReply(content="ok")])
    result = agent.run("anything")
    assert result.stopped_reason == "done"
    tool_msg = [m for m in result.transcript if m.get("role") == "tool"][0]
    assert json.loads(tool_msg["content"])["ok"] is False


def test_max_steps_stops_endless_tool_calls(tmp_path):
    endless = [tool_call_reply("grep_search", {"pattern": f"p{i}"}) for i in range(50)]
    agent = make_agent(tmp_path, endless, max_steps=5)
    result = agent.run("search forever")
    assert result.stopped_reason == "max_steps"
    assert result.steps == 5


def test_repeated_identical_call_is_detected(tmp_path):
    same = [tool_call_reply("grep_search", {"pattern": "same"}) for _ in range(10)]
    agent = make_agent(tmp_path, same, repeat_limit=3)
    result = agent.run("loop please")
    assert result.stopped_reason == "repeated_calls"


def test_shell_end_to_end(tmp_path):
    agent = make_agent(tmp_path, [
        tool_call_reply("write_file", {"path": "hello.py", "content": "print('mew')\n"}),
        tool_call_reply("run_shell", {"command": "python3 hello.py"}),
        AssistantReply(content="ran it"),
    ])
    result = agent.run("create and run hello.py")
    assert result.stopped_reason == "done"
    tool_msgs = [m for m in result.transcript if m.get("role") == "tool"]
    assert "mew" in json.loads(tool_msgs[-1]["content"])["result"]
