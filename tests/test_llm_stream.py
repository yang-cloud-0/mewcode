import json

from mewcode.llm import assemble_stream, iter_sse_json


def _chunk(delta):
    return {"choices": [{"delta": delta}]}


def test_content_fragments_concatenate():
    chunks = [_chunk({"content": "Hel"}), _chunk({"content": "lo"}), _chunk({"content": ""})]
    reply = assemble_stream(chunks)
    assert reply.content == "Hello"
    assert reply.tool_calls == []


def test_tool_call_arguments_assembled_from_fragments():
    # One tool call split across three chunks: name first, then the
    # arguments JSON arriving in two string fragments.
    chunks = [
        _chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "write_file", "arguments": ""}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\": \"a.txt\","}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": " \"content\": \"hi\"}"}}]}),
    ]
    reply = assemble_stream(chunks)
    assert len(reply.tool_calls) == 1
    tc = reply.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "write_file"
    assert tc.parsed_arguments() == {"path": "a.txt", "content": "hi"}


def test_two_parallel_tool_calls_kept_separate_by_index():
    chunks = [
        _chunk({"tool_calls": [
            {"index": 0, "id": "c0", "function": {"name": "read_file", "arguments": "{\"path\": \"a\"}"}},
            {"index": 1, "id": "c1", "function": {"name": "read_file", "arguments": "{\"path\": \"b\"}"}},
        ]}),
    ]
    reply = assemble_stream(chunks)
    assert [tc.parsed_arguments()["path"] for tc in reply.tool_calls] == ["a", "b"]


def test_iter_sse_json_skips_done_and_blank_lines():
    lines = [
        "",
        'data: {"choices": [{"delta": {"content": "x"}}]}',
        "data: [DONE]",
        'data: {"choices": []}',  # after DONE: must not be yielded
    ]
    objs = list(iter_sse_json(lines))
    assert len(objs) == 1
    assert objs[0]["choices"][0]["delta"]["content"] == "x"


def test_malformed_arguments_raise_value_error():
    reply = assemble_stream([
        _chunk({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "not-json{"}}]})
    ])
    try:
        reply.tool_calls[0].parsed_arguments()
        raise AssertionError("expected ValueError")
    except (ValueError, json.JSONDecodeError):
        pass
