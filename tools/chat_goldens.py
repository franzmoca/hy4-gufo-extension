#!/usr/bin/env python3
"""Renders tests/fixtures/chat_goldens.json with the GGUF's chat template.

    chat_goldens.py TEMPLATE.jinja OUT.json

TEMPLATE.jinja is `tokenizer.chat_template` of the model GGUF (for example
via gguf-py's GGUFReader). Rendering follows Transformers' chat template
environment (trim_blocks, lstrip_blocks, its `tojson`), with
add_generation_prompt. Each case is stored in a neutral form that
tests/chat_template_test.cpp turns into a gufo ChatRequest:

  messages: role, content, reasoning, tool_call_id, images (byte offsets
            into content), tool_calls [{id, name, arguments [{name, value,
            is_string}]}] where a non-string value is JSON text
  tools:    function definitions
  effort:   high | max | no_think;  tool_choice: auto | none
"""
import hashlib
import json
import sys

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

IMAGE = {"type": "image_url", "image_url": {"url": "data:"}}

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city, in °C.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "minimum": 1},
            },
            "required": ["city"],
        },
        "strict": True,
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the <web> & return \"results\".",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}
DEFERRED = {
    "type": "function",
    "function": {"name": "later", "description": "Loaded on demand.", "parameters": {}, "defer_loading": True},
}


def call(id_, name, **arguments):
    args = []
    for key, value in arguments.items():
        if isinstance(value, str):
            args.append({"name": key, "value": value, "is_string": True})
        else:
            args.append({"name": key, "value": json.dumps(value, ensure_ascii=False), "is_string": False})
    return {"id": id_, "name": name, "arguments": args}


CASES = [
    {"name": "single user", "messages": [{"role": "user", "content": "Hello!"}]},
    {"name": "system max", "effort": "max",
     "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "2+2?"}]},
    {"name": "developer as system",
     "messages": [{"role": "developer", "content": "Answer in French."}, {"role": "user", "content": "Hi"}]},
    {"name": "no_think drops history reasoning", "effort": "no_think",
     "messages": [{"role": "user", "content": "First"},
                  {"role": "assistant", "content": "One", "reasoning": "thinking one"},
                  {"role": "user", "content": "Second"}]},
    {"name": "reasoning kept only after the last user",
     "messages": [{"role": "user", "content": "First"},
                  {"role": "assistant", "content": "One", "reasoning": "old thought"},
                  {"role": "user", "content": "Second"},
                  {"role": "assistant", "content": "Two", "reasoning": "new thought"}]},
    {"name": "tool loop reorders responses", "tools": [WEATHER, SEARCH],
     "messages": [{"role": "user", "content": "Weather in Rome and Oslo?"},
                  {"role": "assistant", "content": "", "reasoning": "Two cities.",
                   "tool_calls": [call("a1", "get_weather", city="Rome", days=2),
                                  call("b2", "get_weather", city="Oslo")]},
                  {"role": "tool", "tool_call_id": "b2", "content": "Oslo: 3"},
                  {"role": "tool", "tool_call_id": "a1", "content": "Rome: 18"}]},
    {"name": "tool responses without ids keep their order", "tools": [WEATHER],
     "messages": [{"role": "user", "content": "Weather?"},
                  {"role": "assistant", "content": "Checking.",
                   "tool_calls": [call("", "get_weather", city="Rome"), call("c3", "get_weather", city="Oslo")]},
                  {"role": "tool", "tool_call_id": "c3", "content": "second"},
                  {"role": "tool", "tool_call_id": "", "content": "first"},
                  {"role": "assistant", "content": "Done.", "reasoning": "kept: tools preserve thinking"},
                  {"role": "user", "content": "Thanks"}]},
    {"name": "tools with a system message and typed arguments", "tools": [SEARCH, DEFERRED, WEATHER],
     "messages": [{"role": "system", "content": "Use tools."},
                  {"role": "user", "content": "Find ünïcode"},
                  {"role": "assistant", "content": "",
                   "tool_calls": [call("s1", "search", q="ünïcode <tags>", limit=5, exact=True,
                                       filters={"lang": ["en", "it"], "safe": None})]},
                  {"role": "tool", "tool_call_id": "s1", "content": "{\"hits\": 0}"}]},
    {"name": "tool choice none hides tools", "tools": [WEATHER], "tool_choice": "none",
     "messages": [{"role": "user", "content": "Just talk."}]},
    {"name": "images in user content",
     "messages": [{"role": "user", "content": "Compare these two.", "images": [0, 7, 18]}]},
    {"name": "image in a tool response", "tools": [SEARCH],
     "messages": [{"role": "user", "content": "Look"},
                  {"role": "assistant", "content": "", "tool_calls": [call("i1", "search", q="cat")]},
                  {"role": "tool", "tool_call_id": "i1", "content": "Here:", "images": [5]}]},
    {"name": "system after the first message",
     "messages": [{"role": "user", "content": "A"}, {"role": "system", "content": "Mid system."},
                  {"role": "user", "content": "B"}]},
    {"name": "assistant prefill stays open",
     "messages": [{"role": "user", "content": "Count"}, {"role": "assistant", "content": "1, 2,"}]},
]


def to_template(case):
    messages = []
    for m in case["messages"]:
        content = m["content"]
        if m.get("images"):
            parts, cursor = [], 0
            for offset in sorted(m["images"]):
                if offset > cursor:
                    parts.append({"type": "text", "text": content[cursor:offset]})
                parts.append(IMAGE)
                cursor = offset
            if cursor < len(content):
                parts.append({"type": "text", "text": content[cursor:]})
            content = parts
        out = {"role": "system" if m["role"] == "developer" else m["role"], "content": content}
        if m.get("reasoning"):
            out["reasoning_content"] = m["reasoning"]
        if m["role"] == "tool":
            out["tool_call_id"] = m.get("tool_call_id", "")
        if m.get("tool_calls"):
            out["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"],
                              "arguments": {a["name"]: a["value"] if a["is_string"] else json.loads(a["value"])
                                            for a in c["arguments"]}}}
                for c in m["tool_calls"]]
        messages.append(out)
    tools = case.get("tools") if case.get("tool_choice", "auto") != "none" else None
    return messages, tools


def main():
    template_path, out_path = sys.argv[1], sys.argv[2]
    source = open(template_path, encoding="utf-8").read()

    def raise_exception(message):
        raise TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(source)
    cases = []
    for case in CASES:
        messages, tools = to_template(case)
        rendered = template.render(messages=messages, tools=tools, add_generation_prompt=True,
                                   reasoning_effort=case.get("effort", "high"))
        cases.append(dict(case, rendered=rendered))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"template_sha256": hashlib.sha256(source.encode()).hexdigest(), "cases": cases}, f,
                  ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
