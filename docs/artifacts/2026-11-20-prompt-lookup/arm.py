"""One arm: the file edit (greedy, seeds 1-2) and, unless a fourth argument
is given, the three standard prompts greedy and sampled (seed 1).
usage: arm.py PORT OUT.json SOURCE_FILE [edit]
The runs used the first 95 lines of runtime/hy4_dfly.hpp at 830596c."""
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
src = open(sys.argv[3]).read()
EDIT = ("Here is a C++ header:\n\n```cpp\n" + src + "\n```\n\nRewrite the whole file with the class `Drafter` renamed to "
        "`BlockDrafter` everywhere. Output only the complete code, nothing else.")
P = {"code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
     "prose": "Explain in three short paragraphs why the sky is blue.",
     "list": "List the first 20 prime numbers, comma separated, then explain how you found them."}
def ask(content, max_tokens, **kw):
    body = {"model": "Hy4 Flash", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "reasoning_effort": "none", **kw}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    u = r["usage"]
    return {"reply": r["choices"][0]["message"].get("content"), "completion_tokens": u["completion_tokens"],
            "decode_tps": u.get("completion_tokens_per_second"), "draft_tokens": u.get("draft_tokens"),
            "draft_accepted": u.get("draft_tokens_accepted"), "rounds": u.get("gufo", {}).get("draft_rounds")}
res = {}
res["edit/greedy"] = ask(EDIT, 1500, temperature=0)
for seed in (1, 2):
    res[f"edit/sampled/{seed}"] = ask(EDIT, 1500, seed=seed)
if len(sys.argv) < 5:
    for name, p in P.items():
        res[f"greedy/{name}"] = ask(p, 400, temperature=0)
        res[f"sampled/{name}/1"] = ask(p, 400, seed=1)
for k, v in res.items():
    print(k, {x: v[x] for x in v if x != "reply"}, flush=True)
json.dump(res, open(out, "w"), indent=1)
