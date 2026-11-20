"""One arm: the three standard prompts, greedy and seeds 1-3 (default sampling).
usage: arm.py PORT OUT.json"""
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
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
for name, p in P.items():
    res[f"greedy/{name}"] = ask(p, 400, temperature=0)
    for seed in (1, 2, 3):
        res[f"sampled/{name}/{seed}"] = ask(p, 400, seed=seed)
for k, v in res.items():
    print(k, {x: v[x] for x in v if x != "reply"}, flush=True)
json.dump(res, open(out, "w"), indent=1)
