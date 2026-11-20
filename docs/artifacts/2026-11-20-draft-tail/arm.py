import json, sys, urllib.request
port, out, doc = sys.argv[1], sys.argv[2], sys.argv[3]
P = {"code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
     "prose": "Explain in three short paragraphs why the sky is blue.",
     "list": "List the first 20 prime numbers, comma separated, then explain how you found them."}
def ask(content, max_tokens, **kw):
    body = {"model": "Hy4 Flash", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "reasoning_effort": "none", **kw}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    u = r["usage"]
    return {"reply": r["choices"][0]["message"].get("content"), "completion_tokens": u["completion_tokens"],
            "decode_tps": u.get("completion_tokens_per_second"), "draft_tokens": u.get("draft_tokens"),
            "draft_accepted": u.get("draft_tokens_accepted"), "rounds": u.get("gufo", {}).get("draft_rounds")}
res = {}
for name, p in P.items():
    res[f"greedy/{name}"] = ask(p, 400, temperature=0)
res["greedy/long"] = ask(open(doc).read()[:14000] + "\n\nSummarize the text above in five bullet points.", 300, temperature=0)
for name, p in P.items():
    for seed in (1, 2, 3, 4):
        res[f"sampled/{name}/{seed}"] = ask(p, 400, seed=seed)
for k, v in res.items():
    print(k, {x: v[x] for x in v if x != "reply"}, flush=True)
json.dump(res, open(out, "w"), indent=1)
