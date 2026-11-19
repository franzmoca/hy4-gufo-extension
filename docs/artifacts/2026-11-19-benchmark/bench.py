# One arm: decode prompts, a text prefill and image prompts, against a fresh server.
import base64, json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
def ask(content, max_tokens):
    body = {"model": "Hy4 Flash", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": "none"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=1800)); dt = time.time() - t
    u = r["usage"]; g = u.get("gufo", {})
    return {"reply": r["choices"][0]["message"].get("content"), "seconds": dt,
            "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
            "decode_tps": u.get("completion_tokens_per_second"), "prefill_tps": u.get("prompt_tokens_per_second"),
            "ttft_ms": g.get("ttft_ms"), "draft_tokens": u.get("draft_tokens"),
            "draft_accepted": u.get("draft_tokens_accepted")}
def image(path, detail, question):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    url = "data:%s;base64,%s" % (mime, base64.b64encode(open(path, "rb").read()).decode())
    return [{"type": "image_url", "image_url": {"url": url, "detail": detail}}, {"type": "text", "text": question}]
res = {}
for name, p in (("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."),
                ("prose", "Explain in three short paragraphs why the sky is blue."),
                ("list", "List the first 20 prime numbers, comma separated, then explain how you found them.")):
    res[name] = ask(p, 400)
res["prefill"] = ask(open(sys.argv[3]).read() + "\n\nSummarize the text above in one sentence.", 1)
res["image_2048"] = ask(image(sys.argv[4], "auto", "Describe this image in two sentences."), 80)
res["image_low"] = ask(image(sys.argv[4], "low", "Describe this image in two sentences."), 80)
for k, v in res.items():
    print(k, {x: v[x] for x in ("prompt_tokens", "completion_tokens", "decode_tps", "prefill_tps", "ttft_ms", "draft_tokens", "draft_accepted")}, flush=True)
json.dump(res, open(out, "w"), indent=1)
