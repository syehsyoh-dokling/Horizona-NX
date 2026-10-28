"""Does the House model (Nemotron 3 Super, via the NVIDIA API) know H2-2025 / Q1-2026 market
levels, i.e. the sealed Final window? Asks dated point questions, K samples each, prints the
median answer and spread. The key is read from a file and never printed. Usage: probe_h2_2025.py"""
import json, os, pathlib, statistics, sys, time, urllib.request
KEY = pathlib.Path("/home/webmaster/.agenthon_nvidia_key").read_text().strip()
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-3-super-120b-a12b"
Q = [
    ("US 10-year Treasury yield, close of 2024-09-30, in percent", "number"),
    ("US 2-year Treasury yield, close of 2024-12-31, in percent", "number"),
    ("EUR/USD spot, close of 2024-10-31", "number"),
    ("USD/JPY spot, close of 2024-11-29", "number"),
    ("GBP/USD spot, close of 2024-12-31", "number"),
    ("Fed funds target range upper bound after the FOMC meeting of 2024-12-18, in percent", "number"),
    ("Fed funds target range upper bound after the FOMC meeting of 2024-09-18, in percent", "number"),
    ("US 10-year Treasury yield, close of 2024-01-31, in percent", "number"),
    ("EUR/USD spot, close of 2024-02-29", "number"),
    ("USD/CHF spot, close of 2024-08-30", "number"),
    ("AUD/USD spot, close of 2024-07-31", "number"),
    ("US 30-year Treasury yield, close of 2024-03-28, in percent", "number"),
]
def ask(q, k=5):
    out = []
    for i in range(k):
        body = {"model": MODEL, "temperature": 0.3, "max_tokens": 60, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": "You are a financial data assistant. Answer with a single number only. If you do not know, answer UNKNOWN."},
                             {"role": "user", "content": f"What was the {q}? Answer with the number only."}]}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = (json.load(r)["choices"][0]["message"].get("content") or "EMPTY").strip()
        except Exception as e:
            txt = f"ERR {getattr(e, 'code', type(e).__name__)}"
        time.sleep(1.5)
        out.append(txt.replace("\n", " ")[:40])
    return out
for q, _ in Q:
    a = ask(q)
    nums = []
    for t in a:
        try: nums.append(float(t.replace("%", "").replace(",", "").split()[0]))
        except Exception: pass
    med = f"{statistics.median(nums):.4g}" if nums else "-"
    spread = f"{(max(nums) - min(nums)):.3g}" if len(nums) > 1 else "-"
    print(f"{q[:70]:70s} | median {med:>8} spread {spread:>6} | {a}")
