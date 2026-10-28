"""Find where the m0xw live result (0.9933) entered the transcript; dump user-side text near it."""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "recovered"
LOGS = sorted(pathlib.Path("/mnt/c/Users/Saifuddin/.claude/projects/C--Users-Saifuddin-Documents-Agenthon-2026").glob("*.jsonl"))
n = 0
for log in LOGS:
    with open(log, "r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh):
            if "0.9933" not in line and "0,9933" not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message") or {}
            content = msg.get("content")
            texts = []
            if isinstance(content, str):
                texts.append(("str", content))
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        texts.append(("text", part.get("text", "")))
                    elif part.get("type") == "tool_result":
                        c = part.get("content")
                        texts.append(("result", c if isinstance(c, str) else json.dumps(c)))
                    elif part.get("type") == "tool_use":
                        texts.append(("tool_use", json.dumps(part.get("input"))))
            for kind, t in texts:
                if "0.9933" in t or "0,9933" in t:
                    n += 1
                    i = t.find("0.9933")
                    print("----", log.name[:8], ln, rec.get("timestamp"), msg.get("role"), kind, len(t))
                    if n <= 12:
                        print(t[max(0, i - 700): i + 500].encode("ascii", "replace").decode())
                    if kind in ("str",) and len(t) > 2000:
                        (OUT / f"m0xw_{ln}.txt").write_text(t, encoding="utf-8")
print("hits", n)
