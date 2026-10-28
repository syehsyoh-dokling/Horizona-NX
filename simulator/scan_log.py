"""Scan the session transcripts for the lost simulator: Write calls of t2_*.py and pasted per-card scores.
Prints only locations and short, key-free snippets."""
import json
import pathlib
import re
import sys

OUT = pathlib.Path(__file__).resolve().parent / "recovered"
OUT.mkdir(exist_ok=True)
LOGS = sorted(pathlib.Path("/mnt/c/Users/Saifuddin/.claude/projects/C--Users-Saifuddin-Documents-Agenthon-2026").glob("*.jsonl"))
pat_card = re.compile(r"t2-F4-vaccine-timeline-2020")
n_write = 0
for log in LOGS:
    with open(log, "r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh):
            if "t2_" not in line and not pat_card.search(line):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, list):
                if isinstance(content, str) and pat_card.search(content):
                    name = OUT / f"user_{log.stem[:8]}_{ln}.txt"
                    name.write_text(content, encoding="utf-8")
                    print("USERTEXT", log.name[:8], ln, rec.get("timestamp"), len(content))
                continue
            for part in content:
                t = part.get("type")
                if t == "tool_use" and part.get("name") in ("Write", "Edit"):
                    inp = part.get("input") or {}
                    fp = str(inp.get("file_path", ""))
                    base = fp.replace("\\", "/").split("/")[-1]
                    if base.startswith("t2_") or "truth" in base or "t2sim" in fp:
                        n_write += 1
                        if part["name"] == "Write":
                            name = OUT / f"{ln:07d}_{base}"
                            name.write_text(inp.get("content", ""), encoding="utf-8")
                            print("WRITE", log.name[:8], ln, rec.get("timestamp"), fp[-60:], len(inp.get("content", "")))
                        else:
                            name = OUT / f"{ln:07d}_EDIT_{base}.json"
                            name.write_text(json.dumps(inp, indent=1), encoding="utf-8")
                            print("EDIT ", log.name[:8], ln, rec.get("timestamp"), fp[-60:])
                elif t == "text" and pat_card.search(part.get("text", "")):
                    name = OUT / f"text_{log.stem[:8]}_{ln}.txt"
                    name.write_text(part["text"], encoding="utf-8")
                    print("TEXT ", log.name[:8], ln, rec.get("timestamp"), msg.get("role"), len(part["text"]))
                elif t == "tool_result":
                    c = part.get("content")
                    s = c if isinstance(c, str) else json.dumps(c)
                    if pat_card.search(s) and len(s) < 400000:
                        name = OUT / f"result_{log.stem[:8]}_{ln}.txt"
                        name.write_text(s, encoding="utf-8")
                        print("RESULT", log.name[:8], ln, rec.get("timestamp"), len(s))
print("writes", n_write)
