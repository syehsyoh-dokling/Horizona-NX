"""Print a unit's as-of date, read from the card rather than grepped.

`grep -o` over card.toml returns every date in the file - the as-of, the panel
end, the created date, the target date - and feeding all of them to pandas gives
"Unknown datetime string format". The card knows which is which, but not always
under the same key: the exemplar has no `asof` field at all, it has a panel
`end_date` of 2024-06-28 and a separate `target_dates` of 2024-07-31. Taking
the wrong one silently forecasts from the future.

So: walk the whole card, collect only keys that mean "data ends here", and
ignore anything that means "predict this far ahead".
"""

import pathlib
import sys
import tomllib

ASOF_KEYS = {"asof", "as_of", "end_date", "data_end", "cutoff", "data_cutoff"}
#: Never an as-of: these describe the thing being predicted.
FORBIDDEN = {"target_date", "target_dates", "created_date",
             "public_release_date", "start_date"}


def walk(node, found: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = key.lower()
            if lowered in FORBIDDEN:
                continue
            if lowered in ASOF_KEYS and isinstance(value, str) and len(value) >= 10:
                found.append(value[:10])
            else:
                walk(value, found)
    elif isinstance(node, list):
        for item in node:
            walk(item, found)


card = tomllib.loads(
    (pathlib.Path(sys.argv[1]) / "card.toml").read_text(encoding="utf-8"))

dates: list[str] = []
walk(card, dates)
if not dates:
    raise SystemExit("card declares no as-of / end_date")

# The latest date at which data is still legitimately available.
print(max(dates))
