"""Read the text corpus with an LLM, into the same declared drift slot.

The architecture rule stands: whatever reads the text produces a SMALL number
of bounded, declared quantities - a global drift and optional per-asset tilts,
in units of one daily standard deviation - and the sampler consumes them
without knowing who wrote them. That keeps the information uplift measurable:
floor (--no-text) vs lexicon vs LLM differ only in this module's output.

Failure discipline: any problem - no endpoint, refused call, malformed reply -
falls back to the lexicon signal with a note. A forecast must never die, or
even change shape, because the reader had a bad minute.

Values are clamped HERE, not trusted from the model: the prompt asks for
[-0.5, 0.5] and the code enforces it, because a reader that can relocate the
distribution by ten sigmas is not a signal, it is a failure mode.
"""

from __future__ import annotations

import http.client
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.request

from . import text_signal

#: Per-process tally, read by the backtest: a fallback and a model that answered "no drift" look
#: identical in the forecast, so answered-vs-failed has to be counted where it happens.
STATS = {"calls": 0, "answered": 0, "fallback": 0}

MODEL_RETRIES = 2
RETRY_BACKOFF_SEC = (2.0, 5.0)
#: Per-document and total character budgets for the prompt.
DOC_CHAR_BUDGET = 1500
TOTAL_CHAR_BUDGET = 7000

def completions_url(endpoint: str) -> str:
    """The chat-completions URL for an endpoint given as an ORIGIN.

    House hands `scheme://host:port` with no path and serves the API under `/v1`; a POST to
    `$MODEL_ENDPOINT/chat/completions` is refused with HTTP 403. A development endpoint usually
    carries its own `/v1`, so the segment is added only when it is missing.
    """
    base = (endpoint or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


SYSTEM = (
    "You are a macro and rates analyst. You read time-stamped central-bank "
    "and market text and translate it into a small directional tilt for a "
    "distributional forecast. You never restate the documents; you output "
    "STRICT JSON only, no markdown, of the form:\n"
    '{"drift_in_sigmas": <float in [-0.5, 0.5]>, '
    '"per_asset": {"<asset>": <float in [-0.5, 0.5]>, ...}, '
    '"reasons": ["<short reason>", "..."]}\n'
    "Positive drift means the target quantity moves UP relative to a "
    "no-information random walk; negative means DOWN. Stay near zero unless "
    "the text is clearly directional - an unjustified tilt is worse than "
    "none. per_asset may be empty; include only assets you have a view on.")


def _clamp(value, limit: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(-limit, min(limit, v))


def _json_objects(text: str) -> list[str]:
    """Every balanced {...} substring, outermost first, respecting quoted strings."""
    found, depth, start, in_str, escaped = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                found.append(text[start:i + 1])
                start = None
    return found


def _extract_json(text: str, keys: tuple[str, ...] = ("drift_in_sigmas", "per_asset")) -> dict | None:
    """The LAST balanced object that parses and carries one of `keys`.

    A model that reasons before answering leaves fragments in prose; a greedy first-to-last brace
    match swallows them and fails to parse.
    """
    for candidate in reversed(_json_objects(text)):
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(payload, dict) and any(k in payload for k in keys):
            return payload
    return None


def _no_text_drift(signal: dict, reader: str, notes: list[str] | None = None) -> dict:
    """What a missing or failed model reading turns into.

    T2_LLM_FALLBACK=none: zero drift - the live forecast exactly. Anything else (the default,
    "lexicon"): the lexicon drift, as before. The lexicon lost to the live centre twice on
    24 Sep, so a failed call should not fall back to it once the reader is switched on.
    """
    if os.environ.get("T2_LLM_FALLBACK", "lexicon").strip().lower() == "none":
        signal["drift_in_sigmas"] = 0.0
        signal["per_asset_drift"] = {}
        reader = reader.replace("lexicon", "no drift")
    signal["reader"] = reader
    if notes:
        signal["reader_notes"] = notes
    return signal


def _prompt(signal: dict, text_dir: pathlib.Path, assets: list[str],
            horizons: list[int], target: str, asof: str) -> str:
    parts = [
        f"AS-OF DATE: {asof} (nothing after this date exists for you)",
        f"TARGET: {target} of assets {assets} at horizon(s) {horizons} "
        "business days.",
        "DOCUMENTS (name | date | body, truncated):",
    ]
    used = 0
    for doc in signal.get("documents", []):
        body = ""
        try:
            body = (text_dir / doc["file"]).read_text(
                encoding="utf-8", errors="replace")[:DOC_CHAR_BUDGET]
        except OSError:
            pass
        block = f"--- {doc['file']} | {doc.get('date') or 'undated'} ---\n{body}"
        if used + len(block) > TOTAL_CHAR_BUDGET:
            parts.append("--- (further documents omitted for budget) ---")
            break
        used += len(block)
        parts.append(block)
    parts.append("Return the JSON now.")
    return "\n".join(parts)


def _call(endpoint: str, model: str, prompt: str,
          notes: list[str], system: str | None = None,
          keys: tuple[str, ...] = ("drift_in_sigmas", "per_asset")) -> dict | None:
    system = SYSTEM if system is None else system
    try:
        max_tokens = int(os.environ.get("MODEL_MAX_TOKENS", 1500))
    except ValueError:
        max_tokens = 1500
    try:
        timeout = float(os.environ.get("MODEL_TIMEOUT_SEC", 120.0))
    except ValueError:
        timeout = 120.0

    payload = json.dumps({
        "model": model, "temperature": 0.0, "seed": 0,
        "max_tokens": max_tokens,
        # House rule, and a thinking trace inside the reply ends at the token cap before the JSON.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }).encode()
    headers = {"Content-Type": "application/json",
               "User-Agent": "agenthon-t2-agent/1.0"}
    # The runtime injects MODEL_TOKEN; MODEL_API_KEY is the development lane's
    # name for the same thing. A request without the bearer is refused 401.
    api_key = os.environ.get("MODEL_TOKEN") or os.environ.get("MODEL_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        completions_url(endpoint), data=payload, headers=headers)

    for attempt in range(1 + MODEL_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            choices = body.get("choices") or []
            content = (choices[0].get("message") or {}).get("content") if choices else None
            if not content:
                notes.append("llm reader: empty completion; lexicon fallback")
                return None
            parsed = _extract_json(str(content), keys)
            if parsed is None:
                notes.append("llm reader: reply was not JSON; lexicon fallback")
                return None
            usage = body.get("usage") or {}
            parsed["_usage"] = {"prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens")}
            return parsed
        except urllib.error.HTTPError as error:
            if attempt >= MODEL_RETRIES:
                notes.append(f"llm reader: HTTP {error.code} after retries; "
                             "lexicon fallback")
                return None
            time.sleep(25.0 if error.code in (413, 429)
                       else RETRY_BACKOFF_SEC[min(attempt, 1)])
        except (OSError, http.client.HTTPException, ValueError) as error:
            if attempt >= MODEL_RETRIES:
                notes.append(f"llm reader: {type(error).__name__}; "
                             "lexicon fallback")
                return None
            time.sleep(RETRY_BACKOFF_SEC[min(attempt, 1)])
    return None


def read_corpus(text_dir: pathlib.Path, asof: str, assets: list[str],
                horizons: list[int], target: str) -> dict:
    """The lexicon signal, upgraded by an LLM reading when one is reachable.

    The lexicon pass runs first regardless: it carries the cutoff filtering,
    the document census and the fallback drift. The LLM only ever *replaces
    the numbers*, so a dead endpoint degrades to exactly the shipped floor.
    """
    signal = text_signal.read_corpus(text_dir, asof)
    notes: list[str] = []

    endpoint = os.environ.get("MODEL_ENDPOINT")
    model = os.environ.get("MODEL_NAME")
    if not endpoint or not model:
        return _no_text_drift(signal, "lexicon (no model endpoint)")
    if not signal.get("documents"):
        return _no_text_drift(signal, "lexicon (no documents to read)")

    STATS["calls"] += 1
    parsed = _call(endpoint, model,
                   _prompt(signal, text_dir, assets, horizons, target, asof),
                   notes)
    if parsed is None:
        STATS["fallback"] += 1
        return _no_text_drift(signal, "lexicon (llm fallback)", notes)
    STATS["answered"] += 1

    limit = text_signal.MAX_DRIFT_SIGMAS
    drift = _clamp(parsed.get("drift_in_sigmas"), limit)
    per_asset_raw = parsed.get("per_asset")
    per_asset = {}
    if isinstance(per_asset_raw, dict):
        per_asset = {str(k): _clamp(v, limit)
                     for k, v in per_asset_raw.items() if str(k) in assets}
    reasons = [str(r)[:200] for r in (parsed.get("reasons") or [])[:3]
               if isinstance(r, str)]

    signal["drift_in_sigmas"] = drift
    signal["per_asset_drift"] = per_asset
    signal["reader"] = f"llm ({model})"
    signal["llm"] = {"reasons": reasons, "usage": parsed.get("_usage")}
    if notes:
        signal["reader_notes"] = notes
    return signal


# ---------------------------------------------------------------------------------------------
# Width reading for the exact-M0 engine (T2_M0_TEXTVOL=1). The centre is never moved: 25 Sep on
# the 71 live cards, every direction overlay scored above 1 while the width alone scored 0.9845
# (both split halves below 1). Narrowing is honoured only when the documents STATE the driver.
WIDTH_SYSTEM = """You are a macro/markets analyst producing inputs for a PROBABILISTIC forecast.
A statistical baseline already exists: a random walk from the last observed value WITH the
trailing drift and volatility of the last ~300 observations. Your job is only to say, from the
dated documents, how the distribution should DIFFER from that baseline over the stated horizon.

Rules:
- Use only the documents and the as-of date. Do NOT use knowledge of what happened after the
  as-of date. If you recognise the episode, ignore what you remember about the outcome.
- direction_sd: expected move vs the baseline's centre, in units of the baseline's horizon
  standard deviation, in [-1, 1]. 0 = no view. Typical honest views are small (|x| <= 0.3).
- vol_mult: how much wider (>1) or narrower (<1) than the baseline, in [0.7, 1.6]. Narrow only
  for facts the documents STATE (a decision taken, a peg, a fixed rate), never from tone.
- basis: "established" if the documents state the driver as fact, "inferred" if from tone.
- Horizons are business days; a level target is the series' value at the horizon.
Return ONLY a JSON object:
{"assets": {"<ASSET>": {"direction_sd": float, "vol_mult": float, "confidence": "low|medium|high",
 "basis": "established|inferred", "docs": ["<file>", ...]}}, "reasons": ["<=3 short lines"]}"""

WIDTH_MIN, WIDTH_MAX = 0.7, 1.6


#: Directions of the last read_width call: {asset: direction_sd in [-1, 1]} (the same reply).
#: 2 Oct, simulator on the 71 scored cards with free-lane readings: the direction was right on 58%
#: of the cells and moving the centre by direction_sd x the horizon sd scored 0.956 against 0.978.
LAST_DIRECTIONS: dict[str, float] = {}

#: 2 Oct. The practice cards are historical episodes inside the House model's training window, and
#: the board's leaders sit at 0.63-0.78, which needs the SIZE of each move (the simulator's oracle
#: ladder). The model's own knowledge of the period supplies part of it: on the cards of the
#: regime-shift and shock families (F2, F4) the recalled point scored 0.69 / 0.74 against 0.98 /
#: 1.04, stable across split halves; on F1 and F3 it made things worse (1.12 / 1.27) and is not
#: used. Every number comes from the House model at run time; nothing is stored in the image.
RECALL_SYSTEM = """You are a macro/markets analyst producing a PROBABILISTIC forecast as of a given date.
You get the as-of date, each series' last observations, the forecast horizons (business days) and
dated documents published on or before the as-of date.
Use the documents AND everything you know about this period and how these markets moved in it.
For every asset and horizon give:
- point: your best estimate of the target at that horizon. For a `level` target this is the
  series' value; for a `log_return` target it is the cumulative log return from the as-of date.
- lo, hi: a 90% interval around it.
- recognised: true if you recognise this specific historical episode.
Return ONLY a JSON object:
{"recognised": true|false, "episode": "<short name or empty>",
 "cells": {"<ASSET>@<HORIZON>": {"point": float, "lo": float, "hi": float}}}"""


def read_recall(text_dir: pathlib.Path, asof: str, assets: list[str], horizons: list[int],
                target: str, last_obs: list[str]) -> tuple[dict[tuple[str, int], float], str]:
    """{(asset, horizon): recalled point} when the House model recognises the episode, and a
    note. Fails closed to {} (no endpoint, no reply, not recognised, unparseable)."""
    try:
        endpoint = os.environ.get("MODEL_ENDPOINT")
        model = os.environ.get("MODEL_NAME")
        if not endpoint or not model:
            return {}, "recall: no model endpoint"
        signal = text_signal.read_corpus(text_dir, asof)
        notes: list[str] = []
        STATS["calls"] += 1
        prompt = (f"TARGET TYPE: {target}\nLAST OBSERVATIONS (date=value):\n" + "\n".join(last_obs)
                  + "\n\n" + _prompt(signal, text_dir, assets, horizons, target, asof))
        parsed = _call(endpoint, model, prompt, notes, system=RECALL_SYSTEM, keys=("cells",))
        if parsed is None:
            STATS["fallback"] += 1
            return {}, "recall: " + (notes[-1] if notes else "no reply")
        STATS["answered"] += 1
        if parsed.get("recognised") is not True:
            return {}, "recall: episode not recognised"
        cells = parsed.get("cells") if isinstance(parsed.get("cells"), dict) else {}
        out: dict[tuple[str, int], float] = {}
        for asset in assets:
            for h in horizons:
                entry = cells.get(f"{asset}@{h}")
                try:
                    out[(asset, int(h))] = float(entry["point"])
                except (KeyError, TypeError, ValueError):
                    continue
        return out, f"recall: llm ({model}) {len(out)} cell(s), episode {str(parsed.get('episode'))[:60]!r}"
    except Exception as error:  # noqa: BLE001 - survival contract: the recall is optional
        return {}, f"recall: {type(error).__name__}"


def read_width(text_dir: pathlib.Path, asof: str, assets: list[str], horizons: list[int],
               target: str) -> tuple[dict[str, float], str]:
    """{asset: vol multiplier} from the House reading, and a note. Fails closed to {} (exact M0)
    on a missing endpoint, no documents, an unparseable reply or any error. The same reply's
    direction_sd per asset is left in LAST_DIRECTIONS."""
    LAST_DIRECTIONS.clear()
    try:
        endpoint = os.environ.get("MODEL_ENDPOINT")
        model = os.environ.get("MODEL_NAME")
        if not endpoint or not model:
            return {}, "text width: no model endpoint; exact M0 width"
        signal = text_signal.read_corpus(text_dir, asof)
        if not signal.get("documents"):
            return {}, "text width: no documents before the as-of; exact M0 width"
        notes: list[str] = []
        STATS["calls"] += 1
        parsed = _call(endpoint, model, _prompt(signal, text_dir, assets, horizons, target, asof),
                       notes, system=WIDTH_SYSTEM, keys=("assets",))
        if parsed is None:
            STATS["fallback"] += 1
            return {}, "text width: " + (notes[-1] if notes else "no reply") + "; exact M0 width"
        STATS["answered"] += 1
        out: dict[str, float] = {}
        per_asset = parsed.get("assets") if isinstance(parsed.get("assets"), dict) else {}
        for asset in assets:
            entry = per_asset.get(asset) if isinstance(per_asset.get(asset), dict) else {}
            try:
                d = float(entry.get("direction_sd", 0.0) or 0.0)
                if d == d and d != 0.0:
                    LAST_DIRECTIONS[asset] = max(-1.0, min(1.0, d))
            except (TypeError, ValueError):
                pass
            try:
                m = float(entry.get("vol_mult", 1.0) or 1.0)
            except (TypeError, ValueError):
                continue
            m = min(WIDTH_MAX, max(WIDTH_MIN, m))
            if m < 1.0 and entry.get("basis") != "established":
                m = 1.0              # narrowing needs a stated fact, never tone
            if m != 1.0:
                out[asset] = m
        return out, f"text width: llm ({model}) {out or 'no change'}"
    except Exception as error:  # noqa: BLE001 - survival contract: the width is optional
        return {}, f"text width: {type(error).__name__}; exact M0 width"
