"""Per-card data cutoff (as-of) and family, sorted by date: which cards lie after the House model's
knowledge cutoffs (pre-training June 2025, post-training ~Feb 2026)? Usage: card_dates.py"""
import t2_core as C, t2_sim as T
import pandas as pd
truth = C.load_truth(); names = T.card_set(truth, "all")
rows = []
for n in names:
    card = C.card_of(n)
    rows.append((pd.Timestamp(card["provenance"]["data_cutoff"]), C.family(n), n))
rows.sort()
for d, f, n in rows:
    print(d.date(), f, n)
cut1, cut2 = pd.Timestamp("2025-06-30"), pd.Timestamp("2026-02-01")
print("cards after 2025-06-30:", sum(1 for d, f, n in rows if d > cut1), "| after 2026-02-01:", sum(1 for d, f, n in rows if d > cut2), "| total", len(rows))
