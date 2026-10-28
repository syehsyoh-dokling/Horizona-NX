"""What shape is the exemplar panel actually in?"""

import pandas as pd

P = ("/mnt/c/Users/Saifuddin/Documents/track2-forecasting-public/"
     "units/t2-EXAMPLE-ust-curve-1m/rates_daily.parquet")

frame = pd.read_parquet(P)
print("columns:", list(frame.columns))
print("dtypes:")
print(frame.dtypes.to_string())
print("\nrows:", len(frame))
print(frame.head(5).to_string(index=False))
print("\ntail:")
print(frame.tail(3).to_string(index=False))
for col in frame.columns:
    if frame[col].dtype == object or str(frame[col].dtype).startswith("str"):
        print(f"\nunique in {col!r}:", sorted(frame[col].unique())[:12])
