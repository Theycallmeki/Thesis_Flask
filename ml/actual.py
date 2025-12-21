import pandas as pd

# Load your file
df = pd.read_csv("tester2.csv")

# Clean columns
df.columns = [c.strip() for c in df.columns]
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["category"] = df["category"].astype(str).str.strip()

# Drop rows with invalid dates
df = df.dropna(subset=["Date"]).reset_index(drop=True)

# Latest date in dataset (so windows are relative to your data)
latest_date = df["Date"].max()
print("Latest date in dataset:", latest_date.date())

def best_selling(window_days: int):
    start_date = latest_date - pd.Timedelta(days=window_days)
    window_df = df[df["Date"] > start_date]

    counts = (
        window_df["category"]
        .value_counts()
        .reset_index()
    )
    counts.columns = ["category", "transactions"]

    best = counts.iloc[0] if not counts.empty else None
    return start_date, counts, best

# Past 7 days
start_7, counts_7, best_7 = best_selling(7)
print(f"\n=== BEST SELLING PAST 7 DAYS ({start_7.date()} to {latest_date.date()}) ===")
if best_7 is not None:
    print("Best-selling category:", best_7["category"])
    print(counts_7)
else:
    print("No transactions in this window.")

# Past 30 days
start_30, counts_30, best_30 = best_selling(30)
print(f"\n=== BEST SELLING PAST 30 DAYS ({start_30.date()} to {latest_date.date()}) ===")
if best_30 is not None:
    print("Best-selling category:", best_30["category"])
    print(counts_30)
else:
    print("No transactions in this window.")
