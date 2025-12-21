# ml/user_behavior.py
# USER "OFTEN BUYS" BEHAVIOR ANALYSIS (NO ML)

import pandas as pd
import os


def run_user_behavior():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "tester.csv")

    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    df = df.sort_values(["Member_number", "Date"])
    results = []

    for user, group in df.groupby("Member_number"):
        counts = group["category"].value_counts()
        if counts.empty:
            continue

        max_count = counts.max()
        candidates = counts[counts == max_count].index.tolist()

        top_cat = candidates[0]
        if len(candidates) > 1:
            for c in group.sort_values("Date", ascending=False)["category"]:
                if c in candidates:
                    top_cat = c
                    break

        total = int(counts.sum())
        support = int(counts[top_cat])
        confidence = support / total

        results.append({
            "user": user,
            "suggested_category": top_cat,
            "confidence": confidence,
            "support": support,
            "total": total,
            "top_3": counts.head(3).index.tolist()
        })

    results.sort(key=lambda x: (x["confidence"], x["support"]), reverse=True)
    return results
