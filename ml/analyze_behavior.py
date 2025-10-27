import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

print("📊 Starting behavioral analysis...\n")

start_time = time.time()

# === Load the trained data ===
csv_path = os.path.join(os.path.dirname(__file__), "trained_sales.csv")
df = pd.read_csv(csv_path)

# --- Clean and preprocess ---
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Member_number", "itemDescription", "category"])

print(f"✅ Loaded {len(df)} sales records.\n")

# === Buying frequency ===
print("🔹 Analyzing purchase frequency...")
user_activity = df.groupby("Member_number")["Date"].agg(["min", "max", "count"])
user_activity["days_active"] = (user_activity["max"] - user_activity["min"]).dt.days + 1
user_activity["avg_per_week"] = user_activity["count"] / (user_activity["days_active"] / 7)
user_activity["frequency_label"] = pd.cut(
    user_activity["avg_per_week"],
    bins=[0, 0.5, 1.5, 7, np.inf],
    labels=["Occasional", "Monthly", "Weekly", "Daily"],
)
print("✅ Frequency classification complete.\n")

# === Best-selling items ===
print("🔹 Finding best-selling items...")
top_items = df["itemDescription"].value_counts().head(10)
print("✅ Top 10 best-selling items:\n")
print(top_items, "\n")

# === Simple recommendation engine ===
print("🔹 Generating user-item matrix...")
user_item_matrix = pd.crosstab(df["Member_number"], df["itemDescription"])
similarity = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_for_user(user_id, n=5):
    if user_id not in similarity_df.index:
        return []
    similar_users = similarity_df[user_id].sort_values(ascending=False).index[1:6]
    similar_users_purchases = df[df["Member_number"].isin(similar_users)]
    user_purchases = set(df[df["Member_number"] == user_id]["itemDescription"])
    recommended_items = (
        similar_users_purchases[~similar_users_purchases["itemDescription"].isin(user_purchases)]
        ["itemDescription"]
        .value_counts()
        .head(n)
        .index.tolist()
    )
    return recommended_items

# Pick a random user for demo
sample_user = df["Member_number"].sample(1).iloc[0]
recommendations = recommend_for_user(sample_user)

# === Save analysis ===
output_path = os.path.join(os.path.dirname(__file__), "behavior_analysis.csv")
summary = user_activity.reset_index()
summary.to_csv(output_path, index=False)

end_time = time.time()

print(f"📈 Behavior analysis saved to {output_path}")
print(f"🎯 Sample recommendations for user {sample_user}: {recommendations}")
print(f"⏱️ Total analysis time: {end_time - start_time:.2f} seconds\n")
print("✅ Behavioral analysis complete!")
