import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib

print("🤖 Starting full machine learning behavioral analysis...\n")
start_time = time.time()

# === Load dataset ===
csv_path = os.path.join(os.path.dirname(__file__), "trained_sales.csv")
df = pd.read_csv(csv_path)

# === Clean and preprocess ===
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Member_number", "itemDescription", "category"])

print(f"✅ Loaded {len(df)} sales records.\n")

# === STEP 1: Feature engineering for user behavior ===
print("🔹 Computing user purchase frequency...")
user_activity = df.groupby("Member_number")["Date"].agg(["min", "max", "count"])
user_activity["days_active"] = (user_activity["max"] - user_activity["min"]).dt.days + 1
user_activity["avg_per_week"] = user_activity["count"] / (user_activity["days_active"] / 7)

# Label frequency categories
user_activity["frequency_label"] = pd.cut(
    user_activity["avg_per_week"],
    bins=[0, 0.5, 1.5, 7, np.inf],
    labels=["Occasional", "Monthly", "Weekly", "Daily"],
)

print("✅ Frequency classification complete.\n")

# === STEP 2: Build an ML model to predict frequency ===
print("🎯 Training RandomForest model to predict buying frequency...")

# Prepare features and labels
X = user_activity[["count", "days_active", "avg_per_week"]]
y = user_activity["frequency_label"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
model_path = os.path.join(os.path.dirname(__file__), "buying_frequency_model.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")
joblib.dump(model, model_path)
joblib.dump(le, encoder_path)
print(f"💾 Model saved to {model_path}")
print(f"💾 Encoder saved to {encoder_path}\n")

# === STEP 3: Best-selling items ===
print("🏆 Finding top-selling items...")
top_items = df["itemDescription"].value_counts().head(10)
print(top_items, "\n")

# === STEP 4: Recommendation model (cosine similarity) ===
print("🧠 Building item similarity matrix (for recommendations)...")
user_item_matrix = pd.crosstab(df["Member_number"], df["itemDescription"])
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Function to recommend similar items
def recommend_items(item_name, n=5):
    if item_name not in item_similarity_df.columns:
        return []
    similar_items = item_similarity_df[item_name].sort_values(ascending=False)[1:n+1]
    return list(similar_items.index)

# Pick a sample item
sample_item = df["itemDescription"].sample(1).iloc[0]
recommendations = recommend_items(sample_item)

# Save recommendations and behavior summary
behavior_summary_path = os.path.join(os.path.dirname(__file__), "behavior_analysis_ml.csv")
user_activity.reset_index().to_csv(behavior_summary_path, index=False)

end_time = time.time()

print("📈 Machine Learning behavioral analysis complete!\n")
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print(f"🛒 Sample item recommendations for '{sample_item}': {recommendations}")
print(f"📊 Analysis saved to {behavior_summary_path}")
print(f"⏱️ Total runtime: {end_time - start_time:.2f} seconds\n")
print("✅ Done!")
