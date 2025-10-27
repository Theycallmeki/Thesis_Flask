import os
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib

print("🤖 Starting full machine learning behavioral analysis...\n")
start_time = time.time()

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "raw_sales.csv")
TRAINED_PATH = os.path.join(BASE_DIR, "trained_sales.csv")
MODEL_PATH = os.path.join(BASE_DIR, "buying_frequency_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
BEHAVIOR_SUMMARY_PATH = os.path.join(BASE_DIR, "behavior_analysis_ml.csv")

# === Step 0: Load raw dataset ===
df = pd.read_csv(RAW_PATH)

# === Step 1: Categorize items ===
categories = {
    'Fruits': ['fruit', 'citrus', 'apple', 'banana', 'mango', 'pip fruit', 'tropical'],
    'Vegetables': ['vegetable', 'root', 'spinach', 'tomato', 'broccoli'],
    'Meat': ['beef', 'chicken', 'pork', 'sausage', 'ham', 'frankfurter'],
    'Seafood': ['fish', 'shrimp', 'tuna', 'salmon'],
    'Dairy': ['milk', 'yogurt', 'butter', 'cheese', 'cream', 'butter milk'],
    'Beverages': ['soda', 'juice', 'coffee', 'tea', 'water'],
    'Snacks': ['snack', 'chips', 'crisps', 'nuts', 'chocolate', 'specialty bar'],
    'Bakery': ['bread', 'pastry', 'cake', 'buns', 'rolls'],
    'Frozen': ['frozen', 'ice cream'],
    'Canned Goods': ['canned', 'tin', 'soup'],
    'Condiments': ['ketchup', 'mustard', 'mayo', 'sauce'],
    'Dry Goods': ['flour', 'sugar', 'salt'],
    'Grains & Pasta': ['rice', 'pasta', 'noodles', 'spaghetti'],
    'Spices & Seasonings': ['pepper', 'herbs', 'spice'],
    'Breakfast & Cereal': ['cereal', 'oats', 'granola'],
    'Personal Care': ['soap', 'shampoo', 'toothpaste'],
    'Household': ['detergent', 'tissue', 'cleaner'],
    'Baby Products': ['diaper', 'baby'],
    'Pet Supplies': ['dog', 'cat', 'pet'],
    'Health & Wellness': ['vitamin', 'supplement', 'medicine'],
    'Cleaning Supplies': ['bleach', 'cleaner', 'disinfectant']
}

# Detect item column
def get_item_column(df):
    for col in ['item', 'itemDescription', 'Item', 'ItemDescription']:
        if col in df.columns:
            return col
    raise KeyError("No valid item column found (expected 'item' or 'itemDescription')")

item_col = get_item_column(df)

# Match items to categories
def match_category(item):
    if pd.isna(item):
        return None
    item_lower = str(item).lower()
    for category, keywords in categories.items():
        if any(word in item_lower for word in keywords):
            return category
    return 'Uncategorized'

df['category'] = df[item_col].apply(match_category)

# Save cleaned dataset
df.to_csv(TRAINED_PATH, index=False)
print(f"✅ Raw data categorized and saved to {TRAINED_PATH}\n")

# === Step 2: Compute user behavior ===
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Member_number", item_col, "category"])
print(f"✅ Loaded {len(df)} sales records.\n")

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

# === Step 3: Train ML model ===
print("🎯 Training RandomForest model to predict buying frequency...")
X = user_activity[["count", "days_active", "avg_per_week"]]
y = user_activity["frequency_label"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and encoder
joblib.dump(model, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
print(f"💾 Encoder saved to {ENCODER_PATH}\n")

# === Step 4: Top-selling items ===
print("🏆 Finding top-selling items...")
top_items = df[item_col].value_counts().head(10)
print(top_items, "\n")

# === Step 5: Item recommendations ===
print("🧠 Building item similarity matrix (for recommendations)...")
user_item_matrix = pd.crosstab(df["Member_number"], df[item_col])
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def recommend_items(item_name, n=5):
    if item_name not in item_similarity_df.columns:
        return []
    similar_items = item_similarity_df[item_name].sort_values(ascending=False)[1:n+1]
    return list(similar_items.index)

sample_item = df[item_col].sample(1).iloc[0]
recommendations = recommend_items(sample_item)

# Save behavior summary
user_activity.reset_index().to_csv(BEHAVIOR_SUMMARY_PATH, index=False)

end_time = time.time()
print("📈 Machine Learning behavioral analysis complete!\n")
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print(f"🛒 Sample recommendations for '{sample_item}': {recommendations}")
print(f"📊 Analysis saved to {BEHAVIOR_SUMMARY_PATH}")
print(f"⏱️ Total runtime: {end_time - start_time:.2f} seconds\n")
print("✅ Done!")
