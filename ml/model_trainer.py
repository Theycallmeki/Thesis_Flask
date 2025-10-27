import os
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
from tqdm import tqdm

# === Paths ===
ml_dir = os.path.dirname(__file__)
csv_path = os.path.join(ml_dir, "trained_sales.csv")

print("🔄 Loading data...")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df.rename(columns={"Category": "category", "Categories": "category"}, inplace=True)

required_cols = {"itemDescription", "category"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"❌ Missing columns! Found: {df.columns.tolist()}")

print(f"✅ Data loaded successfully: {len(df)} records found.\n")

# === Label encoding ===
print("🔢 Encoding categories...")
le_cat = LabelEncoder()
df["category_encoded"] = le_cat.fit_transform(df["category"])

# === Training setup ===
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

X = df["itemDescription"]
y = df["category_encoded"]

print("\n🚀 Starting model training...\n")

# === Progress bar for actual training ===
start_time = time.time()
with tqdm(total=100, desc="Model Training Progress", ncols=90, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

    # Step 1: Fit vectorizer
    pbar.set_description("🔹 Step 1/3: Vectorizing text")
    pipeline.named_steps["vectorizer"].fit(X)
    pbar.update(30)

    # Step 2: Transform text
    pbar.set_description("🔹 Step 2/3: Transforming data")
    X_vec = pipeline.named_steps["vectorizer"].transform(X)
    pbar.update(30)

    # Step 3: Train classifier
    pbar.set_description("🔹 Step 3/3: Training classifier")
    pipeline.named_steps["classifier"].fit(X_vec, y)
    pbar.update(40)

end_time = time.time()
training_time = end_time - start_time

print(f"\n✅ Model training complete in {training_time:.2f} seconds!\n")

# === Save model + vectorizer + predictions ===
print("💾 Saving artifacts...")
model_path = os.path.join(ml_dir, "category_model.pkl")
vectorizer_path = os.path.join(ml_dir, "vectorizer.pkl")
predictions_path = os.path.join(ml_dir, "model_predictions.csv")

joblib.dump(pipeline, model_path)
joblib.dump(pipeline.named_steps["vectorizer"], vectorizer_path)

# Save predictions
df["predicted_category"] = le_cat.inverse_transform(pipeline.predict(X))
df.to_csv(predictions_path, index=False)

print(f"✅ Model saved to {model_path}")
print(f"✅ Vectorizer saved to {vectorizer_path}")
print(f"✅ Predictions saved to {predictions_path}")
print(f"⏱️ Total training time: {training_time:.2f} seconds")
print("🎯 Training process finished successfully!")
