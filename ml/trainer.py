import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("🔥 GPU detected:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("⚠️ No GPU detected — using CPU")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "raw_sales.csv")

df = pd.read_csv(RAW_PATH)

def get_item_column(df):
    for col in ["item", "itemDescription", "ItemDescription", "Item"]:
        if col in df.columns:
            return col
    raise KeyError("❌ No item description column found.")

item_col = get_item_column(df)
print("✅ Using item column:", item_col)

CATEGORIES = [
    'Fruits', 'Vegetables', 'Meat', 'Seafood', 'Dairy', 'Beverages',
    'Snacks', 'Bakery', 'Frozen', 'Canned Goods', 'Condiments',
    'Dry Goods', 'Grains & Pasta', 'Spices & Seasonings',
    'Breakfast & Cereal', 'Personal Care', 'Household',
    'Baby Products', 'Pet Supplies', 'Health & Wellness',
    'Cleaning Supplies'
]

print("⏳ Loading 22MB embedding model... (MiniLM-L6-v2)")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = model.to(DEVICE)
print("✅ Model loaded!")

print("⏳ Encoding category labels...")
category_embeddings = model.encode(
    CATEGORIES, convert_to_tensor=True, device=DEVICE
)

items = df[item_col].fillna("").astype(str).tolist()
item_categories = []

print("⏳ Classifying products using embedding similarity...")

BATCH_SIZE = 512

for i in tqdm(range(0, len(items), BATCH_SIZE)):
    batch = items[i:i + BATCH_SIZE]

    item_emb = model.encode(
        batch, convert_to_tensor=True, device=DEVICE
    )

    scores = util.cos_sim(item_emb, category_embeddings)

    best_indices = torch.argmax(scores, dim=1).tolist()

    for idx in best_indices:
        item_categories.append(CATEGORIES[idx])

df["category"] = item_categories

OUTPUT_PATH = os.path.join(BASE_DIR, "sales_with_categories_fast.csv")
df.to_csv(OUTPUT_PATH, index=False)

print("🎉 DONE! Fast file saved to:", OUTPUT_PATH)
print("⚡ Processed ALL rows in minutes!")
