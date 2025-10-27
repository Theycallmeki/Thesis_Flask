import os
import pandas as pd

# ✅ Automatically find the correct folder (ml)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "raw_sales.csv")

# === Load dataset ===
df = pd.read_csv(RAW_PATH)

# === Define your categories ===
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

# === Helper: auto-detect correct column ===
def get_item_column(df):
    for possible_col in ['item', 'itemDescription', 'Item', 'ItemDescription']:
        if possible_col in df.columns:
            return possible_col
    raise KeyError("No valid item column found (expected 'item' or 'itemDescription')")

item_col = get_item_column(df)

# === Match each item to a category ===
def match_category(item):
    if pd.isna(item):
        return None
    item_lower = str(item).lower()
    for category, keywords in categories.items():
        if any(word in item_lower for word in keywords):
            return category
    return None

df['category'] = df[item_col].apply(match_category)
df['category'] = df['category'].fillna('Uncategorized')

# === Save the trained file inside /ml folder ===
TRAINED_PATH = os.path.join(BASE_DIR, "trained_sales.csv")
df.to_csv(TRAINED_PATH, index=False)

print(f"✅ Training complete! File saved at: {TRAINED_PATH}")
