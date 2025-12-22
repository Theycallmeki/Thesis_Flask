import random
import uuid

from app import app
from db import db
from models.item import Item


# -----------------------------
# CONFIG
# -----------------------------
ITEMS_PER_CATEGORY = 8

MIN_PRICE = 10.00
MAX_PRICE = 500.00

MIN_STOCK = 100
MAX_STOCK = 500


CATEGORIES = {
    "Fruits": ["Apple", "Banana", "Orange", "Mango", "Grapes", "Pineapple"],
    "Vegetables": ["Carrot", "Broccoli", "Spinach", "Potato", "Tomato"],
    "Meat": ["Chicken Breast", "Pork Chop", "Beef Steak", "Ground Beef"],
    "Seafood": ["Salmon", "Tuna", "Shrimp", "Tilapia"],
    "Dairy": ["Milk", "Cheese", "Butter", "Yogurt"],
    "Beverages": ["Cola", "Orange Juice", "Water", "Coffee", "Tea"],
    "Snacks": ["Chips", "Cookies", "Popcorn", "Chocolate Bar"],
    "Bakery": ["Bread", "Croissant", "Muffin", "Donut"],
    "Frozen": ["Frozen Pizza", "Ice Cream", "Frozen Nuggets"],
    "Canned Goods": ["Canned Tuna", "Canned Corn", "Canned Beans"],
    "Condiments": ["Ketchup", "Mayonnaise", "Soy Sauce"],
    "Dry Goods": ["Sugar", "Salt", "Flour"],
    "Grains & Pasta": ["Rice", "Spaghetti", "Macaroni"],
    "Spices & Seasonings": ["Pepper", "Paprika", "Cumin"],
    "Breakfast & Cereal": ["Cornflakes", "Oatmeal", "Granola"],
    "Personal Care": ["Shampoo", "Soap", "Toothpaste"],
    "Household": ["Trash Bags", "Light Bulbs", "Paper Towels"],
    "Baby Products": ["Baby Diapers", "Baby Wipes"],
    "Pet Supplies": ["Dog Food", "Cat Litter"],
    "Health & Wellness": ["Vitamins", "Pain Reliever"],
    "Cleaning Supplies": ["Laundry Detergent", "Dish Soap"]
}


def generate_barcode():
    return str(uuid.uuid4().int)[:13]


def seed_items(clear_existing=False):
    with app.app_context():

        # ---------------------------------
        # OPTIONAL: Clear existing items
        # ---------------------------------
        if clear_existing:
            Item.query.delete()
            db.session.commit()
            print("🧹 Cleared existing items")

        existing_barcodes = {
            item.barcode for item in Item.query.with_entities(Item.barcode).all()
        }

        total_items = 0

        # -----------------------------
        # Create items by category
        # -----------------------------
        for category, names in CATEGORIES.items():

            for i in range(ITEMS_PER_CATEGORY):
                base_name = random.choice(names)

                name = f"{base_name} {i + 1}"

                barcode = generate_barcode()
                while barcode in existing_barcodes:
                    barcode = generate_barcode()

                existing_barcodes.add(barcode)

                item = Item(
                    name=name,
                    category=category,
                    price=round(random.uniform(MIN_PRICE, MAX_PRICE), 2),
                    quantity=random.randint(MIN_STOCK, MAX_STOCK),
                    barcode=barcode
                )

                db.session.add(item)
                total_items += 1

        db.session.commit()

        print("✅ Item seeding complete")
        print(f"📦 Total items created: {total_items}")


if __name__ == "__main__":
    # Set to True ONLY if you want to wipe items table
    seed_items(clear_existing=False)
