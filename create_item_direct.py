# create_item_direct.py
from app import app, db
from models.item import Item

with app.app_context():
    # Check if the item already exists
    existing = Item.query.filter_by(barcode="555-600").first()
    if existing:
        print("Item already exists:", existing)
    else:
        new_item = Item(
            name="Apple",
            category="Fruits",
            price=25.00,
            barcode="555-600",
            quantity=10  # optional
        )
        db.session.add(new_item)
        db.session.commit()
        print("Item created:", new_item)
