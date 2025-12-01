from db import db

class Item(db.Model):
    __tablename__ = 'items'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    quantity = db.Column(db.Integer, default=0)

    category = db.Column(db.Enum(
        'Fruits',
        'Vegetables',
        'Meat',
        'Seafood',
        'Dairy',
        'Beverages',
        'Snacks',
        'Bakery',
        'Frozen',
        'Canned Goods',
        'Condiments',
        'Dry Goods',
        'Grains & Pasta',
        'Spices & Seasonings',
        'Breakfast & Cereal',
        'Personal Care',
        'Household',
        'Baby Products',
        'Pet Supplies',
        'Health & Wellness',
        'Cleaning Supplies',
        name='category_enum'
    ), nullable=False)

    price = db.Column(db.Numeric(10, 2), nullable=False, default=0.00)
    barcode = db.Column(db.String(255), unique=True, nullable=False)

    # 🔄 Replaced: sales_history → transaction_items
    transaction_items = db.relationship(
        "SalesTransactionItem",
        back_populates="item",
        lazy=True,
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Item {self.name}>"
