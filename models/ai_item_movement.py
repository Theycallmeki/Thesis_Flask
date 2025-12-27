# models/ai_item_movement.py

from db import db
from datetime import datetime


class AIItemMovement(db.Model):
    __tablename__ = "ai_item_movements"

    id = db.Column(db.Integer, primary_key=True)

    # 🔗 link to items table (matches your design)
    item_id = db.Column(
        db.Integer,
        db.ForeignKey("items.id"),
        nullable=False
    )

    item_name = db.Column(db.String(255), nullable=False)

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

    avg_daily_sales = db.Column(db.Float, nullable=False)
    days_since_last_sale = db.Column(db.Integer, nullable=False)

    movement_class = db.Column(
        db.Enum("Fast", "Medium", "Slow", name="movement_enum"),
        nullable=False
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # relationship (optional but clean)
    item = db.relationship("Item", lazy=True)

    def __repr__(self):
        return f"<AIItemMovement Item{self.item_id} {self.movement_class}>"
