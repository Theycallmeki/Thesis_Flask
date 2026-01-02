from db import db
from datetime import datetime


class AIStockoutRisk(db.Model):
    __tablename__ = "ai_stockout_risks"

    id = db.Column(db.Integer, primary_key=True)

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
        name='category_enum_stockout'
    ), nullable=False)

    current_stock = db.Column(db.Integer, nullable=False)
    avg_daily_sales = db.Column(db.Float, nullable=False)
    days_of_stock_left = db.Column(db.Float, nullable=False)

    risk_level = db.Column(
        db.Enum("Low", "Medium", "High", name="stockout_risk_enum"),
        nullable=False
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    item = db.relationship("Item", lazy=True)

    def __repr__(self):
        return f"<AIStockoutRisk Item{self.item_id} {self.risk_level}>"
