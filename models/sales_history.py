from db import db
from datetime import date

class SalesHistory(db.Model):
    __tablename__ = 'sales_history'

    id = db.Column(db.Integer, primary_key=True)

    # ✅ IMPORTANT FIX:
    # DB column is "itemId" (camelCase), but we want to use item_id in Python.
    item_id = db.Column(
        "itemId",
        db.Integer,
        db.ForeignKey('items.id'),
        nullable=False
    )

    # ✅ Linked back to Item explicitly
    item = db.relationship("Item", back_populates="sales_history")

    date = db.Column(db.Date, default=date.today, nullable=False)
    quantity_sold = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<SalesHistory item_id={self.item_id} date={self.date} qty={self.quantity_sold}>"
