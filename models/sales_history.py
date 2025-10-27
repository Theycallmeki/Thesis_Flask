from db import db
from datetime import date

class SalesHistory(db.Model):
    __tablename__ = 'sales_history'

    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('items.id'), nullable=False)
    date = db.Column(db.Date, default=date.today, nullable=False)
    quantity_sold = db.Column(db.Integer, nullable=False)

    item = db.relationship('Item', back_populates='sales')

    def __repr__(self):
        return f"<SalesHistory {self.id} - Item {self.item_id}>"
