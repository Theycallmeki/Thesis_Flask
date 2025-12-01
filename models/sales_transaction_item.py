from db import db

class SalesTransactionItem(db.Model):
    __tablename__ = "sales_transaction_items"

    id = db.Column(db.Integer, primary_key=True)

    transaction_id = db.Column(
        db.Integer,
        db.ForeignKey("sales_transactions.id"),
        nullable=False
    )

    item_id = db.Column(
        db.Integer,
        db.ForeignKey("items.id"),
        nullable=False
    )

    quantity = db.Column(db.Integer, nullable=False)
    price_at_sale = db.Column(db.Numeric(10, 2), nullable=False)

    # Relationships
    transaction = db.relationship(
        "SalesTransaction",
        back_populates="items"
    )

    item = db.relationship(
        "Item",
        back_populates="transaction_items"
    )

    def __repr__(self):
        return f"<SalesTransactionItem T{self.transaction_id} Item{self.item_id}>"
