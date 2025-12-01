from db import db
from datetime import datetime

class SalesTransaction(db.Model):
    __tablename__ = "sales_transactions"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # One transaction → many items
    items = db.relationship(
        "SalesTransactionItem",
        back_populates="transaction",
        lazy=True,
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<SalesTransaction {self.id}>"
