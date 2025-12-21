import random
from datetime import datetime, timedelta

from app import app
from db import db

from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item


# -----------------------------
# CONFIG
# -----------------------------
DAYS_BACK = 30
MIN_TRANSACTIONS_PER_DAY = 2
MAX_TRANSACTIONS_PER_DAY = 8

MIN_ITEMS_PER_TRANSACTION = 1
MAX_ITEMS_PER_TRANSACTION = 4

MIN_QTY = 1
MAX_QTY = 5


def seed_sales_30_days():
    with app.app_context():

        # ❗ Clear existing sales
        SalesTransactionItem.query.delete()
        SalesTransaction.query.delete()
        db.session.commit()

        print("🧹 Cleared existing sales data")

        items = Item.query.all()
        if not items:
            raise Exception("No items found. Seed items first.")

        now = datetime.utcnow()
        total_transactions = 0

        # -----------------------------
        # Loop through each day
        # -----------------------------
        for day_offset in range(DAYS_BACK):
            day_date = now - timedelta(days=day_offset)

            transactions_today = random.randint(
                MIN_TRANSACTIONS_PER_DAY,
                MAX_TRANSACTIONS_PER_DAY
            )

            for _ in range(transactions_today):
                transaction = SalesTransaction(
                    date=day_date.replace(
                        hour=random.randint(8, 21),
                        minute=random.randint(0, 59),
                        second=random.randint(0, 59)
                    )
                )

                db.session.add(transaction)

                cart_size = random.randint(
                    MIN_ITEMS_PER_TRANSACTION,
                    MAX_ITEMS_PER_TRANSACTION
                )

                cart_items = random.sample(items, min(cart_size, len(items)))

                for item in cart_items:
                    qty = random.randint(MIN_QTY, MAX_QTY)

                    # Skip if insufficient stock
                    if item.quantity < qty:
                        continue

                    item.quantity -= qty

                    db.session.add(SalesTransactionItem(
                        transaction=transaction,
                        item=item,
                        quantity=qty,
                        price_at_sale=item.price
                    ))

                total_transactions += 1

        db.session.commit()

        print("✅ Sales seeding complete")
        print(f"📊 Total transactions created: {total_transactions}")
        print(f"📅 Date range: last {DAYS_BACK} days")


if __name__ == "__main__":
    seed_sales_30_days()
