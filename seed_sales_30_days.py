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


def seed_sales_30_days(clear_existing=False):
    with app.app_context():

        # ---------------------------------
        # OPTIONAL: Clear existing sales
        # ---------------------------------
        if clear_existing:
            SalesTransactionItem.query.delete()
            SalesTransaction.query.delete()
            db.session.commit()
            print("🧹 Cleared existing sales data")

        items = Item.query.all()
        if not items:
            raise Exception("No items found. Seed items first.")

        now = datetime.utcnow()
        total_transactions = 0
        skipped_transactions = 0

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

                cart_size = random.randint(
                    MIN_ITEMS_PER_TRANSACTION,
                    MAX_ITEMS_PER_TRANSACTION
                )

                cart_items = random.sample(items, min(cart_size, len(items)))
                transaction_items = []

                # -----------------------------
                # Build transaction items FIRST
                # -----------------------------
                for item in cart_items:
                    qty = random.randint(MIN_QTY, MAX_QTY)

                    if item.quantity < qty:
                        continue

                    item.quantity -= qty

                    transaction_items.append(
                        SalesTransactionItem(
                            item=item,
                            quantity=qty,
                            price_at_sale=item.price
                        )
                    )

                # 🚨 Skip transaction if no valid items
                if not transaction_items:
                    skipped_transactions += 1
                    continue

                # -----------------------------
                # Create transaction ONLY now
                # -----------------------------
                transaction = SalesTransaction(
                    date=day_date.replace(
                        hour=random.randint(8, 21),
                        minute=random.randint(0, 59),
                        second=random.randint(0, 59)
                    )
                )

                db.session.add(transaction)
                db.session.flush()  # ensures transaction.id exists

                for ti in transaction_items:
                    ti.transaction = transaction
                    db.session.add(ti)

                total_transactions += 1

        db.session.commit()

        print("✅ Sales seeding complete")
        print(f"📊 Total transactions created: {total_transactions}")
        print(f"⏭️ Transactions skipped (no stock): {skipped_transactions}")
        print(f"📅 Date range: last {DAYS_BACK} days")


if __name__ == "__main__":
    # Set to True ONLY if you want to wipe sales history
    seed_sales_30_days(clear_existing=False)
