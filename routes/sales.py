from flask import Blueprint, jsonify, request
from db import db
from datetime import datetime

from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item

sales_bp = Blueprint("sales", __name__)


# 🔵 GET all transactions
@sales_bp.route("/", methods=["GET"])
def get_all_transactions():
    transactions = SalesTransaction.query.all()
    result = []

    for t in transactions:
        result.append({
            "transaction_id": t.id,
            "date": t.date.isoformat(),
            "items": [
                {
                    "item_id": ti.item_id,
                    "item_name": ti.item.name,
                    "quantity": ti.quantity,
                    "price_at_sale": float(ti.price_at_sale)
                }
                for ti in t.items
            ]
        })

    return jsonify(result), 200


# 🔵 GET a single transaction by ID
@sales_bp.route("/<int:id>", methods=["GET"])
def get_transaction(id):
    t = SalesTransaction.query.get(id)
    if not t:
        return jsonify({"error": "Transaction not found"}), 404

    return jsonify({
        "transaction_id": t.id,
        "date": t.date.isoformat(),
        "items": [
            {
                "item_id": ti.item_id,
                "item_name": ti.item.name,
                "quantity": ti.quantity,
                "price_at_sale": float(ti.price_at_sale)
            }
            for ti in t.items
        ]
    }), 200


# 🔵 CREATE new transaction (cart checkout)
@sales_bp.route("/", methods=["POST"])
def create_transaction():
    data = request.get_json() or {}

    cart_items = data.get("items", [])  
    # format: [{ "item_id": 1, "quantity": 2 }, ...]

    if not cart_items:
        return jsonify({"error": "No items provided"}), 400

    # Create transaction
    transaction = SalesTransaction()
    db.session.add(transaction)

    for entry in cart_items:
        item_id = entry.get("item_id")
        qty = entry.get("quantity")

        if not item_id or not qty:
            return jsonify({"error": "item_id and quantity required for each cart item"}), 400

        item = Item.query.get(item_id)
        if not item:
            return jsonify({"error": f"Item {item_id} not found"}), 400

        if item.quantity < qty:
            return jsonify({"error": f"Not enough stock for {item.name}"}), 400

        # Reduce stock
        item.quantity -= qty

        # Create transaction item entry
        trans_item = SalesTransactionItem(
            transaction=transaction,
            item=item,
            quantity=qty,
            price_at_sale=item.price
        )
        db.session.add(trans_item)

    db.session.commit()

    return jsonify({
        "message": "Transaction recorded",
        "transaction_id": transaction.id
    }), 201


# 🔵 DELETE transaction (restore stock)
@sales_bp.route("/<int:id>", methods=["DELETE"])
def delete_transaction(id):
    t = SalesTransaction.query.get(id)
    if not t:
        return jsonify({"error": "Transaction not found"}), 404

    # Restore stock for each item
    for ti in t.items:
        item = ti.item
        item.quantity += ti.quantity

    db.session.delete(t)
    db.session.commit()

    return jsonify({"message": "Transaction deleted"}), 200
