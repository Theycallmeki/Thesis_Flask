from flask import Blueprint, jsonify, request
from models.sales_history import SalesHistory
from models.item import Item
from db import db
from datetime import date

sales_bp = Blueprint('sales', __name__)


# ✅ GET all sales
@sales_bp.route('/', methods=['GET'])
def get_all_sales():
    sales = SalesHistory.query.all()
    result = []

    for s in sales:
        result.append({
            "id": s.id,
            "item_id": s.item_id,
            "item_name": s.item.name if s.item else None,
            "category": s.item.category if s.item else None,
            "date": s.date.isoformat(),
            "quantity_sold": s.quantity_sold
        })

    return jsonify(result), 200


# ✅ GET sale by ID
@sales_bp.route('/<int:id>', methods=['GET'])
def get_sale_by_id(id):
    sale = SalesHistory.query.get(id)

    if not sale:
        return jsonify({"error": "Sale not found"}), 404

    return jsonify({
        "id": sale.id,
        "item_id": sale.item_id,
        "item_name": sale.item.name if sale.item else None,
        "category": sale.item.category if sale.item else None,
        "date": sale.date.isoformat(),
        "quantity_sold": sale.quantity_sold
    }), 200


# ✅ CREATE new sale
@sales_bp.route('/', methods=['POST'])
def create_sale():
    data = request.get_json() or {}

    item_id = data.get('item_id')
    quantity_sold = data.get('quantity_sold')
    sale_date = data.get('date', date.today().isoformat())

    # ✅ Validate inputs
    if item_id is None or quantity_sold is None:
        return jsonify({"error": "item_id and quantity_sold are required"}), 400

    if quantity_sold <= 0:
        return jsonify({"error": "quantity_sold must be greater than 0"}), 400

    # Check item exists
    item = Item.query.get(item_id)
    if not item:
        return jsonify({"error": "Invalid item_id"}), 400

    # Prevent overselling
    if item.quantity < quantity_sold:
        return jsonify({"error": "Not enough stock"}), 400

    # Create sale record
    sale = SalesHistory(
        item_id=item_id,
        quantity_sold=quantity_sold,
        date=date.fromisoformat(sale_date)
    )

    # Update item stock
    item.quantity -= quantity_sold

    db.session.add(sale)
    db.session.commit()

    return jsonify({
        "message": "Sale recorded successfully",
        "sale_id": sale.id,
        "remaining_stock": item.quantity
    }), 201


# ✅ UPDATE sale record (fixes inventory delta)
@sales_bp.route('/<int:id>', methods=['PUT'])
def update_sale(id):
    sale = SalesHistory.query.get(id)
    if not sale:
        return jsonify({"error": "Sale not found"}), 404

    data = request.get_json() or {}

    item = sale.item
    if not item:
        return jsonify({"error": "Linked item not found"}), 400

    # Update quantity_sold with stock adjustment
    if "quantity_sold" in data:
        new_qty = data["quantity_sold"]

        if new_qty <= 0:
            return jsonify({"error": "quantity_sold must be greater than 0"}), 400

        old_qty = sale.quantity_sold
        diff = new_qty - old_qty  # + diff means sell more, - diff means sell less

        # If selling more, check stock
        if diff > 0 and item.quantity < diff:
            return jsonify({"error": "Not enough stock to increase sale"}), 400

        # Apply stock change
        item.quantity -= diff
        sale.quantity_sold = new_qty

    # Update date
    if "date" in data:
        sale.date = date.fromisoformat(data["date"])

    db.session.commit()
    return jsonify({"message": "Sale updated successfully"}), 200


# ✅ DELETE sale record (restores stock)
@sales_bp.route('/<int:id>', methods=['DELETE'])
def delete_sale(id):
    sale = SalesHistory.query.get(id)
    if not sale:
        return jsonify({"error": "Sale not found"}), 404

    item = sale.item
    if item:
        item.quantity += sale.quantity_sold

    db.session.delete(sale)
    db.session.commit()

    return jsonify({"message": "Sale deleted successfully"}), 200
