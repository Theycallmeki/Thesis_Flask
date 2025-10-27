from flask import Blueprint, request, jsonify
from models.item import Item
from db import db

items_bp = Blueprint('items', __name__, url_prefix='/items')


# Helper: list of valid categories
def valid_categories():
    return [choice for choice in Item.__table__.columns.category.type.enums]


# 🟢 GET all items
@items_bp.route('/', methods=['GET'])
def get_items():
    try:
        items = Item.query.all()
        return jsonify([
            {
                'id': i.id,
                'name': i.name,
                'quantity': i.quantity,
                'category': i.category,
                'price': float(i.price),
                'barcode': i.barcode
            } for i in items
        ])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 🟢 GET item by barcode
@items_bp.route('/barcode/<string:barcode>', methods=['GET'])
def get_item_by_barcode(barcode):
    try:
        item = Item.query.filter_by(barcode=barcode).first()
        if not item:
            return jsonify({'error': 'Item not found'}), 404
        return jsonify({
            'id': item.id,
            'name': item.name,
            'quantity': item.quantity,
            'category': item.category,
            'price': float(item.price),
            'barcode': item.barcode
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 🟢 CREATE item
@items_bp.route('/', methods=['POST'])
def create_item():
    data = request.get_json()
    name = data.get('name')
    quantity = data.get('quantity', 0)
    category = data.get('category')
    price = data.get('price', 0.00)
    barcode = data.get('barcode')

    try:
        # Validate category
        if category not in valid_categories():
            return jsonify({'error': f"Invalid category. Allowed: {', '.join(valid_categories())}"}), 400

        # Create item
        new_item = Item(name=name, quantity=quantity, category=category, price=price, barcode=barcode)
        db.session.add(new_item)
        db.session.commit()
        return jsonify({
            'id': new_item.id,
            'name': new_item.name,
            'quantity': new_item.quantity,
            'category': new_item.category,
            'price': float(new_item.price),
            'barcode': new_item.barcode
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


# 🟡 UPDATE item
@items_bp.route('/<int:id>', methods=['PUT'])
def update_item(id):
    data = request.get_json()
    try:
        item = Item.query.get(id)
        if not item:
            return jsonify({'error': 'Item not found'}), 404

        category = data.get('category')
        if category and category not in valid_categories():
            return jsonify({'error': f"Invalid category. Allowed: {', '.join(valid_categories())}"}), 400

        # Update fields
        item.name = data.get('name', item.name)
        item.category = category or item.category
        item.barcode = data.get('barcode', item.barcode)

        if 'price' in data:
            item.price = data['price']
        if 'quantity' in data:
            item.quantity = data['quantity']

        db.session.commit()
        return jsonify({
            'id': item.id,
            'name': item.name,
            'quantity': item.quantity,
            'category': item.category,
            'price': float(item.price),
            'barcode': item.barcode
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


# 🔴 DELETE item
@items_bp.route('/<int:id>', methods=['DELETE'])
def delete_item(id):
    try:
        item = Item.query.get(id)
        if not item:
            return jsonify({'error': 'Item not found'}), 404

        db.session.delete(item)
        db.session.commit()
        return jsonify({'message': 'Item deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400
