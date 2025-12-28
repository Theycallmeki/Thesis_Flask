# routes/recommendation.py
from flask import Blueprint, jsonify, current_app
from ml.recommendation import recommend_products, build_user_item_matrix
from models.user import User

recommendations_bp = Blueprint("recommendations_bp", __name__)

@recommendations_bp.route("/recommendations/<int:user_id>", methods=["GET"])
def get_recommendations(user_id):
    with current_app.app_context():
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Build the matrix dynamically in the app context
        user_item_matrix = build_user_item_matrix()
        items = recommend_products(user_id, user_item_matrix=user_item_matrix)

    if not items:
        return jsonify({
            "user_id": user_id,
            "recommendations": [],
            "message": "No recommendations available"
        }), 200

    return jsonify({
        "user_id": user_id,
        "recommendations": [
            {"id": item.id, "name": item.name, "category": item.category, "price": float(item.price)}
            for item in items
        ]
    }), 200
