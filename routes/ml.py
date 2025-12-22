# routes/ml.py  (FIXED — NO 3 DAYS)
from flask import Blueprint, jsonify, request
from ml.time_series_forecast import run_time_series_forecast
from models.ai_forecast import AIForecast
from db import db

ml_bp = Blueprint("ml_bp", __name__)

# ---------------------------------
# CREATE (run model + save)
# ---------------------------------
@ml_bp.route("/forecast", methods=["POST"])
def create_forecast():
    result = run_time_series_forecast()
    if result is None:
        return jsonify({"success": False, "message": "Not enough data"}), 400

    AIForecast.query.delete()

    for cat, qty in result["tomorrow"].items():
        db.session.add(AIForecast(
            horizon="tomorrow",
            category=cat,
            predicted_quantity=int(round(qty))
        ))

    for cat, qty in result["next_7_days"].items():
        db.session.add(AIForecast(
            horizon="7_days",
            category=cat,
            predicted_quantity=qty
        ))

    for cat, qty in result["next_30_days"].items():
        db.session.add(AIForecast(
            horizon="30_days",
            category=cat,
            predicted_quantity=qty
        ))

    db.session.commit()
    return jsonify({"success": True}), 201


# ---------------------------------
# READ (GROUPED JSON — CLEAN)
# ---------------------------------
@ml_bp.route("/forecast", methods=["GET"])
def get_forecasts_grouped():
    forecasts = AIForecast.query.order_by(AIForecast.category).all()

    grouped = {
        "tomorrow": [],
        "next_7_days": [],
        "next_30_days": []
    }

    for f in forecasts:
        data = {
            "id": f.id,
            "category": f.category,
            "predicted_quantity": f.predicted_quantity,
            "created_at": f.created_at.isoformat()
        }

        if f.horizon == "tomorrow":
            grouped["tomorrow"].append(data)
        elif f.horizon == "7_days":
            grouped["next_7_days"].append(data)
        elif f.horizon == "30_days":
            grouped["next_30_days"].append(data)

    return jsonify(grouped), 200


# ---------------------------------
# UPDATE
# ---------------------------------
@ml_bp.route("/forecast/<int:id>", methods=["PUT"])
def update_forecast(id):
    forecast = AIForecast.query.get(id)
    if not forecast:
        return jsonify({"error": "Forecast not found"}), 404

    data = request.get_json() or {}
    if "predicted_quantity" in data:
        forecast.predicted_quantity = int(data["predicted_quantity"])

    db.session.commit()
    return jsonify({"success": True}), 200


# ---------------------------------
# DELETE
# ---------------------------------
@ml_bp.route("/forecast/<int:id>", methods=["DELETE"])
def delete_forecast(id):
    forecast = AIForecast.query.get(id)
    if not forecast:
        return jsonify({"error": "Forecast not found"}), 404

    db.session.delete(forecast)
    db.session.commit()
    return jsonify({"success": True}), 200
