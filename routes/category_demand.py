from flask import Blueprint, jsonify, request
from db import db
from models.sales_history import SalesHistory
import pandas as pd
import os
from datetime import datetime

category_demand_bp = Blueprint("category_demand", __name__)

# ---------------------------
# CREATE FROM CSV
# ---------------------------
@category_demand_bp.route("/import-category-demand", methods=["POST"])
def import_category_demand():
    csv_path = os.path.join("ml", "raw_sales.csv")

    if not os.path.exists(csv_path):
        return jsonify({"error": "CSV file not found"}), 404

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    grouped = (
        df.groupby(["Date", "category"])
          .size()
          .reset_index(name="count")
    )

    for _, row in grouped.iterrows():
        record = SalesHistory(
            date=row["Date"].date(),
            category=row["category"],
            quantity_sold=row["count"]
        )
        db.session.add(record)

    db.session.commit()

    return jsonify({"message": "Category demand imported successfully!"}), 201


# ---------------------------
# READ ALL
# ---------------------------
@category_demand_bp.route("/category-demand", methods=["GET"])
def get_all_category_demand():
    records = SalesHistory.query.all()
    return jsonify([
        {
            "id": r.id,
            "date": r.date.isoformat(),
            "category": r.category,
            "quantity_sold": r.quantity_sold
        }
        for r in records
    ])


# ---------------------------
# READ BY ID
# ---------------------------
@category_demand_bp.route("/category-demand/<int:id>", methods=["GET"])
def get_category_demand(id):
    record = SalesHistory.query.get(id)
    if not record:
        return jsonify({"error": "Record not found"}), 404

    return jsonify({
        "id": record.id,
        "date": record.date.isoformat(),
        "category": record.category,
        "quantity_sold": record.quantity_sold
    })


# ---------------------------
# READ BY DATE
# ---------------------------
@category_demand_bp.route("/category-demand/date/<string:date_str>", methods=["GET"])
def get_category_demand_by_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    records = SalesHistory.query.filter_by(date=date_obj).all()

    return jsonify([
        {
            "id": r.id,
            "date": r.date.isoformat(),
            "category": r.category,
            "quantity_sold": r.quantity_sold
        }
        for r in records
    ])


# ---------------------------
# READ BY CATEGORY
# ---------------------------
@category_demand_bp.route("/category-demand/category/<string:category>", methods=["GET"])
def get_category_demand_by_category(category):
    records = SalesHistory.query.filter_by(category=category).all()

    return jsonify([
        {
            "id": r.id,
            "date": r.date.isoformat(),
            "category": r.category,
            "quantity_sold": r.quantity_sold
        }
        for r in records
    ])


# ---------------------------
# UPDATE (PUT)
# ---------------------------
@category_demand_bp.route("/category-demand/<int:id>", methods=["PUT"])
def update_category_demand(id):
    data = request.json
    record = SalesHistory.query.get(id)

    if not record:
        return jsonify({"error": "Record not found"}), 404

    if "date" in data:
        try:
            record.date = datetime.strptime(data["date"], "%Y-%m-%d").date()
        except:
            return jsonify({"error": "Invalid date format"}), 400

    if "category" in data:
        record.category = data["category"]

    if "quantity_sold" in data:
        record.quantity_sold = data["quantity_sold"]

    db.session.commit()

    return jsonify({"message": "Record updated successfully"})


# ---------------------------
# DELETE
# ---------------------------
@category_demand_bp.route("/category-demand/<int:id>", methods=["DELETE"])
def delete_category_demand(id):
    record = SalesHistory.query.get(id)

    if not record:
        return jsonify({"error": "Record not found"}), 404

    db.session.delete(record)
    db.session.commit()

    return jsonify({"message": "Record deleted successfully"})
