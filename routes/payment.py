# routes/payment.py
from flask import Blueprint, request, jsonify
import requests
from requests.auth import HTTPBasicAuth
import json
from db import db
from models.item import Item
import os
from dotenv import load_dotenv

load_dotenv()

payment_bp = Blueprint("payment", __name__, url_prefix="/payment")

PAYMONGO_SECRET_KEY = os.getenv("PAYMONGO_TEST_SECRET_KEY", "")
auth = HTTPBasicAuth(PAYMONGO_SECRET_KEY, "")

# Create Payment Intent (GCash)
@payment_bp.route("/intent", methods=["POST"])
def create_payment_intent():
    data = request.get_json()
    amount = data.get("amount", 2000)
    currency = data.get("currency", "PHP")

    url = "https://api.paymongo.com/v1/payment_intents"
    payload = {
        "data": {
            "attributes": {
                "amount": amount,
                "payment_method_allowed": ["gcash"],
                "payment_method_options": {},
                "currency": currency,
                "capture_type": "automatic"
            }
        }
    }

    response = requests.post(url, json=payload, auth=auth)
    data = response.json()
    
    print("=== PayMongo /intent response ===")
    print(json.dumps(data, indent=4))
    print("=== End response ===")

    if response.status_code not in (200, 201):
        return jsonify({"error": data}), response.status_code

    payment_intent_id = data.get("data", {}).get("id")
    return jsonify({"id": payment_intent_id}), response.status_code

# Create Checkout Session
@payment_bp.route("/checkout", methods=["POST"])
def create_checkout_session():
    data = request.get_json()
    payment_intent_id = data.get("payment_intent_id")
    success_url = data.get("success_url", "http://localhost:3000/success")
    cancel_url = data.get("cancel_url", "http://localhost:3000/cancel")
    cart_items = data.get("cart", [])

    if not payment_intent_id:
        return jsonify({"error": "Missing payment_intent_id"}), 400

    # Prepare line_items for PayMongo
    line_items = [
        {
            "name": item["name"],
            "amount": int(round(item["price"] * 100)),  # convert to centavos
            "currency": "PHP",
            "quantity": max(1, item.get("quantity", 1)),
            "barcode": item.get("barcode")  # include barcode for tracking
        }
        for item in cart_items if item.get("price", 0) > 0
    ]

    url = "https://api.paymongo.com/v1/checkout_sessions"
    payload = {
        "data": {
            "attributes": {
                "payment_intent": payment_intent_id,
                "success_url": success_url,
                "cancel_url": cancel_url,
                "send_email_receipt": False,
                "show_description": False,
                "show_line_items": True,
                "payment_method_types": ["gcash"],
                "line_items": line_items
            }
        }
    }

    response = requests.post(url, json=payload, auth=auth)
    data = response.json()

    print("=== PayMongo /checkout response ===")
    print(json.dumps(data, indent=4))

    if response.status_code not in (200, 201):
        return jsonify({"error": data}), response.status_code

    checkout_url = data.get("data", {}).get("attributes", {}).get("checkout_url")
    if not checkout_url:
        return jsonify({"error": "Failed to get checkout URL", "raw": data}), 500

    return jsonify({"checkoutUrl": checkout_url}), 200



# Webhook: handle successful payments
@payment_bp.route("/webhook", methods=["POST"])
def paymongo_webhook():
    payload = request.get_json()

    # Log the entire webhook for debugging
    print("=== Incoming PayMongo Webhook ===")
    print(json.dumps(payload, indent=4))
    print("=== End Webhook ===")

    event_type = payload.get("data", {}).get("attributes", {}).get("event", "")

    if event_type == "checkout_session.payment.paid":
        session_id = payload["data"]["id"]
        print(f"✅ Payment successful for checkout session: {session_id}")

        # Update inventory quantities
        line_items = payload["data"]["attributes"].get("line_items", [])
        for item in line_items:
            barcode = item.get("barcode")
            quantity = item.get("quantity", 1)
            if not barcode:
                continue

            db_item = Item.query.filter_by(barcode=barcode).first()
            if db_item:
                db_item.quantity = max(0, db_item.quantity - quantity)
                db.session.commit()
                print(f"Updated {db_item.name} quantity to {db_item.quantity}")

    return jsonify({"status": "success"}), 200
