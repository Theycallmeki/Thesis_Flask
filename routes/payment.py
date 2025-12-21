from flask import Blueprint, request, jsonify
import requests
from requests.auth import HTTPBasicAuth
import json
from db import db
from models.item import Item
import os
from dotenv import load_dotenv
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem

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
    cart_items = data.get("cart", [])

    # Store cart in metadata
    cart_metadata = [
        {"barcode": item["barcode"], "quantity": item["quantity"]}
        for item in cart_items
    ]

    url = "https://api.paymongo.com/v1/payment_intents"
    payload = {
        "data": {
            "attributes": {
                "amount": amount,
                "payment_method_allowed": ["gcash"],
                "payment_method_options": {},
                "currency": currency,
                "capture_type": "automatic",
                "metadata": {
                    "cart": json.dumps(cart_metadata)
                }
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

    # Prepare line_items for display (barcode is only in metadata)
    line_items = [
        {
            "name": item["name"],
            "amount": int(round(item["price"] * 100)),
            "currency": "PHP",
            "quantity": max(1, item.get("quantity", 1))
        }
        for item in cart_items if item.get("price", 0) > 0
    ]

    # Include full cart in metadata
    cart_metadata = [{"barcode": item["barcode"], "quantity": item["quantity"]} for item in cart_items]

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
                "line_items": line_items,
                "metadata": {"cart": json.dumps(cart_metadata)}  # <-- important
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

    return jsonify({"checkoutUrl": checkout_url, "sessionId": data.get("data", {}).get("id")}), 200


@payment_bp.route("/webhook", methods=["POST"])
def paymongo_webhook():
    payload = request.get_json()

    print("=== Incoming PayMongo Webhook ===")
    print(json.dumps(payload, indent=4))
    print("=== End Webhook ===")

    event_type = payload.get("data", {}).get("attributes", {}).get("type", "")

    if event_type == "checkout_session.payment.paid":
        session_id = payload["data"]["id"]
        print(f"Payment successful for checkout session: {session_id}")

        # Extract metadata cart
        session_data = payload["data"]["attributes"].get("data", {}).get("attributes", {})
        metadata = session_data.get("metadata", {})
        cart_json = metadata.get("cart", "[]")
        cart_items = json.loads(cart_json)

        try:
            # Create sales transaction
            transaction = SalesTransaction()
            db.session.add(transaction)
            db.session.flush()  # get transaction.id

            # Process each cart item
            for cart_item in cart_items:
                barcode = cart_item["barcode"]
                quantity = cart_item.get("quantity", 1)

                item = Item.query.filter_by(barcode=barcode).first()
                if not item:
                    raise Exception(f"Item not found: {barcode}")

                if item.quantity < quantity:
                    raise Exception(f"Insufficient stock for {item.name}")

                # Create transaction item
                tx_item = SalesTransactionItem(
                    transaction_id=transaction.id,
                    item_id=item.id,
                    quantity=quantity,
                    price_at_sale=item.price
                )
                db.session.add(tx_item)

                # Update inventory
                item.quantity -= quantity

                print(
                    f"Sold {quantity}x {item.name} | Remaining stock: {item.quantity}"
                )

            # Commit everything once
            db.session.commit()
            print(f"Sales transaction {transaction.id} recorded successfully")

        except Exception as e:
            db.session.rollback()
            print("Payment processing failed:", str(e))
            return jsonify({"error": str(e)}), 500

    return jsonify({"status": "success"}), 200


