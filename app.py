# app.py
# FINAL SAFE VERSION — NO .env

from flask import Flask
from flask_cors import CORS
import os

from db import db
from urls import register_routes

app = Flask(__name__)

# ✅ REQUIRED for JWT signing
app.config["SECRET_KEY"] = "super-secret"   # hard-coded

# ✅ COOKIE SETTINGS
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"

CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": "http://localhost:3000"}}
)

# ------------------------------
# Database config
# ------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:12345678@localhost:5432/niggas"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
register_routes(app)


@app.route("/")
def index():
    return {"message": "Flask API running successfully"}


# ------------------------------
# APP ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True, host="0.0.0.0", port=5000)
