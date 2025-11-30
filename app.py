# app.py
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

from db import db
from urls import register_routes

# ✅ import the ML runner function (make sure ml/analyze_behavior.py defines run_ml())
from ml.analyze_behavior import run_ml

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Enable CORS for your React frontend
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# ------------------------------
# Database config (PostgreSQL)
# ------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:12345678@localhost:5432/THESIS"  # fallback if .env missing
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize SQLAlchemy
db.init_app(app)

# Register all blueprints
register_routes(app)

@app.route("/")
def index():
    return {"message": "Flask API running successfully"}

if __name__ == "__main__":
    # Create tables if they don't exist yet
    with app.app_context():
        db.create_all()

    # ------------------------------
    # ✅ RUN ML INSIDE app.py (ONLY ONCE)
    # ------------------------------
    # Flask debug mode runs the app twice because of the reloader.
    # This guard ensures ML runs only in the "real" server process.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        print("Running ML inside app.py ONCE...")
        run_ml()
    # ------------------------------

    # Run Flask server
    app.run(debug=True, host="0.0.0.0", port=5000)
