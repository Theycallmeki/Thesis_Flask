# app.py
# FINAL SAFE VERSION — ML RUNS ONCE, NO CIRCULAR IMPORTS

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

from db import db
from urls import register_routes

# ✅ Import ML function (SAFE: ML does NOT import app)
from ml.time_series_forecast import run_time_series_forecast

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# ------------------------------
# Database config
# ------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:12345678@localhost:5432/THESIS"
)
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

    # 🔒 Prevent double-run in Flask debug reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:

        with app.app_context():
            db.create_all()

            print("\n==============================")
            print(" Running TIME-SERIES ANALYSIS ")
            print("==============================\n")

            run_time_series_forecast()

            print("\n==============================")
            print(" ML ANALYSIS FINISHED ")
            print("==============================\n")

    # ------------------------------
    # RUN SERVER
    # ------------------------------
    app.run(debug=True, host="0.0.0.0", port=5000)
