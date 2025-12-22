# app.py
# FINAL SAFE VERSION — ML RUNS VIA API ONLY

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

from db import db
from urls import register_routes

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# ------------------------------
# Database config
# ------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:12345678@localhost:5432/rat ni jehu"
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
    with app.app_context():
        db.create_all()

    app.run(debug=True, host="0.0.0.0", port=5000)
# urls.py