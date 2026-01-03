# app.py
# FINAL SAFE VERSION — COOKIE AUTH + POSTGRES SAFE

from flask import Flask
from flask_cors import CORS

from db import db
from urls import register_routes

app = Flask(__name__)

# ==============================
# SECURITY
# ==============================
app.config["SECRET_KEY"] = "super-secret"

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = False

# ==============================
# CORS (ALLOW COOKIES)
# ==============================
CORS(
    app,
    supports_credentials=True,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173"
            ]
        }
    }
)

# ==============================
# DATABASE
# ==============================
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql://kian_nf61_user:"
    "dZ4z60B6JFAL8QFNqt2dN3f8FsfkmG7p"
    "@dpg-d58kic2li9vc73a4k8v0-a.oregon-postgres.render.com/kian_nf61"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 🔴 CRITICAL FIX (PREVENTS DB DISCONNECT CRASH)
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 280,
}

db.init_app(app)

# ==============================
# ROUTES
# ==============================
register_routes(app)

@app.route("/")
def index():
    return {"message": "Flask API running successfully"}

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
    )
