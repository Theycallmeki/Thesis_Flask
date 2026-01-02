# app.py
# FINAL SAFE VERSION — NO .env — NO RELOADER

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
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"

# ==============================
# CORS
# ==============================
CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": "http://localhost:3000"}}
)

# ==============================
# DATABASE
# ==============================
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:12345678@localhost:5432/2026"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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
        use_reloader=False  
    )
