# routes/users.py
# FULL FILE — COOKIE AUTH + /me — CORS SAFE

from flask import Blueprint, request, jsonify, make_response, g
from db import db
from datetime import datetime, timedelta
from models.user import User
import jwt

from utils.auth_restrict import require_auth  # ✅ USE SINGLE AUTH

user_routes = Blueprint("user_routes", __name__)

# 🔐 JWT CONFIG
JWT_SECRET = "super-secret"
ACCESS_EXPIRES = timedelta(minutes=15)
REFRESH_EXPIRES = timedelta(days=7)


# --------------------------------------------------
# TOKEN CREATION
# --------------------------------------------------
def create_token(user_id, token_type="access"):
    payload = {
        "user_id": user_id,
        "type": token_type,
        "exp": datetime.utcnow()
        + (ACCESS_EXPIRES if token_type == "access" else REFRESH_EXPIRES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


# --------------------------------------------------
# GET CURRENT USER (USED BY VUE)
# --------------------------------------------------
@user_routes.route("/me", methods=["GET"])
@require_auth(roles=None)   # admin / staff / customer
def me():
    user = g.current_user
    return jsonify({
        "authenticated": True,
        "id": user.id,
        "username": user.username,
        "role": user.role
    }), 200


# --------------------------------------------------
# GET ALL USERS
# --------------------------------------------------
@user_routes.route("", methods=["GET"])
@user_routes.route("/", methods=["GET"])
def get_users():
    users = User.query.all()
    return jsonify([
        {
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "created_at": u.created_at,
            "updated_at": u.updated_at,
        }
        for u in users
    ]), 200


# --------------------------------------------------
# GET USER BY ID
# --------------------------------------------------
@user_routes.route("/<int:id>", methods=["GET"])
def get_user(id):
    user = User.query.get(id)
    if not user:
        return jsonify({"error": "user not found"}), 404

    return jsonify({
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }), 200


# --------------------------------------------------
# CREATE USER
# --------------------------------------------------
@user_routes.route("", methods=["POST"])
@user_routes.route("/", methods=["POST"])
def create_user():
    data = request.json or {}

    if not data.get("username") or not data.get("password"):
        return jsonify({"error": "username and password required"}), 400

    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "username exists"}), 400

    user = User(
        username=data["username"],
        password=data["password"],
        role=data.get("role", "customer"),
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "user created", "id": user.id}), 201


# --------------------------------------------------
# LOGIN (COOKIE BASED)
# --------------------------------------------------
@user_routes.route("/login", methods=["POST"])
def login():
    data = request.json or {}

    user = User.query.filter_by(username=data.get("username")).first()
    if not user or user.password != data.get("password"):
        return jsonify({"error": "invalid credentials"}), 401

    access_token = create_token(user.id, "access")
    refresh_token = create_token(user.id, "refresh")

    user.refresh_token = refresh_token
    db.session.commit()

    resp = make_response(jsonify({
        "message": "login success",
        "role": user.role
    }))

    # ✅ MUST MATCH CORS (SameSite=None)
    resp.set_cookie(
        "access_token",
        access_token,
        httponly=True,
        samesite="None",
        secure=False
    )
    resp.set_cookie(
        "refresh_token",
        refresh_token,
        httponly=True,
        samesite="None",
        secure=False
    )

    return resp, 200


# --------------------------------------------------
# REFRESH TOKEN
# --------------------------------------------------
@user_routes.route("/refresh", methods=["POST"])
def refresh():
    token = request.cookies.get("refresh_token")
    if not token:
        return jsonify({"error": "no refresh token"}), 401

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])

        if payload["type"] != "refresh":
            return jsonify({"error": "invalid token type"}), 401

        user = User.query.get(payload["user_id"])
        if not user or user.refresh_token != token:
            return jsonify({"error": "invalid refresh"}), 401

        new_access = create_token(user.id, "access")

        resp = make_response(jsonify({"message": "token refreshed"}))
        resp.set_cookie(
            "access_token",
            new_access,
            httponly=True,
            samesite="None",
            secure=False
        )
        return resp, 200

    except:
        return jsonify({"error": "invalid refresh token"}), 401


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------
@user_routes.route("/logout", methods=["POST"])
def logout():
    token = request.cookies.get("refresh_token")
    if token:
        user = User.query.filter_by(refresh_token=token).first()
        if user:
            user.refresh_token = None
            db.session.commit()

    resp = make_response(jsonify({"message": "logged out"}))
    resp.delete_cookie("access_token")
    resp.delete_cookie("refresh_token")
    return resp, 200
