from flask import Blueprint, request, jsonify
from db import db
from datetime import datetime
from models.user import User  # Correct import from models/user.py

user_routes = Blueprint('user_routes', __name__)

# GET all users
@user_routes.route('/', methods=['GET'])
def get_users():
    users = User.query.all()
    users_list = [
        {
            "id": u.id, 
            "username": u.username, 
            "created_at": u.created_at, 
            "updated_at": u.updated_at
        } 
        for u in users
    ]
    return jsonify(users_list), 200

# GET a single user by id
@user_routes.route('/<int:id>', methods=['GET'])
def get_user(id):
    user = User.query.get(id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({
        "id": user.id, 
        "username": user.username, 
        "created_at": user.created_at, 
        "updated_at": user.updated_at
    }), 200

# CREATE a new user
@user_routes.route('/', methods=['POST'])
def create_user():
    data = request.json
    if not data.get('username') or not data.get('password'):
        return jsonify({"error": "Username and password required"}), 400

    if User.query.filter_by(username=data['username']).first():
        return jsonify({"error": "Username already exists"}), 400

    new_user = User(
        username=data['username'],
        password=data['password']  # Hash this in production!
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User created", "id": new_user.id}), 201

# UPDATE a user
@user_routes.route('/<int:id>', methods=['PUT'])
def update_user(id):
    user = User.query.get(id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    data = request.json
    if 'username' in data:
        user.username = data['username']
    if 'password' in data:
        user.password = data['password']  # Hash this in production!
    user.updated_at = datetime.utcnow()

    db.session.commit()
    return jsonify({"message": "User updated"}), 200

# DELETE a user
@user_routes.route('/<int:id>', methods=['DELETE'])
def delete_user(id):
    user = User.query.get(id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "User deleted"}), 200
