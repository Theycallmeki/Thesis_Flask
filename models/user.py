from db import db
from datetime import datetime

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    # ✅ roles
    role = db.Column(
        db.Enum('admin', 'customer', name='user_roles'),
        default='customer',
        nullable=False
    )

    # ✅ refresh token (cookie-based auth)
    refresh_token = db.Column(db.String(500), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"
