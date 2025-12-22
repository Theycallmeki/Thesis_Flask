# models/ai_forecast.py
from db import db
from datetime import datetime

class AIForecast(db.Model):
    __tablename__ = "ai_forecasts"

    id = db.Column(db.Integer, primary_key=True)

    # metadata
    model_name = db.Column(db.String(100), nullable=False, default="lstm_time_series")
    horizon = db.Column(db.String(20), nullable=False)  # tomorrow | 7_days | 30_days

    # stored prediction
    category = db.Column(db.String(100), nullable=False)
    predicted_quantity = db.Column(db.Integer, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "horizon": self.horizon,
            "category": self.category,
            "predicted_quantity": self.predicted_quantity,
            "created_at": self.created_at.isoformat(),
        }
