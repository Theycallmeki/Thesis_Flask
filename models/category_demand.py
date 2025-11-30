from db import db
from datetime import date

class CategoryDemand(db.Model):
    __tablename__ = 'category_demand'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    category = db.Column(db.String(255), nullable=False)
    count = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<CategoryDemand {self.category} - {self.date} ({self.count})>"
