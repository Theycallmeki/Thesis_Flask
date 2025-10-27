# app.py
from flask import Flask
from flask_cors import CORS
from db import db
from urls import register_routes   # <-- Import here

app = Flask(__name__)
CORS(app)  # allow frontend access

# Database config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///thesis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Register all routes from urls.py
register_routes(app)

@app.route('/')
def index():
    return {'message': 'Flask API running successfully 🎉'}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
