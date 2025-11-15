from flask import Flask
from flask_cors import CORS
from db import db
from urls import register_routes  
import subprocess
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///thesis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
register_routes(app)

@app.route('/')
def index():
    return {'message': 'Flask API running successfully'}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    # ---------------------------------------------------
    # FIX: Run ml/analyze_behavior.py with correct folder
    # ---------------------------------------------------
    BASE_DIR = os.getcwd()                     # C:\Users\acer\Downloads\Thesis_Flask-1
    ML_FOLDER = os.path.join(BASE_DIR, "ml")   # C:\Users\acer\Downloads\Thesis_Flask-1\ml
    ML_SCRIPT = os.path.join(ML_FOLDER, "analyze_behavior.py")

    print("Running ML script:", ML_SCRIPT)

    # Run script as: python ml/analyze_behavior.py
    subprocess.Popen(["python", ML_SCRIPT])

    # ---------------------------------------------------

    app.run(debug=True, host='0.0.0.0', port=5000)
