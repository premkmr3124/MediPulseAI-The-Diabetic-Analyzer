# -*- coding: utf-8 -*-
import os
from datetime import datetime
from dotenv import load_dotenv

import joblib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from pymongo import MongoClient, DESCENDING
from pymongo.server_api import ServerApi
import certifi

# ─── Load environment variables (.env file) ───────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "medipulse-secret-change-in-prod")

# ─── MongoDB Atlas connection ─────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")          # set in .env
client     = MongoClient(
    MONGO_URI,
    server_api=ServerApi('1'),
    tls=True,
    tlsCAFile=certifi.where(),
)
db         = client["medipulse"]            # database name
users_col  = db["users"]                    # collection: users
history_col= db["history"]                  # collection: history

# Ensure unique index on username
users_col.create_index("username", unique=True)
# Ensure index on history for fast per-user queries
history_col.create_index([("username", 1), ("timestamp", DESCENDING)])

# ─── Flask-Login ──────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = ""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    doc = users_col.find_one({"username": user_id})
    if doc:
        return User(user_id)
    return None

# ─── User helpers ─────────────────────────────────────────────────────────────
def get_user(username):
    return users_col.find_one({"username": username})

def create_user(username, password):
    users_col.insert_one({
        "username": username,
        "password": generate_password_hash(password),
        "created_at": datetime.utcnow(),
    })

# ─── Seed default accounts if collection is empty ────────────────────────────
if users_col.count_documents({}) == 0:
    create_user("admin",  "admin123")
    create_user("doctor", "doctor123")

# ─── History helpers ──────────────────────────────────────────────────────────
def add_to_history(username, inputs, result, result_type, probability):
    history_col.insert_one({
        "username":    username,
        "timestamp":   datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "inputs":      inputs,
        "result":      result,
        "result_type": result_type,
        "probability": probability,
        "created_at":  datetime.utcnow(),
    })

def get_history(username, limit=50):
    cursor = history_col.find(
        {"username": username},
        {"_id": 0}
    ).sort("created_at", DESCENDING).limit(limit)
    return list(cursor)

def clear_user_history(username):
    history_col.delete_many({"username": username})

# ─── ML model & preprocessors ────────────────────────────────────────────────
model    = load_model(os.path.join(BASE_DIR, "diabetes_ann_smote_model.keras"))
le_gen   = joblib.load(os.path.join(BASE_DIR, "le_gender.pkl"))
le_smoke = joblib.load(os.path.join(BASE_DIR, "le_smoking.pkl"))
scaler   = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# ─── Auth routes ─────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        doc = get_user(username)
        if doc and check_password_hash(doc["password"], password):
            login_user(User(username), remember=True)
            return redirect(url_for("home"))
        error = "Invalid username or password. Please try again."
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        username  = request.form.get("username", "").strip()
        password  = request.form.get("password", "")
        password2 = request.form.get("password2", "")
        if not username or not password:
            error = "Username and password are required."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != password2:
            error = "Passwords do not match."
        elif get_user(username):
            error = f"Username '{username}' is already taken."
        else:
            create_user(username, password)
            login_user(User(username), remember=True)
            return redirect(url_for("home"))
    return render_template("register.html", error=error)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

# ─── History routes ───────────────────────────────────────────────────────────
@app.route("/history")
@login_required
def history():
    records = get_history(current_user.username)
    return render_template("history.html", records=records)

@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    clear_user_history(current_user.username)
    return redirect(url_for("history"))

# ─── Main routes ─────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender          = request.form["gender"]
        age             = float(request.form["age"])
        hypertension    = int(request.form["hypertension"])
        heart_disease   = int(request.form["heart_disease"])
        smoking_history = request.form["smoking_history"]
        bmi             = float(request.form["bmi"])
        hba1c           = float(request.form["HbA1c_level"])
        glucose         = float(request.form["blood_glucose_level"])

        gender_enc  = le_gen.transform([gender])[0]
        smoking_enc = le_smoke.transform([smoking_history])[0]

        raw        = np.array([[gender_enc, age, hypertension, heart_disease,
                                smoking_enc, bmi, hba1c, glucose]])
        raw_scaled = scaler.transform(raw)
        prediction = model.predict(raw_scaled)
        probability = round(float(prediction[0][0]) * 100, 1)

        result_type = "diabetic" if probability >= 50 else "not_diabetic"
        result      = ("⚠️ High Diabetes Risk Detected"
                       if probability >= 50 else "✅ Low Diabetes Risk")

        # Save to MongoDB if logged in
        if current_user.is_authenticated:
            add_to_history(
                username    = current_user.username,
                inputs      = {
                    "Gender":          gender,
                    "Age":             age,
                    "Hypertension":    "Yes" if hypertension else "No",
                    "Heart Disease":   "Yes" if heart_disease else "No",
                    "Smoking History": smoking_history,
                    "BMI":             bmi,
                    "HbA1c Level":     hba1c,
                    "Blood Glucose":   glucose,
                },
                result      = result,
                result_type = result_type,
                probability = probability,
            )

        return render_template("index.html",
                               prediction_text=result,
                               result_type=result_type,
                               probability=probability)

    except KeyError as e:
        return render_template("index.html",
                               prediction_text=f"❌ Missing field: {e}",
                               result_type="error"), 400
    except ValueError as e:
        return render_template("index.html",
                               prediction_text=f"❌ Invalid value: {e}",
                               result_type="error"), 400
    except Exception:
        return render_template("index.html",
                               prediction_text="❌ An unexpected error occurred.",
                               result_type="error"), 500

if __name__ == "__main__":
    app.run(debug=True)