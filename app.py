from flask import Flask, request, jsonify, render_template, redirect, url_for, session, make_response
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

from database import db
from models import User, FarmerProfile, Recommendation

# ===========================
# APP SETUP
# ===========================
BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["SECRET_KEY"] = "crop_project_2026"

# SESSION CONFIG (IMPORTANT FIX)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)

# Database
os.makedirs(app.instance_path, exist_ok=True)
DATABASE_PATH = os.path.join(app.instance_path, "database.db")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + DATABASE_PATH
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# ===========================
# LOAD DATASET + TRAIN MODEL
# ===========================
csv_path = os.path.join(BASE, "Crop_recommendation.csv")
df = pd.read_csv(csv_path)

X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

all_crops = sorted(df['label'].unique().tolist())
crop_count = len(all_crops)

# ===========================
# SESSION CONTROL (IMPORTANT FIX)
# ===========================
@app.before_request
def session_control():
    session.permanent = False  # default OFF unless remember me enabled


# ===========================
# ROUTES
# ===========================
@app.route("/")
def home():
    return render_template("homepage.html")


@app.route("/about")
def about():
    return render_template("aboutpage.html")


# ===========================
# REGISTER
# ===========================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template("registerpage.html")

    user = User(
        full_name=request.form['fullname'],
        username=request.form['username'],
        email=request.form['email'],
        password=request.form['password'],
        role="farmer"
    )

    db.session.add(user)
    db.session.commit()

    profile = FarmerProfile(
        user_id=user.id,
        phone=request.form['phone'],
        district=request.form['district'],
        municipality=request.form['municipality'],
        ward=int(request.form['ward']),
        farm_size=float(request.form['farm_size'])
    )

    db.session.add(profile)
    db.session.commit()

    return redirect(url_for("login"))


# ===========================
# LOGIN (FIXED REMEMBER ME)
# ===========================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "GET":
        # Prevent browser caching (fix autofill/back button issues)
        response = make_response(render_template("loginpage.html"))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    username = request.form["username"]
    password = request.form["password"]
    remember = request.form.get("remember")

    user = User.query.filter(
        (User.username == username) | (User.email == username)
    ).first()

    if not user or user.password != password:
        return "Invalid credentials"

    # clear old session first (VERY IMPORTANT)
    session.clear()

    session["user_id"] = user.id
    session["username"] = user.username
    session["role"] = user.role

    # REMEMBER ME FIX
    if remember:
        session.permanent = True   # lasts longer (30 min configured)
    else:
        session.permanent = False  # ends when browser closes

    if user.role == "admin":
        return redirect(url_for("admin_dashboard"))

    return redirect(url_for("farmer_dashboard"))


# ===========================
# DASHBOARDS
# ===========================
@app.route("/farmer/dashboard")
def farmer_dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "farmer":
        return "Access Denied"

    return render_template("farmer_dashboard.html", username=session["username"])


@app.route("/admin/dashboard")
def admin_dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        return "Access Denied"

    return render_template("admin_dashboard.html", username=session["username"])


# ===========================
# LOGOUT
# ===========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ===========================
# PREDICT PAGE (LOGIN REQUIRED FIX)
# ===========================
@app.route("/predict-page")
def predict_page():
    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "predict.html",
        accuracy=round(accuracy * 100, 2),
        crop_count=crop_count,
        crops=all_crops,
        total_samples=len(df),
        session=session
    )


# ===========================
# PREDICT API (SECURED)
# ===========================
@app.route('/predict', methods=['POST'])
def predict():

    if "user_id" not in session:
        return jsonify({"error": "Login required"}), 401

    data = request.get_json()

    vals = [
        float(data['N']),
        float(data['P']),
        float(data['K']),
        float(data['temperature']),
        float(data['humidity']),
        float(data['ph']),
        float(data['rainfall'])
    ]

    feat = pd.DataFrame([vals], columns=X.columns)
    crop = model.predict(feat)[0]

    return jsonify({
        "crop": crop,
        "accuracy": round(accuracy * 100, 2),
        "inputs": data
    })


# ===========================
# STATS API
# ===========================
@app.route('/dataset-stats')
def dataset_stats():
    result = {}

    for col in X.columns:
        result[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean())
        }

    return jsonify({
        "stats": result,
        "crops": all_crops,
        "accuracy": round(accuracy * 100, 2)
    })


# ===========================
# RUN APP
# ===========================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True, port=5050)