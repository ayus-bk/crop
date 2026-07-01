from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import json

from database import db
from models import User, FarmerProfile, Recommendation
import os

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(BASE, "instance", "database.db")

app.config['SECRET_KEY'] = 'crop_project_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DATABASE_PATH}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

print("Database Path:", DATABASE_PATH)
print("Database URI:", app.config['SQLALCHEMY_DATABASE_URI'])


# ── Load & Train ─────────────────────────────────────────────────────────────
print("🌱 Loading dataset and training model...")
df = pd.read_csv(os.path.join(BASE, 'Crop_recommendation.csv'))

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred      = model.predict(X_test)
accuracy    = accuracy_score(y_test, y_pred)
crop_count  = df['label'].nunique()
all_crops   = sorted(df['label'].unique().tolist())

# Per-crop stats from the real dataset
crop_stats = {}
for crop in all_crops:
    sub = df[df['label'] == crop]
    crop_stats[crop] = {
        'count': int(len(sub)),
        'N_mean':    round(float(sub['N'].mean()), 1),
        'P_mean':    round(float(sub['P'].mean()), 1),
        'K_mean':    round(float(sub['K'].mean()), 1),
        'temp_mean': round(float(sub['temperature'].mean()), 1),
        'hum_mean':  round(float(sub['humidity'].mean()), 1),
        'ph_mean':   round(float(sub['ph'].mean()), 2),
        'rain_mean': round(float(sub['rainfall'].mean()), 1),
    }

print(f"✅ Model ready! Accuracy: {accuracy:.4f} | Crops: {crop_count}")

# ── Crop metadata ─────────────────────────────────────────────────────────────
CROP_META = {
    'rice':        {'emoji':'🌾','season':'Kharif','water':'High',    'color':'#22d3ee','bg':'#164e63'},
    'maize':       {'emoji':'🌽','season':'Kharif','water':'Medium',  'color':'#facc15','bg':'#713f12'},
    'chickpea':    {'emoji':'🫘','season':'Rabi',  'water':'Low',     'color':'#fb923c','bg':'#7c2d12'},
    'kidneybeans': {'emoji':'🫘','season':'Kharif','water':'Medium',  'color':'#f87171','bg':'#7f1d1d'},
    'pigeonpeas':  {'emoji':'🫘','season':'Kharif','water':'Low',     'color':'#c084fc','bg':'#4a1d96'},
    'mothbeans':   {'emoji':'🫘','season':'Kharif','water':'Very Low','color':'#a78bfa','bg':'#3b0764'},
    'mungbean':    {'emoji':'🫘','season':'Zaid',  'water':'Low',     'color':'#86efac','bg':'#14532d'},
    'blackgram':   {'emoji':'🫘','season':'Kharif','water':'Low',     'color':'#94a3b8','bg':'#1e293b'},
    'lentil':      {'emoji':'🫘','season':'Rabi',  'water':'Low',     'color':'#fcd34d','bg':'#78350f'},
    'pomegranate': {'emoji':'🍎','season':'Annual','water':'Low',     'color':'#fb7185','bg':'#881337'},
    'banana':      {'emoji':'🍌','season':'Annual','water':'High',    'color':'#fde047','bg':'#713f12'},
    'mango':       {'emoji':'🥭','season':'Summer','water':'Low',     'color':'#fb923c','bg':'#7c2d12'},
    'grapes':      {'emoji':'🍇','season':'Annual','water':'Medium',  'color':'#a78bfa','bg':'#4c1d95'},
    'watermelon':  {'emoji':'🍉','season':'Zaid',  'water':'Medium',  'color':'#4ade80','bg':'#14532d'},
    'muskmelon':   {'emoji':'🍈','season':'Zaid',  'water':'Medium',  'color':'#a3e635','bg':'#365314'},
    'apple':       {'emoji':'🍏','season':'Annual','water':'Medium',  'color':'#86efac','bg':'#052e16'},
    'orange':      {'emoji':'🍊','season':'Winter','water':'Medium',  'color':'#f97316','bg':'#7c2d12'},
    'papaya':      {'emoji':'🍑','season':'Annual','water':'Medium',  'color':'#fdba74','bg':'#7c2d12'},
    'coconut':     {'emoji':'🥥','season':'Annual','water':'High',    'color':'#d8b4fe','bg':'#4a1d96'},
    'cotton':      {'emoji':'🪡','season':'Kharif','water':'Medium',  'color':'#e2e8f0','bg':'#1e293b'},
    'jute':        {'emoji':'🌿','season':'Kharif','water':'High',    'color':'#bef264','bg':'#365314'},
    'coffee':      {'emoji':'☕','season':'Annual','water':'Medium',  'color':'#d97706','bg':'#451a03'},
}

def get_soil_health(N, P, K, ph):
    score, tips = 0, []
    # Nitrogen
    if   40 <= N <= 100: score += 25
    elif N < 40:  tips.append({"icon":"⬇️","msg":"Low nitrogen (N) — consider urea or compost","param":"N"})
    else:         tips.append({"icon":"⬆️","msg":"High nitrogen (N) — risk of leaf burn","param":"N"})
    # Phosphorus
    if   20 <= P <= 80: score += 25
    elif P < 20:  tips.append({"icon":"⬇️","msg":"Low phosphorus (P) — add superphosphate","param":"P"})
    else:         tips.append({"icon":"⬆️","msg":"Excess phosphorus (P) — may block zinc uptake","param":"P"})
    # Potassium
    if   20 <= K <= 80: score += 25
    elif K < 20:  tips.append({"icon":"⬇️","msg":"Low potassium (K) — apply muriate of potash","param":"K"})
    else:         tips.append({"icon":"⬆️","msg":"High potassium (K) — reduce fertilizer input","param":"K"})
    # pH
    if   5.5 <= ph <= 7.5: score += 25
    elif ph < 5.5: tips.append({"icon":"🧪","msg":f"Acidic soil (pH {ph:.1f}) — apply agricultural lime","param":"ph"})
    else:          tips.append({"icon":"🧪","msg":f"Alkaline soil (pH {ph:.1f}) — add elemental sulphur","param":"ph"})
    return score, tips

def get_alternatives(vals, top_n=5):
    feat = pd.DataFrame([vals], columns=['N','P','K','temperature','humidity','ph','rainfall'])
    leaf = model.apply(feat)[0]
    mask = model.apply(X_train) == leaf
    if mask.sum() == 0:
        return []
    counts = y_train[mask].value_counts()
    total  = counts.sum()
    return [{'crop': c, 'pct': round(n/total*100)} for c, n in counts.items()][:top_n]

def get_ideal_comparison(crop, vals):
    """Compare user inputs vs ideal values for that crop from real dataset"""
    if crop not in crop_stats:
        return []
    ideal = crop_stats[crop]
    params = [
        {'name':'N',           'unit':'mg/kg','user':round(vals[0],1), 'ideal':ideal['N_mean']},
        {'name':'P',           'unit':'mg/kg','user':round(vals[1],1), 'ideal':ideal['P_mean']},
        {'name':'K',           'unit':'mg/kg','user':round(vals[2],1), 'ideal':ideal['K_mean']},
        {'name':'Temperature', 'unit':'°C',   'user':round(vals[3],1), 'ideal':ideal['temp_mean']},
        {'name':'Humidity',    'unit':'%',    'user':round(vals[4],1), 'ideal':ideal['hum_mean']},
        {'name':'pH',          'unit':'',     'user':round(vals[5],2), 'ideal':ideal['ph_mean']},
        {'name':'Rainfall',    'unit':'mm',   'user':round(vals[6],1), 'ideal':ideal['rain_mean']},
    ]
    for p in params:
        diff = p['user'] - p['ideal']
        p['diff']   = round(diff, 2)
        p['status'] = 'good' if abs(diff) / (p['ideal'] or 1) < 0.15 else ('high' if diff > 0 else 'low')
    return params

# ── Routes ────────────────────────────────────────────────────────────────────
# ==========================
# Page Routes
# ==========================

@app.route("/")
def home():
    return render_template("homepage.html")


@app.route("/about")
def about():
    return render_template("aboutpage.html")


@app.route('/register', methods=['GET', 'POST'])
def register():

    # Show registration page
    if request.method == 'GET':
        return render_template("registerpage.html")

    # Receive form data
    fullname = request.form['fullname']
    username = request.form['username']
    email = request.form['email']
    phone = request.form['phone']
    district = request.form['district']
    municipality = request.form['municipality']
    ward = request.form['ward']
    farm_size = request.form['farm_size']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    # Password check
    if password != confirm_password:
        return "Passwords do not match!"

    # Check existing username
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return "Username already exists!"

    # Check existing email
    existing_email = User.query.filter_by(email=email).first()
    if existing_email:
        return "Email already registered!"

    # Create user
    new_user = User(
        full_name=fullname,
        username=username,
        email=email,
        password=password,      # We'll hash this later
        role="farmer"
    )

    db.session.add(new_user)
    db.session.commit()

    # Create farmer profile
    profile = FarmerProfile(
        user_id=new_user.id,
        phone=phone,
        district=district,
        municipality=municipality,
        ward=int(ward),
        farm_size=float(farm_size)
    )

    db.session.add(profile)
    db.session.commit()
    return redirect(url_for("login"))

    # return "Registration Successful!"


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "GET":
        return render_template("loginpage.html")

    username = request.form["username"]
    password = request.form["password"]

    # Search by username OR email
    user = User.query.filter(
        (User.username == username) |
        (User.email == username)
    ).first()

    if user is None:
        return "Invalid username or email."

    # For now (plain-text password)
    if user.password != password:
        return "Invalid password."

    # Store login session
    session["user_id"] = user.id
    session["username"] = user.username
    session["role"] = user.role

    # Redirect according to role
    if user.role == "admin":
        return redirect(url_for("admin_dashboard"))

    return redirect(url_for("farmer_dashboard"))



@app.route("/farmer/dashboard")
def farmer_dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session["role"] != "farmer":
        return "Access Denied"

    return render_template(
        "farmer_dashboard.html",
        username=session["username"]
    )



@app.route("/admin/dashboard")
def admin_dashboard():

    # Must be logged in
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Only admin
    if session["role"] != "admin":
        return "Access Denied"

    # Dashboard Statistics
    total_farmers = User.query.filter_by(role="farmer").count()

    total_admins = User.query.filter_by(role="admin").count()

    total_predictions = Recommendation.query.count()

    latest_farmer = User.query.filter_by(role="farmer")\
                              .order_by(User.created_at.desc())\
                              .first()

    return render_template(
        "admin_dashboard.html",
        username=session["username"],
        total_farmers=total_farmers,
        total_admins=total_admins,
        total_predictions=total_predictions,
        latest_farmer=latest_farmer
    )


@app.route("/admin/farmers")
def admin_farmers():

    # Check login
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Check admin
    if session["role"] != "admin":
        return "Access Denied"

    farmers = User.query.filter_by(role="farmer").all()

    return render_template(
        "farmers.html",
        farmers=farmers
    )



@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("home"))



@app.route("/predict-page")
def predict_page():
    return render_template(
        "predict.html",
        accuracy=round(accuracy * 100, 2),
        crop_count=crop_count,
        crops=all_crops,
        total_samples=len(df)
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        vals = [float(data[k]) for k in ['N','P','K','temperature','humidity','ph','rainfall']]
        inp  = dict(zip(['N','P','K','temperature','humidity','ph','rainfall'], vals))
        feat = pd.DataFrame([vals], columns=['N','P','K','temperature','humidity','ph','rainfall'])
        crop = model.predict(feat)[0]

        meta       = CROP_META.get(crop, {'emoji':'🌱','season':'Annual','water':'Medium','color':'#4ade80','bg':'#14532d'})
        soil_score, soil_tips = get_soil_health(vals[0], vals[1], vals[2], vals[5])
        alternatives          = get_alternatives(vals)
        comparison            = get_ideal_comparison(crop, vals)

        return jsonify({
            'crop':           crop,
            'emoji':          meta['emoji'],
            'color':          meta['color'],
            'bg':             meta['bg'],
            'season':         meta['season'],
            'water':          meta['water'],
            'soil_score':     soil_score,
            'soil_tips':      soil_tips,
            'alternatives':   alternatives,
            'comparison':     comparison,
            'crop_stats':     crop_stats.get(crop, {}),
            'model_accuracy': round(accuracy*100, 2),
            'inputs':         inp
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dataset-stats')
def dataset_stats():
    result = {}
    for col in ['N','P','K','temperature','humidity','ph','rainfall']:
        result[col] = {
            'min':  round(float(df[col].min()), 2),
            'max':  round(float(df[col].max()), 2),
            'mean': round(float(df[col].mean()), 2),
            'std':  round(float(df[col].std()), 2),
        }
    return jsonify({'stats': result, 'crops': all_crops,
                    'accuracy': round(accuracy*100,2), 'total': len(df)})



if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

    app.run(debug=True, port=5050)
