from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os, json

app = Flask(__name__)

BASE = os.path.dirname(__file__)

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


@app.route("/register")
def register():
    return render_template("registerpage.html")


@app.route("/login")
def login():
    return render_template("loginpage.html")


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

if __name__ == '__main__':
    app.run(debug=True, port=5050)
