# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import os
from datetime import datetime
from sqlalchemy import create_engine, text

# Configuración DB - ajusta user, password, host, port, dbname según XAMPP
DB_USER = "root"
DB_PASSWORD = ""   # si dejaste vacío en XAMPP
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "shieldnet"

DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

engine = create_engine(DATABASE_URI, pool_recycle=3600)

app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_PATH = "model_shieldnet.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No se encontró el modelo. Ejecuta train_model.py primero.")

model = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    text_input = data.get("text") or data.get("text_input") or ""
    if not text_input:
        return jsonify({"error": "text field is required"}), 400

    # Predicción
    probs = model.predict_proba([text_input])[0]
    classes = model.classes_
    # encontrar label y confidence
    max_idx = probs.argmax()
    label = classes[max_idx]
    confidence = float(probs[max_idx])

    # Guardar en MySQL
    try:
        with engine.connect() as conn:
            stmt = text("INSERT INTO analyses (text_input, label, confidence, created_at) VALUES (:t, :l, :c, :d)")
            conn.execute(stmt, {"t": text_input, "l": label, "c": confidence, "d": datetime.now()})
            conn.commit()
    except Exception as e:
        # no bloquear la respuesta si falla la DB, pero avisamos
        return jsonify({"error": "DB error", "detail": str(e)}), 500

    return jsonify({
        "text": text_input,
        "label": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
