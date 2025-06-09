
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

app = Flask(__name__)
CORS(app)

MODEL_DIR = "model"
TEST_MODEL_PATH = os.path.join(MODEL_DIR, "real_model.pkl")
NEW_MODEL_PATH = os.path.join(MODEL_DIR, "new_model.pkl")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    model_type = request.form.get("modelType", "test")

    if not file:
        return jsonify({"error": "Файл не найден"}), 400

    try:
        df = pd.read_csv(file)
        required = ["DateTime", "Amount", "Region", "DeviceType", "IsAbroad", "TxCountLastHour"]
        if not all(col in df.columns for col in required):
            return jsonify({"error": "Неверная структура CSV"}), 400

        df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
        df['IsNight'] = df['Hour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)
        X = df[["Amount", "IsAbroad", "TxCountLastHour", "IsNight"]]

        model_path = TEST_MODEL_PATH if model_type == "test" else NEW_MODEL_PATH
        if not os.path.exists(model_path):
            return jsonify({"error": "Модель не найдена. Сначала обучите модель."}), 400

        model = joblib.load(model_path)
        preds = model.predict(X)

        df['PredictedFraud'] = preds

        return jsonify({
            "columns": df.columns.tolist(),
            "rows": df.values.tolist()
        })
    except Exception as e:
        return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 500

@app.route("/train", methods=["POST"])
def train():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Файл не найден"}), 400

    try:
        df = pd.read_csv(file)
        required = ["Amount", "IsAbroad", "TxCountLastHour", "DateTime", "PredictedFraud"]
        if not all(col in df.columns for col in required):
            return jsonify({"error": "В файле должны быть колонки: Amount, IsAbroad, TxCountLastHour, DateTime, PredictedFraud"}), 400

        df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
        df['IsNight'] = df['Hour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)

        X = df[["Amount", "IsAbroad", "TxCountLastHour", "IsNight"]]
        y = df["PredictedFraud"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, NEW_MODEL_PATH)

        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        return jsonify({"message": "Обучение завершено успешно.", "score": report["accuracy"]})
    except Exception as e:
        return jsonify({"error": f"Ошибка обучения: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
