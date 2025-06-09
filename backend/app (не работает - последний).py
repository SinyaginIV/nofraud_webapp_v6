
import os
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
CORS(app)

MODEL_DIR = "model"
TEST_MODEL_PATH = os.path.join(MODEL_DIR, "real_model.pkl")
NEW_MODEL_PATH = os.path.join(MODEL_DIR, "new_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "new_model_features.json")
ENCODERS_PATH = os.path.join(MODEL_DIR, "new_model_encoders.pkl")

def preprocess_for_prediction(df, feature_list, encoders):
    df = df.copy()
    if 'DateTime' in df.columns:
        df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
        df['IsNight'] = df['Hour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)

    for col in df.columns:
        if df[col].dtype == object and col in encoders:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    missing = set(feature_list) - set(df.columns)
    for col in missing:
        df[col] = 0  # по умолчанию
    return df[feature_list]

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    model_type = request.form.get("modelType", "test")

    if not file:
        return jsonify({"error": "Файл не найден"}), 400

    try:
        df = pd.read_csv(file)

        if model_type == "test":
            model_path = TEST_MODEL_PATH
            features = ["Amount", "IsAbroad", "TxCountLastHour", "IsNight"]
            df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
            df['IsNight'] = df['Hour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)
            X = df[features]
            model = joblib.load(model_path)
        else:
            if not os.path.exists(NEW_MODEL_PATH):
                return jsonify({"error": "Модель не найдена. Обучите её сначала."}), 400
            model = joblib.load(NEW_MODEL_PATH)
            features = json.load(open(FEATURES_PATH))
            encoders = joblib.load(ENCODERS_PATH)
            X = preprocess_for_prediction(df, features, encoders)

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
        if 'PredictedFraud' not in df.columns:
            return jsonify({"error": "В файле отсутствует столбец PredictedFraud"}), 400

        if 'DateTime' in df.columns:
            df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
            df['IsNight'] = df['Hour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)

        y = df['PredictedFraud']
        X = df.drop(columns=['PredictedFraud'])

        encoders = {}
        for col in X.columns:
            if X[col].dtype == object:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, NEW_MODEL_PATH)
        json.dump(list(X.columns), open(FEATURES_PATH, "w"))
        joblib.dump(encoders, ENCODERS_PATH)

        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        return jsonify({"message": "Обучение завершено успешно.", "score": report["accuracy"]})
    except Exception as e:
        return jsonify({"error": f"Ошибка обучения: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
