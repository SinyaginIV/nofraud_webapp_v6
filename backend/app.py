from flask import Flask, request, jsonify, send_file
from flask import send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return send_from_directory('../frontend', 'index.html')

MODEL_DIR = 'model'
PROCESSED_DIR = 'processed'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Загружаем модель и энкодеры только при запросе
        model_test = joblib.load(f'{MODEL_DIR}/real_model.pkl')
        le_region = joblib.load(f'{MODEL_DIR}/le_region.pkl')
        le_device = joblib.load(f'{MODEL_DIR}/le_device.pkl')

        file = request.files['file']
        model_type = request.form.get('modelType', 'TEST')

        df = pd.read_csv(file)
        df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour

        if 'PredictedFraud' in df.columns:
            df = df.drop(columns=['PredictedFraud'])

        # Преобразуем признаки, если они строковые
        try:
            if df['Region'].dtype == 'object':
                df['Region'] = le_region.transform(df['Region'])
            if df['DeviceType'].dtype == 'object':
                df['DeviceType'] = le_device.transform(df['DeviceType'])
        except Exception as e:
            return jsonify({'error': f'Ошибка энкодинга: {str(e)}'}), 400

        features = ['Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight', 'Hour']
        if not all(col in df.columns for col in features):
            return jsonify({'error': 'Некорректные признаки в файле.'}), 400

        X = df[features]

        # Выбор модели
        if model_type == 'TEST':
            model = model_test
        else:
            model_path = f'{MODEL_DIR}/new_model.pkl'
            if not os.path.exists(model_path):
                return jsonify({'error': 'Новая модель не обучена.'}), 400
            model = joblib.load(model_path)

        preds = model.predict(X)
        df['PredictedFraud'] = preds

        def explain(row):
            reasons = []
            if row['Amount'] > 4000:
                reasons.append("Сумма > 4000")
            if row['IsNight'] == 1:
                reasons.append("Ночная активность")
            if row['IsAbroad'] == 1:
                reasons.append("Заграничная транзакция")
            if row['TxCountLastHour'] > 5:
                reasons.append("Много транзакций в течение часа")
            return ", ".join(reasons)

        df['Explanation'] = df.apply(explain, axis=1)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f'{PROCESSED_DIR}/result_{timestamp}.csv'
        df.to_csv(output_file, index=False)

        return jsonify({
            'columns': df.columns.tolist(),
            'rows': df.values.tolist()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        # Загружаем энкодеры при запросе
        le_region = joblib.load(f'{MODEL_DIR}/le_region.pkl')
        le_device = joblib.load(f'{MODEL_DIR}/le_device.pkl')

        file = request.files['file']
        df = pd.read_csv(file)

        df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour

        try:
            if df['Region'].dtype == 'object':
                df['Region'] = le_region.transform(df['Region'])
            if df['DeviceType'].dtype == 'object':
                df['DeviceType'] = le_device.transform(df['DeviceType'])
        except Exception as e:
            return jsonify({'error': f'Ошибка энкодинга: {str(e)}'}), 500

        features = ['Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight', 'Hour']
        if not all(col in df.columns for col in features + ['PredictedFraud']):
            return jsonify({'error': 'Недостаточно признаков или отсутствует столбец PredictedFraud.'}), 400

        X = df[features]
        y = df['PredictedFraud']

        model_new = RandomForestClassifier(n_estimators=100, random_state=42)
        model_new.fit(X, y)

        # Сохраняем модель
        joblib.dump(model_new, f'{MODEL_DIR}/new_model.pkl')

        return jsonify({'message': 'Новая модель успешно обучена.'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ошибка обучения: {str(e)}'}), 500

@app.route('/download', methods=['GET'])
def download():
    files = sorted(Path(PROCESSED_DIR).glob('result_*.csv'), key=os.path.getmtime, reverse=True)
    if not files:
        return jsonify({'error': 'Файл не найден.'}), 404
    return send_file(files[0], as_attachment=True)
