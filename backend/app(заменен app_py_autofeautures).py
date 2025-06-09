from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import os
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')  # отключает GUI и tkinter
import matplotlib.pyplot as plt
from fpdf.enums import XPos, YPos
from datetime import datetime

app = Flask(__name__)
CORS(app)

model = joblib.load('model/real_model.pkl')

@app.route('/upload', methods=['POST'])
def upload():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    file = request.files['file']
    df = pd.read_csv(file)

    le_region = joblib.load('model/le_region.pkl')
    le_device = joblib.load('model/le_device.pkl')

    df['Region'] = le_region.transform(df['Region'])
    df['DeviceType'] = le_device.transform(df['DeviceType'])
    df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
    df['IsNight'] = df['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

    features = ['Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight']
    X = df[features]
    y_pred = model.predict(X)
    df['PredictedFraud'] = y_pred

    def explain(row):
        reasons = []
        if row['Amount'] > 4000: reasons.append("Сумма > 4000")
        if row['IsAbroad'] == 1: reasons.append("Зарубежная транзакция")
        if row['IsNight'] == 1: reasons.append("Ночная активность")
        if row['TxCountLastHour'] > 5: reasons.append("Много транзакций в час")
        return ", ".join(reasons) if row['PredictedFraud'] == 1 else ""

    df['Explanation'] = df.apply(explain, axis=1)

    os.makedirs('processed', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    csv_filename = f"processed/result_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)

    # Графики
    plt.figure(figsize=(6, 4))
    df[df['PredictedFraud'] == 1]['Amount'].hist(color='red', bins=20)
    plt.title("Суммы фродовых транзакций")
    plt.xlabel("Сумма")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig('static/fraud_amount.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    df[df['PredictedFraud'] == 1]['Hour'].hist(color='orange', bins=24)
    plt.title("Фрод по часам суток")
    plt.xlabel("Час")
    plt.ylabel("Фродов")
    plt.tight_layout()
    plt.savefig('static/fraud_by_hour.png')
    plt.close()

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf")
    pdf.set_font("DejaVu", size=14)
    pdf.cell(200, 10, text="Отчёт о транзакциях NoFraud", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", size=12)
    pdf.cell(200, 10, text=f"Всего операций: {len(df)}, мошеннических: {df['PredictedFraud'].sum()}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.image("static/fraud_amount.png", w=150)
    pdf.ln(10)
    pdf.image("static/fraud_by_hour.png", w=150)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_filename = f"reports/report_{timestamp}.pdf"
    pdf.output(pdf_filename)

    # Возвращаем данные для HTML
    preview_columns = ['DateTime', 'Amount', 'Region', 'DeviceType',
                       'IsAbroad', 'TxCountLastHour', 'IsNight',
                       'PredictedFraud', 'Explanation']
    rows = df[preview_columns].values.tolist()
    return jsonify({
        'columns': preview_columns,
        'rows': rows
    })

@app.route('/download', methods=['GET'])
def download_csv():
    return send_file('processed/result.csv', as_attachment=True)

@app.route('/report', methods=['GET'])
def download_pdf():
    return send_file('reports/report.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
