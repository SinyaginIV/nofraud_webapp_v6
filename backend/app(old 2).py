
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
from preprocessing import preprocess_file
from fpdf import FPDF
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

model = joblib.load('model/real_model.pkl')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    df = pd.read_csv(file)
    processed = preprocess_file(df)
    pred = model.predict(processed)
    df['PredictedFraud'] = pred

    os.makedirs('processed', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    df.to_csv('processed/result.csv', index=False)

    # График
    plt.figure(figsize=(6, 4))
    df[df['PredictedFraud'] == 1]['Amount'].hist(color='red', bins=20)
    plt.title("Суммы мошеннических операций")
    plt.xlabel("Сумма")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.savefig('reports/amount_fraud.png')
    plt.close()

    # PDF с поддержкой кириллицы через fpdf2 и шрифт
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=14)
    pdf.cell(200, 10, txt="Отчёт о транзакциях NoFraud", ln=True, align='C')
    pdf.ln(10)
    total = len(df)
    frauds = df['PredictedFraud'].sum()
    pdf.set_font("DejaVu", size=12)
    pdf.cell(200, 10, txt=f"Всего операций: {total}, мошеннических: {frauds}", ln=True)
    pdf.image("reports/amount_fraud.png", w=150)
    pdf.output("reports/report.pdf")

    return jsonify({'status': 'ok'})

@app.route('/download', methods=['GET'])
def download_csv():
    return send_file('processed/result.csv', as_attachment=True)

@app.route('/report', methods=['GET'])
def download_pdf():
    return send_file('reports/report.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
