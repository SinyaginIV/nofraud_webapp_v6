import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Загрузка датасета
df = pd.read_csv('dataset_with_hour_labeled.csv')

# Загрузка энкодеров
le_region = joblib.load('model/le_region.pkl')
le_device = joblib.load('model/le_device.pkl')

# Преобразуем строковые признаки
if df['Region'].dtype == 'object':
    df['Region'] = le_region.transform(df['Region'])
if df['DeviceType'].dtype == 'object':
    df['DeviceType'] = le_device.transform(df['DeviceType'])

# Обучающие данные и целевая переменная
features = ['Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight', 'Hour']
X = df[features]
y = df['PredictedFraud']

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Сохраняем модель
joblib.dump(model, 'model/real_model.pkl')
print("Модель переобучена и сохранена.")
