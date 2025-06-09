import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Загрузка своего CSV
df = pd.read_csv("fraud_dataset_real.csv")
df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
df['IsNight'] = df['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

# Кодировка
le_region = LabelEncoder()
le_device = LabelEncoder()
df['Region'] = le_region.fit_transform(df['Region'])
df['DeviceType'] = le_device.fit_transform(df['DeviceType'])

X = df[['Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight']]
y = df['Fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Сохраняем локально, ПОД ТВОЮ ВЕРСИЮ sklearn
joblib.dump(model, "model/real_model.pkl")
joblib.dump(le_region, "model/le_region.pkl")
joblib.dump(le_device, "model/le_device.pkl")

print("✅ Модель и энкодеры обучены и сохранены локально.")
