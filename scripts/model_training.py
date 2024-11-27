# Gerekli kütüphaneleri yükleme
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Veri seti yolunu belirtelim
data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Veri setini yükleyelim
df = pd.read_csv(data_path)

# TotalCharges sütununu sayısal tipe dönüştürme
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Eksik değerleri doldurma
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Kategorik sütunları kodlama (One-Hot Encoding)
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                       'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod', 'Churn']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Gereksiz sütunları kaldırma
df.drop(['customerID'], axis=1, inplace=True)

# Hedef değişkeni (y) ve bağımsız değişkenleri (X) ayırma
X = df.drop(columns=['Churn_Yes'])  # 'Churn_Yes' hedef değişken
y = df['Churn_Yes']

# Veriyi eğitim ve test setlerine ayırma (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression, Random Forest, XGBoost, LightGBM modeli oluşturma ve eğitme (iterasyon sayısı artırılmış)
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train_scaled, y_train)

# Tahminler yapma
y_pred_lgbm = lgbm_model.predict(X_test_scaled)

# Model performansını değerlendirme
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
conf_matrix_lgbm = confusion_matrix(y_test, y_pred_lgbm)
class_report_lgbm = classification_report(y_test, y_pred_lgbm)

# Sonuçları yazdırma
print(f"LightGBM Model Doğruluk (Accuracy): {accuracy_lgbm:.2f}")
print("\nKarışıklık Matrisi (Confusion Matrix):")
print(conf_matrix_lgbm)
print("\nSınıflandırma Raporu (Classification Report):")
print(class_report_lgbm)