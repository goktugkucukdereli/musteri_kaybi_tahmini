# Gerekli kütüphaneleri yükleme
import pandas as pd

# Veri seti yolunu belirtelim
data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Veri setini yükleyelim
df = pd.read_csv(data_path)

# Sütun türlerini dönüştürme
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Sayısal dönüştürme
print(f"TotalCharges sütunundaki eksik değer sayısı: {df['TotalCharges'].isnull().sum()}")

# Eksik değerleri doldurma (inplace=False ile güncel yöntem)
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

# Veri setinin son hali
print("\nİşlenmiş Veri Seti:")
print(df.head())
