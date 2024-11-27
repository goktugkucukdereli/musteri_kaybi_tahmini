# Gerekli kütüphaneleri yükleyelim
import pandas as pd

# Veri seti yolunu belirtelim
data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Veri setini yükleme
df = pd.read_csv(data_path)

# Veri setinin ilk 5 satırını inceleyelim
print("Veri Setinin İlk 5 Satırı:")
print(df.head())

# Veri setinin temel bilgileri
print("\nVeri Seti Bilgisi:")
print(df.info())

# Veri setindeki eksik değerleri kontrol etme
print("\nEksik Değerlerin Sayısı:")
print(df.isnull().sum())

# Kategorik ve sayısal sütunları ayırma
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['number']).columns

print("\nKategorik Sütunlar:")
print(categorical_columns)
print("\nSayısal Sütunlar:")
print(numerical_columns)
