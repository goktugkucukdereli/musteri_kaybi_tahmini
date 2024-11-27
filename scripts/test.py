# Gerekli kütüphaneleri yükleme
import pandas as pd

# Veri seti yolunu belirtelim
data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Veri setini yükleyelim
df = pd.read_csv(data_path)

# Tekrar One-Hot Encoding
df = pd.get_dummies(df, columns=['Churn'], drop_first=True)

# Sütunları kontrol et
print(df.columns)
