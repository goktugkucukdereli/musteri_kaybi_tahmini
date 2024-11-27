import joblib
import pandas as pd

# Kaydedilen modeli yükleme
final_model = joblib.load("final_model_rf.pkl")
print("Final model yüklendi.")

# Yeni veri setini yükleme
new_data = pd.read_csv("data/new_data.csv")

# Gerekli ön işleme
# 1. 'customerID' sütununu kaldırma
new_data.drop(columns=["customerID"], inplace=True)

# 2. Modelde kullanılan aynı ön işlemleri uygulama
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                       'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod']

# One-Hot Encoding ile kategorik özellikleri dönüştürme
new_data_encoded = pd.get_dummies(new_data, columns=categorical_columns)

# Eğitim sırasında kullanılan tüm özellikleri garantiye almak için sütunları eşleştirme
model_features = final_model.feature_names_in_  # Eğitimde kullanılan özellikler
for col in model_features:
    if col not in new_data_encoded.columns:
        new_data_encoded[col] = 0  # Eksik özellikleri 0 ile doldurma

# Ekstra sütunları kaldırma
new_data_encoded = new_data_encoded[model_features]

# 3. Tahmin yapma
predictions = final_model.predict(new_data_encoded)

# Tahmin sonuçlarını yeni bir sütun olarak ekleme
new_data["Churn_Prediction"] = predictions

# Tahmin sonuçlarını kaydetme
new_data.to_csv("data/new_data_with_predictions.csv", index=False)
print("Tahmin sonuçları 'data/new_data_with_predictions.csv' dosyasına kaydedildi.")
