# Performans metriklerini bir CSV dosyasına kaydetme
import pandas as pd

# Modellerin performans metrikleri
data = {
    "Model": ["Logistic Regression", "Random Forest (SMOTE)", "XGBoost (SMOTE)", "LightGBM (SMOTE)"],
    "Accuracy": [0.82, 0.85, 0.85, 0.84],
    "Precision": [0.81, 0.84, 0.85, 0.83],
    "Recall": [0.84, 0.86, 0.85, 0.85],
    "F1-Score": [0.83, 0.85, 0.85, 0.84]
}

# Veriyi DataFrame'e dönüştürme
df = pd.DataFrame(data)

# CSV dosyasına kaydetme
df.to_csv("data/model_performance_metrics.csv", index=False)
print("Model performans metrikleri 'data/model_performance_metrics.csv' dosyasına kaydedildi.")
