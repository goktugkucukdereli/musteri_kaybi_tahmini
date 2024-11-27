import matplotlib.pyplot as plt
import pandas as pd

# Modellerin performans metrikleri
data = {
    "Model": [
        "Logistic Regression",
        "Random Forest (SMOTE)",
        "XGBoost (SMOTE)",
        "LightGBM (SMOTE)",
    ],
    "Accuracy": [0.82, 0.85, 0.85, 0.84],
    "Precision (True)": [0.81, 0.84, 0.85, 0.83],
    "Recall (True)": [0.84, 0.86, 0.85, 0.85],
    "F1-Score (True)": [0.83, 0.85, 0.85, 0.84],
}

# Veriyi tabloya dönüştürme
df = pd.DataFrame(data)

# Tabloyu yazdırma
print("Modellerin Performans Karşılaştırması:")
print(df)

# Accuracy grafiği
plt.figure(figsize=(10, 6))
plt.bar(df["Model"], df["Accuracy"], color="skyblue")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Modellerin Accuracy Karşılaştırması")
plt.ylim(0.8, 0.9)  # Y ekseni aralığı
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")  # Grafiği kaydetme
plt.show()

# Precision, Recall ve F1-Score grafikleri
metrics = ["Precision (True)", "Recall (True)", "F1-Score (True)"]
colors = ["lightgreen", "gold", "lightcoral"]

for i, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    plt.bar(df["Model"], df[metric], color=colors[i])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"Modellerin {metric} Karşılaştırması")
    plt.ylim(0.8, 0.9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{metric.lower().replace(' ', '_')}_comparison.png")  # Grafiği kaydetme
    plt.show()
