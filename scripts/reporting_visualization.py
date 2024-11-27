import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Karışıklık Matrisi Görselleştirme
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()

# 2. Precision, Recall, ve F1-Score Grafikleri
def plot_metrics_comparison(data):
    metrics = ["Precision", "Recall", "F1-Score"]
    colors = ["lightblue", "lightgreen", "lightcoral"]
    
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        plt.bar(data["Model"], data[metric], color=colors[i])
        plt.title(f"{metric} Karşılaştırması")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.ylim(0.8, 0.9)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.show()

# 3. Genel Performans Karşılaştırması
def plot_accuracy_comparison(data):
    plt.figure(figsize=(10, 6))
    plt.bar(data["Model"], data["Accuracy"], color="skyblue")
    plt.title("Accuracy Karşılaştırması")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.ylim(0.8, 0.9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# 4. Veriler
# Modellerin Performans Metrikleri
data = {
    "Model": ["Logistic Regression", "Random Forest (SMOTE)", "XGBoost (SMOTE)", "LightGBM (SMOTE)"],
    "Accuracy": [0.82, 0.85, 0.85, 0.84],
    "Precision": [0.81, 0.84, 0.85, 0.83],
    "Recall": [0.84, 0.86, 0.85, 0.85],
    "F1-Score": [0.83, 0.85, 0.85, 0.84]
}
df = pd.DataFrame(data)

# 5. Çalıştırma
if __name__ == "__main__":
    print("Raporlama ve Görselleştirme Başlıyor...")
    
    # Accuracy Karşılaştırması
    plot_accuracy_comparison(df)
    
    # Precision, Recall, F1-Score Karşılaştırmaları
    plot_metrics_comparison(df)
    
    # Örnek Karışıklık Matrisi Görselleştirme (Random Forest)
    # Bu değerler tahmin edilen sonuçlardan alınmalı
    y_true = [0, 1, 0, 1, 1]  # Örnek gerçek değerler
    y_pred = [0, 1, 0, 0, 1]  # Örnek tahminler
    plot_confusion_matrix(y_true, y_pred, title="Random Forest Confusion Matrix")
