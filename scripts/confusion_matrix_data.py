import pandas as pd

# Örnek y_true ve y_pred değerleri
y_true = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  # Gerçek değerler
y_pred = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]  # Tahmin edilen değerler

# Verileri DataFrame'e dönüştürme
confusion_data = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})

# CSV dosyasına kaydetme
confusion_data.to_csv("data/confusion_matrix_data.csv", index=False)
print("Karışıklık matrisi verileri 'data/confusion_matrix_data.csv' dosyasına kaydedildi.")
