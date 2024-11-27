from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import prepare_data, split_and_scale

# Veri hazırlığı
df = prepare_data()

# Veriyi böl ve ölçeklendir
X_train, X_test, y_train, y_test = split_and_scale(df)

# LightGBM modeli
lgbm_model = LGBMClassifier(random_state=42)  # LightGBM modeli oluşturma
lgbm_model.fit(X_train, y_train)  # Modeli eğitme
y_pred_lgbm = lgbm_model.predict(X_test)  # Test verisi ile tahmin yapma

# LightGBM performansı
print("LightGBM Sonuçları:")
print(f"Doğruluk: {accuracy_score(y_test, y_pred_lgbm):.2f}")  # Doğruluk oranını yazdırma
print(confusion_matrix(y_test, y_pred_lgbm))  # Karışıklık matrisi
print(classification_report(y_test, y_pred_lgbm))  # Precision, recall ve f1-score gibi metrikler
