from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import prepare_data, split_and_scale

# Veri hazırlığı
df = prepare_data()

# Veriyi böl ve ölçeklendir
X_train, X_test, y_train, y_test = split_and_scale(df)

# XGBoost modeli
xgb_model = XGBClassifier(random_state=42, eval_metric="logloss")  # XGBoost modeli oluşturma
xgb_model.fit(X_train, y_train)  # Modeli eğitme
y_pred_xgb = xgb_model.predict(X_test)  # Test verisi ile tahmin yapma

# XGBoost performansı
print("XGBoost Sonuçları:")
print(f"Doğruluk: {accuracy_score(y_test, y_pred_xgb):.2f}")  # Doğruluk oranını yazdırma
print(confusion_matrix(y_test, y_pred_xgb))  # Karışıklık matrisi
print(classification_report(y_test, y_pred_xgb))  # Precision, recall ve f1-score gibi metrikler
