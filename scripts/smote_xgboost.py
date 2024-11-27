from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import prepare_data

# Veri hazırlığı
df = prepare_data()

# Veriyi ayırma ve SMOTE uygulama
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Eğitim ve test setlerine ayırma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote)

# XGBoost modeli
xgb_model = XGBClassifier(random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Performans değerlendirme
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
class_report_xgb = classification_report(y_test, y_pred_xgb)

# Sonuçları yazdırma
print("SMOTE ile Dengelenmiş Verilerle XGBoost Sonuçları:")
print(f"Doğruluk: {accuracy_xgb:.2f}")
print("Karışıklık Matrisi:")
print(conf_matrix_xgb)
print("Sınıflandırma Raporu:")
print(class_report_xgb)
