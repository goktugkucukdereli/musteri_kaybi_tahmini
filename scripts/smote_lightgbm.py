from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
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

# LightGBM modeli
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)

# Performans değerlendirme
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
conf_matrix_lgbm = confusion_matrix(y_test, y_pred_lgbm)
class_report_lgbm = classification_report(y_test, y_pred_lgbm)

# Sonuçları yazdırma
print("SMOTE ile Dengelenmiş Verilerle LightGBM Sonuçları:")
print(f"Doğruluk: {accuracy_lgbm:.2f}")
print("Karışıklık Matrisi:")
print(conf_matrix_lgbm)
print("Sınıflandırma Raporu:")
print(class_report_lgbm)
