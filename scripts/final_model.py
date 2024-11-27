import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import prepare_data

# Veri hazırlığı
df = prepare_data()

# SMOTE ile veri dengesizliğini giderme
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote)

# Final model (Random Forest)
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Performans değerlendirme
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Sonuçları yazdırma
print("Final Model Sonuçları (Random Forest):")
print(f"Doğruluk: {accuracy:.2f}")
print("Karışıklık Matrisi:")
print(conf_matrix)
print("Sınıflandırma Raporu:")
print(class_report)

# Final modeli kaydetme
joblib.dump(final_model, "final_model_rf.pkl")
print("Final model 'final_model_rf.pkl' dosyasına kaydedildi.")
