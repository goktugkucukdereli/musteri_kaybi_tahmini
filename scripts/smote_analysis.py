from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import prepare_data, split_and_scale

# Veri hazırlığı
df = prepare_data()

# Veriyi eğitim ve test setlerine ayırma (ölçeklendirme yapılmadan önce)
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']

# SMOTE uygulama
smote = SMOTE(random_state=42)  # SMOTE modeli
X_smote, y_smote = smote.fit_resample(X, y)  # Azınlık sınıfını dengeleme

# Veriyi tekrar eğitim ve test setlerine ayırma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote)

# Logistic Regression ile model eğitimi
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Performans değerlendirme
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Sonuçları yazdırma
print("SMOTE ile Dengelenmiş Verilerle Logistic Regression Sonuçları:")
print(f"Doğruluk: {accuracy:.2f}")
print("Karışıklık Matrisi:")
print(conf_matrix)
print("Sınıflandırma Raporu:")
print(class_report)
