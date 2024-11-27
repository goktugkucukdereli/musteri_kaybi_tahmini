from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import prepare_data, split_and_scale

# Veri hazırlığı
df = prepare_data()

# Veriyi böl ve ölçeklendir
X_train, X_test, y_train, y_test = split_and_scale(df)

# Logistic Regression modeli
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Performans değerlendirme
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Sonuçları yazdırma
print(f"Logistic Regression Doğruluk: {accuracy:.2f}")
print("\nKarışıklık Matrisi:")
print(conf_matrix)
print("\nSınıflandırma Raporu:")
print(class_report)
