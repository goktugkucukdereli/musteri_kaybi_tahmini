from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
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

# XGBoost modeli için hiperparametre arama
xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)

# Hiperparametreler
param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# RandomizedSearchCV ile hiperparametre optimizasyonu
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,  # Rastgele 50 kombinasyon denenecek
    scoring="accuracy",
    cv=3,  # 3 katmanlı çapraz doğrulama
    verbose=1,
    random_state=42,
    n_jobs=-1,  # Tüm çekirdekler kullanılır
)

# En iyi parametreleri arama
random_search.fit(X_train, y_train)

# En iyi parametreleri yazdırma
print("En İyi Parametreler:", random_search.best_params_)

# Optimize edilmiş model ile tahmin yapma
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Performans değerlendirme
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nXGBoost Hiperparametre Optimizasyonu Sonrası Sonuçlar:")
print(f"Doğruluk: {accuracy:.2f}")
print("Karışıklık Matrisi:")
print(conf_matrix)
print("Sınıflandırma Raporu:")
print(class_report)
