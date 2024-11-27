import matplotlib.pyplot as plt
import numpy as np
from utils import prepare_data, split_and_scale
from lightgbm import LGBMClassifier

# Veri hazırlığı
df = prepare_data()

# Veriyi böl ve ölçeklendir
X_train, X_test, y_train, y_test = split_and_scale(df)

# LightGBM modeli
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)

# Özellik önemini çıkarma
feature_importance = lgbm_model.feature_importances_
features = df.drop(columns=["Churn_Yes"]).columns

# Özellik önemini sıralama
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = features[sorted_idx]
sorted_importances = feature_importance[sorted_idx]

# Görselleştirme
plt.figure(figsize=(10, 8))
plt.barh(sorted_features[:10], sorted_importances[:10], color="skyblue")
plt.xlabel("Özellik Önemi")
plt.ylabel("Özellikler")
plt.title("LightGBM - En Önemli 10 Özellik")
plt.gca().invert_yaxis()

# Grafiği kaydet ve göster
plt.savefig("feature_importance.png")  # PNG formatında kaydet
plt.show()  # Grafiği ekranda göster
