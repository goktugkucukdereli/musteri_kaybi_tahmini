from logistic_regression import logistic_regression_analysis
from random_forest import random_forest_analysis
from gradient_boosting import gradient_boosting_analysis
from feature_analysis import feature_analysis
from smote_analysis import smote_analysis

# Logistic Regression sonuçlarını al
logistic_result = logistic_regression_analysis()

# Random Forest sonuçlarını al
random_forest_result = random_forest_analysis()

# Gradient Boosting sonuçlarını al
gradient_boosting_result = gradient_boosting_analysis()

# Feature Analysis sonuçlarını al
feature_analysis_result = feature_analysis()

# Smote Analysis sonuçlarını al
smote_analysis_result = smote_analysis()

# Performans karşılaştırması
print("Performans Karşılaştırması:")
print(f"Logistic Regression: {logistic_result}")
print(f"Random Forest: {random_forest_result}")
print(f"Gradient Boosting: {gradient_boosting_result}")
print(f"Feature_Analysis: {feature_analysis_result}")
print(f"Smote_Analysis: {smote_analysis_result}")