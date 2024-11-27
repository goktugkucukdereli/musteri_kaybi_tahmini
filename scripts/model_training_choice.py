from logistic_regression import logistic_regression_analysis
from random_forest import random_forest_analysis
from gradient_boosting import gradient_boosting_analysis
from feature_analysis import feature_analysis
from smote_analysis import smote_analysis

# Kullanıcıdan hangi analizin çalıştırılacağını seçmesini isteyin
print("Analiz Seçenekleri:")
print("1: Logistic Regression")
print("2: Random Forest")
print("3: Gradient Boosting")
print("4: Feature Analysis")
print("5: SMOTE Analysis")
choice = int(input("Bir analiz seçin (1-5): "))

if choice == 1:
    logistic_regression_analysis()
elif choice == 2:
    random_forest_analysis()
elif choice == 3:
    gradient_boosting_analysis()
elif choice == 4:
    feature_analysis()
elif choice == 5:
    smote_analysis()
else:
    print("Geçersiz seçim.")
