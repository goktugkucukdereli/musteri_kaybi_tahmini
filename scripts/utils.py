import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(data_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    # Veri setini yükleme
    df = pd.read_csv(data_path)

    # TotalCharges sütununu sayısal tipe dönüştürme
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Kategorik sütunları kodlama
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                           'MultipleLines', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod', 'Churn']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Gereksiz sütunları kaldırma
    df.drop(['customerID'], axis=1, inplace=True)

    return df

def split_and_scale(df):
    # Hedef değişkeni (y) ve bağımsız değişkenleri (X) ayırma
    X = df.drop(columns=['Churn_Yes'])
    y = df['Churn_Yes']

    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Veriyi ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
