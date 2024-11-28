# Müşteri Kaybı Tahmini Projesi

## 🎯 Proje Amacı

Bu projeyi, Telco müşteri verilerini analiz ederek müşteri kaybını (churn) tahmin etmek amacıyla geliştirdim. Proje kapsamında, farklı makine öğrenimi modelleri ve veri işleme teknikleri kullanarak, müşteri kaybını önlemeye yönelik öngörüler sağlamayı hedefledim. Python, PostgreSQL ve Tableau gibi araçlar kullanarak veri analizi, modelleme, tahmin ve görselleştirme çalışmaları gerçekleştirdim.

## 📌 Proje Hedefleri

- Müşteri kaybını tahmin ederek, hangi müşterilerin işletmeyi terk etme olasılığının yüksek olduğunu belirlemek.
- Müşteri kaybını azaltmak için işletmelere öngörüler sunmak.
- Sınıf dengesizliğini gidererek model performansını artırmak.
- Makine öğrenimi modellerinin performansını karşılaştırmak ve en iyi modeli seçmek.
- Tableau ile görselleştirme yaparak sonuçları daha anlaşılır hale getirmek.

## 📈 Proje Sonuçları

1. En İyi Model: Random Forest
   - **Accuracy:** 85%
   - **Precision:** 84%
   - **Recall:** 86%
   - **F1-Score:** 85%

2. En Önemli Özellikler:
   - **Tenure:** 25%
   - **MonthlyCharges:** 20%
   - **TotalCharges:** 15%
   - **InternetService:** 10%
   - **Contract:** 8%

3. Yeni verilerle yaptığım tahminlerde, müşteri kaybı oranını %28 olarak belirledim.
4. Performans, karışıklık matrisi ve özellik önemi analizlerini Tableau kullanarak görselleştirdim.

---

## 📋 Projenin İçeriği

1. **Veri Hazırlığı
    - Eksik değerleri doldurdum ve kategorik değişkenleri One-Hot Encoding yöntemiyle dönüştürdüm.
    - İşlenmiş verileri model eğitimi için `processed_data.csv` dosyasına kaydettim.

2. **Sınıf Dengesizliği Giderme
    - Churn sınıfındaki dengesizliği SMOTE (Synthetic Minority Oversampling Technique) yöntemiyle giderdim.
    - Dengelenmiş veriyi `balanced_data.csv` dosyasına kaydettim.

3. **Model Eğitimi
    - Logistic Regression, Random Forest, XGBoost ve LightGBM gibi modelleri eğittim.
    - Modelleri Accuracy, Precision, Recall ve F1-Score metrikleriyle değerlendirdim.

4. **Özellik Önemi Analizi
    - LightGBM modelini kullanarak özelliklerin önem derecelerini hesapladım.
    - Analiz sonuçlarını `feature_importance.csv` dosyasına kaydettim ve Tableau ile görselleştirdim.

5. **Yeni Verilerle Tahmin
    - Eğittiğim Random Forest modelini yeni müşteri verileri üzerinde test ettim.
    - Tahmin sonuçlarını `new_data_with_predictions.csv` dosyasına kaydettim.

6. **Görselleştirme
    - Performans metriklerini, karışıklık matrisini ve özellik önemi analizlerini Tableau kullanarak görselleştirdim.

---

🛠 Kullanılan Teknolojiler

**Teknoloji**	          **Kullanım Amacı**
- Python	                Veri analizi, modelleme ve tahmin işlemleri.
- PostgreSQL	          Veri depolama ve sorgulama.
- Tableau	             Veri görselleştirme.
- pandas	                Veri manipülasyonu ve analizi.
- scikit-learn	          Makine öğrenimi modelleri ve değerlendirme.
- imbalanced-learn	    SMOTE ile sınıf dengesizliği giderme.
- xgboost	             XGBoost modeli için.
- lightgbm	             LightGBM modeli ve özellik önemi analizi.

---

## 🚀 Projenin Çalıştırılması

### Gereksinimler
- Python 3.8+
- PostgreSQL
- Tableau Desktop veya Tableau Public

### Kurulum
1. Gerekli Python paketlerini yüklemek için:
    ```bash
    pip install -r requirements.txt
    ```

2. PostgreSQL veritabanını başlatmak için:
    ```bash
    python3 src/db_initializer.py
    ```

3. Veri hazırlamak ve işlemek için:
    ```bash
    python3 scripts/data_preparation.py
    ```

4. Modelleri eğitmek için:
    ```bash
    python3 scripts/random_forest.py
    ```

5. Performans karşılaştırması yapmak için:
    ```bash
    python3 scripts/performans_comparison.py
    ```

6. Yeni veriler üzerinde tahmin yapmak için:
    ```bash
    python3 scripts/final_model_predict.py
    ```

---

## 📊 Görselleştirmeler ve Analizler

### 1. Performans Karşılaştırması
Modellerin Accuracy, Precision, Recall ve F1-Score metriklerini Tableau’da görselleştirdim.

---

### 2. Karışıklık Matrisi
Gerçek ve tahmin edilen değerler için heatmap oluşturdum.

---

### 3. Özellik Önemi Analizi
LightGBM ile en önemli 10 özelliği görselleştirdim.

---

📂 Proje Yapısı

```plaintext
musteri_kaybi_tahmin_modeli/
│
├── data/                         # Veri dosyalarını saklayan klasör
│   ├── raw/                      # Ham veri dosyaları
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── processed/                     # İşlenmiş veri dosyaları
│   │   ├── processed_data.csv         # Model için temizlenmiş veri
│   │   └── balanced_data.csv          # SMOTE sonrası dengelenmiş veri
│   ├── new_data.csv                   # Yeni veri seti (tahmin için)
│   └── new_data_with_predictions.csv  # Tahmin sonuçlarıyla güncellenmiş veri
│
├── notebooks/                    # Jupyter Notebook dosyaları
│   ├── data_analysis.ipynb       # Verilerin analizi için Jupyter Notebook
│   └── feature_importance.ipynb  # Özellik önemi analizi için notebook
│
├── scripts/                      # Python kodlarını içeren klasör
│   ├── data_preparation.py       # Veri hazırlık işlemleri
│   ├── logistic_regression.py    # Logistic Regression analizi
│   ├── random_forest.py          # Random Forest analizi
│   ├── gradient_boosting.py      # XGBoost ve LightGBM analizi
│   ├── feature_analysis.py       # Özellik seçimi analizi
│   ├── smote_analysis.py         # SMOTE analizi
│   ├── smote_random_forest.py    # SMOTE ile Random Forest analizi
│   ├── smote_xgboost.py          # SMOTE ile XGBoost analizi
│   ├── smote_lightgbm.py         # SMOTE ile LightGBM analizi
│   ├── performans_comparison.py  # Modellerin performans karşılaştırması
│   ├── xgboost_hyperparameter_tuning.py # XGBoost hiperparametre optimizasyonu
│   ├── final_model.py                   # Final modelin eğitilmesi ve kaydedilmesi
│   ├── final_model_predict.py           # Yeni verilerle tahmin
│   └── utils.py                         # Ortak fonksiyonlar
│
├── tableau_exports/                                # Tableau görsellerini içeren klasör
│   └── Musteri_Kaybi_Tahmini_Gorsellestirme.twb    # Model performans, Karışıklık matrisi, Özellik önemi görselleştirmelerinin dosyası
│
├── reports/                        # Sonuç raporları ve görseller için klasör
│   ├── summary.md                  # Projenin genel sonuç raporu
│   ├── performance_comparison.png  # Model performans karşılaştırma görseli
│   └── confusion_matrix.png        # Karışıklık matrisi görseli
│
├── README.md                       # Projenin açıklamalarını içeren dosya
├── requirements.txt                # Python bağımlılıklarını içeren dosya
└── .gitignore                      # Git'e eklenmemesi gereken dosyaları belirler
```

---

## 📜 Lisans
Projeyi MIT Lisansı ile lisansladım. Lisans detayları için LICENSE dosyasını inceleyebilirsiniz.
