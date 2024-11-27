# MMüşteri Kaybı Tahmini Projesi

🎯 Proje Amacı
Bu projeyi, Telco müşteri verilerini analiz ederek müşteri kaybını (churn) tahmin etmek amacıyla geliştirdim. Proje kapsamında, farklı makine öğrenimi modelleri ve veri işleme teknikleri kullanarak, müşteri kaybını önlemeye yönelik öngörüler sağlamayı hedefledim. Python, PostgreSQL ve Tableau gibi araçlar kullanarak veri analizi, modelleme, tahmin ve görselleştirme çalışmaları gerçekleştirdim.

📌 Proje Hedefleri
Müşteri kaybını tahmin ederek, hangi müşterilerin işletmeyi terk etme olasılığının yüksek olduğunu belirlemek.
Müşteri kaybını azaltmak için işletmelere öngörüler sunmak.
Sınıf dengesizliğini gidererek model performansını artırmak.
Makine öğrenimi modellerinin performansını karşılaştırmak ve en iyi modeli seçmek.
Tableau ile görselleştirme yaparak sonuçları daha anlaşılır hale getirmek.

📈 Proje Sonuçları
En İyi Model: Random Forest
Accuracy: 85%
Precision: 84%
Recall: 86%
F1-Score: 85%
En Önemli Özellikler:
Tenure: 25%
MonthlyCharges: 20%
TotalCharges: 15%
InternetService: 10%
Contract: 8%
Yeni verilerle yapılan tahminlerde müşteri kaybı oranı %28 olarak belirlendi.
Performans, karışıklık matrisi ve özellik önemi analizleri Tableau ile görselleştirildi.

📋 Projenin İçeriği
1. Veri Hazırlığı
Eksik değerler dolduruldu ve kategorik değişkenler One-Hot Encoding yöntemiyle dönüştürüldü.
processed_data.csv dosyasında saklanan işlenmiş veriler, model eğitimi için hazır hale getirildi.
2. Sınıf Dengesizliği Giderme
Churn sınıfındaki dengesizlik SMOTE (Synthetic Minority Oversampling Technique) yöntemiyle giderildi.
Dengelenmiş veri, balanced_data.csv dosyasına kaydedildi.
3. Model Eğitimi
Logistic Regression, Random Forest, XGBoost ve LightGBM gibi modeller eğitildi.
Her model Accuracy, Precision, Recall ve F1-Score metrikleriyle değerlendirildi.
4. Özellik Önemi Analizi
LightGBM modeli kullanılarak özelliklerin önem dereceleri hesaplandı.
Sonuçlar feature_importance.csv dosyasına kaydedildi ve Tableau ile görselleştirildi.
5. Yeni Verilerle Tahmin
Eğitilen Random Forest modeli, yeni müşteri verileri üzerinde test edildi.
Tahmin sonuçları new_data_with_predictions.csv dosyasına kaydedildi.
6. Görselleştirme
Tableau kullanılarak performans metrikleri, karışıklık matrisi ve özellik önemi analizleri görselleştirildi.

🛠 Kullanılan Teknolojiler
Teknoloji	              Kullanım Amacı
Python	                  Veri analizi, modelleme ve tahmin işlemleri.
PostgreSQL	              Veri depolama ve sorgulama.
Tableau	                  Veri görselleştirme.
pandas	                  Veri manipülasyonu ve analizi.
scikit-learn	          Makine öğrenimi modelleri ve değerlendirme.
imbalanced-learn	      SMOTE ile sınıf dengesizliği giderme.
xgboost	                  XGBoost modeli için.
lightgbm	              LightGBM modeli ve özellik önemi analizi.

🚀 Projenin Çalıştırılması
Gereksinimler
Python 3.8+
PostgreSQL
Tableau Desktop veya Tableau Public

Kurulum
Gerekli Python paketlerini yüklemek için:

bash
Copy code
pip install -r requirements.txt
PostgreSQL veritabanını başlatmak için:

bash
Copy code
python3 src/db_initializer.py
Veri hazırlamak ve işlemek için:

bash
Copy code
python3 scripts/data_preparation.py
Modelleri eğitmek için:

bash
Copy code
python3 scripts/random_forest.py
Performans karşılaştırması yapmak için:

bash
Copy code
python3 scripts/performans_comparison.py
Yeni veriler üzerinde tahmin yapmak için:

bash
Copy code
python3 scripts/final_model_predict.py

📊 Görselleştirmeler ve Analizler
1. Performans Karşılaştırması
Modellerin Accuracy, Precision, Recall ve F1-Score metriklerini Tableau’da görselleştirdim.


2. Karışıklık Matrisi
Gerçek ve tahmin edilen değerler için heatmap oluşturdum.


3. Özellik Önemi Analizi
LightGBM ile en önemli 10 özelliği görselleştirdim.


📂 Proje Yapısı
plaintext
Copy code
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

📜 Lisans
Bu proje, MIT Lisansı ile lisanslanmıştır. Lisans detayları için LICENSE dosyasını inceleyebilirsiniz.