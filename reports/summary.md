1. Giriş
Müşteri kaybını (churn) tahmin etmek, işletmelerin müşteri ilişkilerini güçlendirmek ve gelir kayıplarını önlemek için kritik öneme sahiptir. Bu projede, Telco müşteri verileri kullanılarak makine öğrenimi modelleriyle müşteri kaybı tahmini gerçekleştirilmiştir.

Projenin temel hedefleri şunlardır:

Veri setini analiz ederek anlamlı özellikler oluşturmak.
Sınıf dengesizliğini ele almak.
Birden fazla model kullanarak en iyi performans gösteren modeli belirlemek.
Performans sonuçlarını görselleştirerek anlaşılır hale getirmek.
2. Veri Hazırlığı
Veri seti, Telco müşterilerine ait demografik, hesap ve hizmet bilgilerini içerir. Veri hazırlığı sürecinde aşağıdaki işlemler uygulanmıştır:

Eksik Değerler: TotalCharges sütunundaki eksik değerler medyan ile doldurulmuştur.
Kategorik Değişkenler: Kategorik sütunlar One-Hot Encoding yöntemiyle dönüştürülmüştür.
Sınıf Dengesizliği: Churn sınıfı dengesiz olduğu için SMOTE yöntemiyle dengelenmiştir.
Sonuç:

Veri seti, model eğitimine uygun hale getirilmiştir.
İşlenmiş veri processed_data.csv olarak kaydedilmiştir.
Dengelenmiş veri balanced_data.csv olarak kaydedilmiştir.
3. Model Eğitimi
Farklı makine öğrenimi modelleri kullanılarak müşteri kaybı tahmini gerçekleştirilmiştir:

Logistic Regression
Random Forest
XGBoost
LightGBM
Tüm modeller Accuracy, Precision, Recall ve F1-Score metrikleriyle değerlendirilmiştir.

Elde Edilen En İyi Sonuç:

Model: Random Forest
Accuracy: 85%
Precision (True): 84%
Recall (True): 86%
F1-Score (True): 85%
4. Performans Karşılaştırması
Modellerin performans metrikleri aşağıdaki gibidir:

Model	Accuracy	Precision (True)	Recall (True)	F1-Score (True)
Logistic Regression	0.82	0.81	0.84	0.83
Random Forest	0.85	0.84	0.86	0.85
XGBoost	0.85	0.85	0.85	0.85
LightGBM	0.84	0.83	0.85	0.84
5. Özellik Önemi Analizi
LightGBM modeli kullanılarak özellik önemleri hesaplanmıştır. En önemli özellikler:

Tenure: 25%
MonthlyCharges: 20%
TotalCharges: 15%
InternetService: 10%
Contract: 8%
Özellik önemi görselleştirilmiştir ve feature_importance.csv dosyasına kaydedilmiştir.

6. Yeni Verilerle Tahmin
Eğitilen final model, yeni müşteri verileri üzerinde test edilmiştir. Tahminler sonucu:

Yeni veri setine ait tahminler new_data_with_predictions.csv dosyasına kaydedilmiştir.
Tahmin sonuçları aşağıdaki gibidir:
Churn Olmayan Müşteriler: 72%
Churn Eden Müşteriler: 28%
7. Görselleştirme
Projenin önemli sonuçları Tableau kullanılarak görselleştirilmiştir:

Performans Karşılaştırması: Accuracy, Precision, Recall ve F1-Score için bar chart'lar.
Karışıklık Matrisi: Gerçek ve tahmin edilen değerler için heatmap.
Özellik Önemi Analizi: LightGBM ile belirlenen en önemli özellikler için bar chart.
8. Sonuç ve Çıkarımlar
Bu projede müşteri kaybı tahminini optimize etmek için farklı modeller ve teknikler kullanılmıştır. Random Forest modeli, performans metrikleri açısından en başarılı model olarak seçilmiştir. Sınıf dengesizliğini gidermek için kullanılan SMOTE yöntemi, tahmin performansını önemli ölçüde artırmıştır.

Ekler
İşlenmiş Veri: processed_data.csv
Dengelenmiş Veri: balanced_data.csv
Tahmin Sonuçları: new_data_with_predictions.csv
Görseller:
Performans Karşılaştırması: performance_comparison.png
Karışıklık Matrisi: confusion_matrix.png
Özellik Önemi: feature_importance.png