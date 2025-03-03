# Kalori Yakma Tahmini

Bu proje, fiziksel aktiviteler sırasında yakılan kaloriyi tahmin etmek için makine öğrenmesi modelleri kullanarak bir tahmin sistemi geliştirmektedir. Proje iki ana bileşenden oluşmaktadır:

1. **Model Eğitimi:** Veriyi temizleyerek, farklı regresyon modelleri eğitip en iyi performansı gösteren modeli seçiyoruz.
2. **Tahmin Arayüzü:** Kullanıcı dostu bir arayüz ile kullanıcıların girdiği veriler üzerinden kalori tahmini yapıyoruz.

---

## 1. Kullanılan Teknolojiler

- Python
- Scikit-Learn
- Pandas
- Matplotlib
- Seaborn
- Tkinter
- Joblib

---

## 2. Veri Seti

Proje, "Calories Burnt Prediction" veri setini kullanmaktadır. Bu veri seti, kullanıcının fiziksel özellikleri ve egzersiz verilerini içermektedir:

- **Gender (Cinsiyet):** Erkek veya Kadın
- **Age (Yaş):** Kullanıcının yaşı
- **Height (Boy):** cm cinsinden boy uzunluğu
- **Weight (Kilo):** kg cinsinden vücut ağırlığı
- **Duration (Süre):** Egzersizin süresi (dakika cinsinden)
- **Heart_Rate (Kalp Hızı):** Egzersiz sırasında ortalama kalp hızı
- **Body_Temp (Vücut Sıcaklığı):** Kullanıcının vücut sıcaklığı (°C cinsinden)
- **Calories (Hedef Değişken):** Egzersiz sırasında yakılan tahmini kalori miktarı

---

## 3. Model Eğitimi

Eğitim süreci aşağıdaki adımlardan oluşmaktadır:

1. **Veri Yükleme ve Ön İşleme:**
    - Eksik ve yinelenen verileri temizleme
    - Kategorik verileri sayısal değerlere dönüştürme
    - Veriyi eğitim ve test setlerine ayırma
2. **Model Eğitimi ve Performans Analizi:**
    - **Linear Regression**
    - **Ridge Regression**
    - **Decision Tree Regressor**
    - **Random Forest Regressor**
    - **K-Nearest Neighbors Regressor**
    - **Gradient Boosting Regressor**
    - Modeller karşılaştırılarak en yüksek başarıyı veren model belirlenmiştir.
3. **Hiperparametre Optimizasyonu:**
    - **GridSearchCV** ile en iyi parametre kombinasyonları belirlenmiştir.
4. **Modelin Kaydedilmesi:**
    - En iyi model **joblib** kütüphanesi ile `calorie_prediction_model.pkl` olarak kaydedilmiştir.
    - Kullanılan veri ölçekleyici de `scaler.pkl` olarak kaydedilmiştir.

---

## 4. Kullanıcı Arayüzü

Proje, tahmin modelini kullanıcı dostu bir grafiksel arayüz (GUI) ile birleştirmektedir. **Tkinter** kullanılarak geliştirilen bu arayüz sayesinde kullanıcılar verilerini girerek tahmini kalori değerini öğrenebilirler.

### Arayüz Özellikleri:
- Kullanıcıdan yaş, kilo, boy, egzersiz süresi, kalp hızı ve vücut sıcaklığı gibi bilgileri alır.
- **RandomForestRegressor** modelini kullanarak tahmini hesaplar.
- Kullanıcıya tahmini kaloriyi mesaj kutusu aracılığıyla gösterir.

---

## 5. Projenin Çalıştırılması

### 5.1. Gerekli Kütüphanelerin Yüklenmesi

Projeyi çalıştırmadan önce aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib tkinter
```

### 5.2. Model Eğitimini Çalıştırma

Aşağıdaki komut, veri seti ile modeli eğitir ve `calorie_prediction_model.pkl` ile `scaler.pkl` dosyalarını oluşturur:

```bash
python caloriees_burnt_prediction.py
```

### 5.3. Arayüzü Çalıştırma

Arayüzü başlatmak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
python caloriees_burnt_arayüz.py
```

---

## 6. Sonuç ve Değerlendirme

Bu proje, makine öğrenmesi tekniklerini kullanarak kişisel sağlık takibini kolaylaştırmayı amaçlamaktadır. Model, farklı algoritmalarla test edilmiş ve en iyi sonuçları veren **Random Forest Regressor** kullanılmıştır. Kullanıcı dostu bir arayüz ile herkesin rahatlıkla kullanabileceği bir tahmin sistemi oluşturulmuştur.

---

## 7. Gelecekteki Geliştirmeler
- Modelin daha büyük ve çeşitli veri setleriyle eğitilmesi
- Mobil veya web tabanlı bir uygulamaya dönüştürülmesi
- Kullanıcıların geçmiş tahminlerini saklayabileceği bir veri tabanı entegrasyonu
