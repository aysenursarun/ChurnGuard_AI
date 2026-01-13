# 🛡️ ChurnGuard AI: Akıllı Müşteri Kayıp Yönetimi ve ROI Analizi

**ChurnGuard AI**, telekomünikasyon sektöründeki müşteri terk (churn) riskini minimize etmek için tasarlanmış, yapay zeka tabanlı bir **Karar Destek Sistemi**dir. Uygulama, sadece "kim gidecek?" sorusuna yanıt vermekle kalmaz, "onu tutmak için ne yapmalıyız ve bu bize ne kazandırır?" sorularına finansal verilerle yanıt verir.

## 🚀 Canlı Uygulama Linki
Uygulamayı tarayıcınızda deneyimlemek için tıklayın:  
👉 [https://churnguardai-dl69eehfcg4dyhm4bycahd.streamlit.app/](https://churnguardai-dl69eehfcg4dyhm4bycahd.streamlit.app/)

## ✨ Öne Çıkan Özellikler

* **🎯 Hassas Risk Tahmini:** %80 Doğruluk (Accuracy) ve %74 Duyarlılık (Recall) ile yüksek riskli müşterileri erken safhada saptama.
* **💰 Finansal Strateji Merkezi:** Risk altındaki aylık geliri hesaplama ve %25 başarı hedefiyle yıllık **400.000$+** potansiyel kazanç projeksiyonu.
* **📊 Karar Destek Görselleri:**
    * **Kritik Eşik Analizi:** Müşterilerin fiyat hassasiyetini gösteren yoğunluk haritaları.
    * **Segmentasyon:** VIP, Riskli Yeni ve Sadık müşteri gruplarının otomatik ayrımı.
* **💡 Akıllı Aksiyon Planları:** Terk eğilimi olan müşteriler için otomatik kişiselleştirilmiş kampanya ve e-posta taslakları üretimi.
* **🔄 Stratejik Simülasyon (What-If):** İndirim veya teknik destek gibi müdahalelerin risk skorunu nasıl düşüreceğini anlık görme.

## 🛠️ Teknik Stack

* **Dil:** Python 3.9+
* **Framework:** Streamlit
* **Makine Öğrenmesi:** Scikit-learn (Random Forest Sınıflandırıcı)
* **Veri Görselleştirme:** Seaborn, Matplotlib
* **Model Yönetimi:** Joblib

## 📦 Proje Yapısı

```text
ChurnGuard_AI/
├── .streamlit/          # Kurumsal tema ayarları (config.toml)
├── app.py               # Ana uygulama kodu
├── requirements.txt     # Kütüphane bağımlılıkları
├── features_v2.pkl      # Model özellik listesi
├── churn_model_v2_...   # Eğitilmiş ML modeli
└── WA_Fn-UseC...csv     # Varsayılan eğitim veri seti
