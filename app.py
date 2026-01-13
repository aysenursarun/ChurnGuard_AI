import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# --- SAYFA YAPILANDIRMASI ---
# Uygulama başlığı ve geniş ekran modunu ayarlar
st.set_page_config(page_title="Dynamic Churn Intelligence", layout="wide")

# --- 1. MODEL VE VARSAYILAN VERİLERİN YÜKLENMESİ ---
@st.cache_resource # Sayfa her yenilendiğinde modelin tekrar yüklenip yavaşlamasını engeller
def load_assets():
    try:
        # Önceden eğitilmiş model ve özellik (feature) listesini yükler
        model = joblib.load('churn_model_v2_recall73.pkl')
        features = joblib.load('features_v2.pkl')
        return model, features
    except:
        return None, None

model, features = load_assets()

def check_data_quality(df):
    """Yüklenen verideki kritik eksikleri ve veri sağlığını denetler."""
    errors = []
    # Analizlerin çalışması için veri setinde mutlaka bulunması gereken sütunlar
    required = ['tenure', 'MonthlyCharges', 'Contract', 'Churn', 'InternetService', 'TechSupport', 'PaymentMethod']
    
    # Sütun varlık kontrolü
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"❌ Eksik Sütunlar: {', '.join(missing)}")
    
    # Boş (NaN) değer kontrolü
    if df.isnull().any().any():
        errors.append("⚠️ Veride boş (NaN) değerler var. Analizler tam doğru olmayabilir.")
        
    # Veri tipi kontrolü (Abonelik süresi sayısal olmalıdır)
    if 'tenure' in df.columns and not pd.api.types.is_numeric_dtype(df['tenure']):
        errors.append("❌ 'tenure' sütunu sayısal olmalıdır.")
        
    return errors

# --- SIDEBAR EN ÜST BOŞLUĞA YERLEŞTİRME (CSS HACK) ---
# Streamlit'in sidebar üst boşluğunu kaldırarak logoyu en tepeye taşır
st.sidebar.markdown("""
    <style>
        [data-testid="stSidebarContent"] {
            padding-top: 0rem !important;
        }
        .sidebar-logo {
            margin-top: -50px; 
            padding-bottom: 20px;
        }
    </style>
    
    <div class="sidebar-logo" style='text-align: left;'>
        <h3 style='color: #FF4B4B; margin-bottom: 0; font-size: 1.5rem;'>🛡️ ChurnGuard AI</h3>
        <p style='font-size: 0.75em; color: gray;'>Akıllı Müşteri Kayıp Yönetimi</p>
    </div>
""", unsafe_allow_html=True)

# --- 2. VERİ YÖNETİMİ (HİBRİT YAPI) ---
st.sidebar.header("📁 Veri Yönetimi")
uploaded_file = st.sidebar.file_uploader("Yeni Şirket Veri Setinizi Yükleyin (CSV)", type="csv")

if uploaded_file is not None:
    temp_df = pd.read_csv(uploaded_file) # Veriyi geçici olarak belleğe alır
    quality_issues = check_data_quality(temp_df) # Kalite kontrolü yapar
    
    if quality_issues:
        for err in quality_issues:
            st.sidebar.error(err)
        # Hata varsa veri setini boşaltır ve sağ tarafın çalışmasını durdurur
        df = pd.DataFrame() 
        st.sidebar.warning("⚠️ Lütfen yukarıdaki hataları düzelttikten sonra tekrar yükleyin.")
    else:
        df = temp_df # Hata yoksa ana dataframe'e aktarır
        st.sidebar.success("✅ Veri seti başarıyla doğrulandı.")
else:
    # Kullanıcı dosya yüklemediyse varsayılan eğitim verisini yüklemeye çalışır
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv') 
        st.sidebar.info("ℹ️ Eğitim veri seti üzerinden analiz yapılıyor.")
    except:
        df = pd.DataFrame()

# --- GÜVENLİK BARİYERİ ---
# Veri seti yoksa veya hatalıysa uygulamanın analiz kısımlarını göstermez
if df.empty:
    st.info("👋 Hoş Geldiniz! Lütfen analizleri başlatmak için sol menüden geçerli ve hatasız bir veri seti yükleyin.")
    st.stop()

# --- DİNAMİK ANALİTİK HESAPLAMALAR ---
# Veri üzerinden kritik eşik ve terk oranlarını hesaplar
if not df.empty and 'MonthlyCharges' in df.columns and 'Churn' in df.columns:
    churn_yes = df[df['Churn'] == 'Yes']
    kritik_esik = churn_yes['MonthlyCharges'].median()
    genel_churn_orani = (df['Churn'] == 'Yes').mean() * 100
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    en_riskli_sozlesme = contract_churn.idxmax() if not contract_churn.empty else "Bilinmiyor"
else:
    # Veri seti gelmezse hata almamak için fallback değerleri
    kritik_esik = 79.65
    genel_churn_orani = 26.5
    contract_churn = pd.Series({'Month-to-month': 42.7, 'One year': 11.2, 'Two year': 2.8})
    en_riskli_sozlesme = "Aylık"

# --- TOPLU TAHMİN FONKSİYONU ---
def run_batch_prediction(df, model, features):
    """Tüm portföyü tarayarak risk skorlarını topluca üretir."""
    X_batch = pd.DataFrame(0, index=df.index, columns=features)
    
    # Sayısal değer aktarımı
    if 'tenure' in features: X_batch['tenure'] = df['tenure']
    if 'MonthlyCharges' in features: X_batch['MonthlyCharges'] = df['MonthlyCharges']
    if 'TotalCharges' in features: X_batch['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
    
    # Kategorik değerlerin One-Hot Encoding formatına eşlenmesi
    cat_cols = ['Contract', 'InternetService', 'TechSupport', 'PaymentMethod']
    for col in cat_cols:
        if col in df.columns:
            for val in df[col].unique():
                feat_name = f"{col}_{val}"
                if feat_name in features:
                    X_batch.loc[df[col] == val, feat_name] = 1
                    
    # Olasılık skorlarını döndürür
    probs = model.predict_proba(X_batch)[:, 1]
    return probs

# Tablo sütunlarını Türkçeleştirmek için mapping sözlüğü
column_mapping = {
    'customerID': 'Müşteri Kimliği',
    'tenure': 'Abonelik Süresi (Ay)',
    'MonthlyCharges': 'Aylık Ücret ($)',
    'Contract': 'Sözleşme Tipi',
    'Risk_Skoru': 'Terk Riski (%)',
    'InternetService': 'İnternet Tipi',
    'TechSupport': 'Teknik Destek',
    'PaymentMethod': 'Ödeme Yöntemi',
    'CLV': 'Müşteri Ömür Boyu Değeri ($)',
    'Segment': 'Değer Segmenti'
}

# --- 3. ANA PANEL TASARIMI (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Tahmin Paneli", "📊 Genel Şirket Analizi", "🚀 Aksiyon ve Strateji Merkezi", "📋 Operasyonel Liste"])

# --- TAB 1: BİREYSEL MÜŞTERİ ANALİZİ ---
with tab1:
    st.markdown("### 🎯 Müşteri Terk Analizi ve Aksiyon Planı")
    
    def user_input_features():
        """Sidebar üzerinden kullanıcıdan müşteri verilerini alır."""
        st.sidebar.header("📝 Müşteri Detayları")
        tenure = st.sidebar.slider("Abonelik Süresi (Ay)", 1, 72, 12)
        monthly_charges = st.sidebar.number_input("Aylık Ücret ($)", 0.0, 150.0, 65.0)
        contract = st.sidebar.selectbox("Sözleşme Tipi", ["Month-to-month", "One year", "Two year"])
        internet = st.sidebar.selectbox("İnternet Servisi", ["Fiber optic", "DSL", "No"])
        tech_support_val = st.sidebar.selectbox("Teknik Destek", ["Yes", "No"])
        payment_method = st.sidebar.selectbox("Ödeme Yöntemi", 
                                             ["Electronic check", "Mailed check", 
                                              "Bank transfer (automatic)", "Credit card (automatic)"])

        # Model girdisini oluşturur
        input_df = pd.DataFrame(0, index=[0], columns=features)
        if 'tenure' in features: input_df['tenure'] = tenure
        if 'MonthlyCharges' in features: input_df['MonthlyCharges'] = monthly_charges
        if 'TotalCharges' in features: input_df['TotalCharges'] = tenure * monthly_charges
        
        # Seçilen kategorik verileri işaretler
        for col in [f"Contract_{contract}", f"InternetService_{internet}", 
                    f"TechSupport_{tech_support_val}", f"PaymentMethod_{payment_method}"]:
            if col in features: input_df[col] = 1
            
        return input_df, contract, monthly_charges, tenure, tech_support_val, payment_method

    input_df, user_contract, user_charges, user_tenure, tech_support, user_payment = user_input_features()

    # Model performans özet bilgisi
    st.sidebar.markdown("---")
    st.sidebar.caption("🤖 **Model Performans Özeti**")
    st.sidebar.caption("Doğruluk (Accuracy): %80")
    st.sidebar.caption("Duyarlılık (Recall): %74")
    st.sidebar.caption("Son Güncelleme: Ocak 2026")

    if st.button("🚀 Analizi Başlat ve Aksiyon Üret"):
        if model is not None:
            # 1. MEVCUT DURUM TAHMİNİ
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]
            
            # --- ULTRA SENARYO HESAPLAMA (Birleşik Teklif Etkisi) ---
            alt_input_ultra = input_df.copy()
            if 'Contract_One year' in features:
                alt_input_ultra[[c for c in features if 'Contract' in c]] = 0
                alt_input_ultra['Contract_One year'] = 1
            
            indirimli_fiyat = round(user_charges * 0.85, 2)
            alt_input_ultra['MonthlyCharges'] = indirimli_fiyat
            
            if 'TechSupport_Yes' in features:
                alt_input_ultra[[c for c in features if 'TechSupport' in c]] = 0
                alt_input_ultra['TechSupport_Yes'] = 1
                
            if 'TotalCharges' in features:
                alt_input_ultra['TotalCharges'] = user_tenure * indirimli_fiyat
            
            prob_ultra = model.predict_proba(alt_input_ultra)[0][1]

            # --- DİĞER WHAT-IF HESAPLAMALARI ---
            alt_input_1 = input_df.copy() # Sadece taahhüt
            if 'Contract_One year' in features:
                alt_input_1[[c for c in features if 'Contract' in c]] = 0
                alt_input_1['Contract_One year'] = 1
            prob_s1 = model.predict_proba(alt_input_1)[0][1]

            alt_input_2 = input_df.copy() # İndirim ve destek
            alt_input_2['MonthlyCharges'] = indirimli_fiyat
            if 'TechSupport_Yes' in features:
                alt_input_2[[c for c in features if 'TechSupport' in c]] = 0
                alt_input_2['TechSupport_Yes'] = 1
            if 'TotalCharges' in features:
                alt_input_2['TotalCharges'] = user_tenure * indirimli_fiyat
            prob_s2 = model.predict_proba(alt_input_2)[0][1]

            # Sonuç Özet Kartları
            st.divider()
            col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
            with col_m1:
                risk_color = "red" if probability > 0.5 else "green"
                st.markdown(f"**Tahmin Edilen Risk**")
                st.markdown(f"<h2 style='color:{risk_color};'>%{probability*100:.1f}</h2>", unsafe_allow_html=True)
            with col_m2:
                st.markdown("**Sistem Kararı**")
                if prediction[0] == 1: st.error("🚨 TERK EĞİLİMİ")
                else: st.success("✅ SADIK PROFİL")
            with col_m3:
                t_med = df['tenure'].median() if not df.empty else 29
                st.markdown("**Müşteri Segmenti**")
                if user_charges >= kritik_esik and user_tenure < t_med: st.warning("📍 Riskli Yeni Müşteri")
                elif user_charges >= kritik_esik and user_tenure >= t_med: st.info("📍 VIP Müşteri")
                else: st.success("📍 Standart / Sadık")

            # Analiz Gövdesi
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🧐 Kararı Etkileyen Faktörler")
                f_imp = {"Sözleşme": 0.45 if user_contract == "Month-to-month" else 0.05,
                         "Fatura": 0.35 if user_charges > kritik_esik else 0.15,
                         "Ödeme": 0.20 if user_payment == "Electronic check" else 0.05}
                st.bar_chart(pd.Series(f_imp))
                
                # CLV ve Gelecek Değer metrikleri
                st.write("---")
                customer_clv = user_charges * user_tenure
                future_revenue = user_charges * 12 
                
                col_clv1, col_clv2 = st.columns(2)
                with col_clv1:
                    st.metric("Mevcut CLV (Geçmiş Değer)", f"{customer_clv:,.2f} $")
                with col_clv2:
                    st.metric("Gelecek 12 Ay Potansiyeli", f"{future_revenue:,.2f} $")
                
                if customer_clv > (df['MonthlyCharges'].mean() * df['tenure'].mean() if not df.empty else 1500):
                    st.info("💎 **Yüksek Değerli Müşteri:** Bu müşteriyi elde tutmak, yıllık bazda ciddi bir gelir koruması sağlar.")

            with c2:
                st.subheader("💡 Önerilen Koruma Aksiyonları")
                t1, t2 = st.columns(2)
                t1.metric("12 Ay Taahhüt İndirimi", f"{user_charges*0.9:.2f} $", "-%10")
                t2.metric("VIP Sadakat Paketi", f"{indirimli_fiyat:.2f} $", "-%15")
                
                st.markdown("**Aksiyon Önceliği:**")
                if probability > 0.7: st.error("🔴 **KRİTİK:** Hemen İletişime Geçilmeli")
                else: st.warning("🟡 **ORTA:** E-posta/SMS Yeterli")

                # Otomatik İletişim Metni Taslağı
                if prediction[0] == 1:
                    st.divider()
                    st.subheader("✉️ Otomatik İletişim Taslağı")
                    email_body = f"""Sayın Müşterimiz,
                    
                        Şirketimize olan {user_tenure} aylık bağlılığınız için teşekkür ederiz. 

                        Aboneliğinizi 1 Yıllık Taahhütle yenilemeniz durumunda:
                        ✅ Aylık ücretinizi {user_charges:.2f} $'dan {indirimli_fiyat:.2f} $'a düşürüyoruz.
                        ✅ Size özel ücretsiz 'Teknik Destek' paketini tanımlıyoruz.

                        Teklifi onaylamak için bu e-postayı yanıtlamanız yeterlidir."""
                    
                    st.text_area("Kampanya Metni", email_body, height=200)
                    # Senkronize edilmiş risk iyileşme tahmini
                    st.success(f"📈 Bu Birleşik Teklifle risk %{probability*100:.1f} -> %{prob_ultra*100:.1f}'e düşer.")

            # Stratejik Senaryo Barları
            st.divider()
            st.subheader("🔄 Stratejik Simülasyon (What-If)")
            w1, w2 = st.columns(2)
            with w1:
                st.write("**Senaryo 1: Sadece Taahhüt**")
                st.progress(prob_s1); st.write(f"Risk: %{prob_s1*100:.1f}")
            with w2:
                st.write("**Senaryo 2: %15 İndirim + Teknik Destek**")
                st.progress(prob_s2); st.write(f"Risk: %{prob_s2*100:.1f}")

# --- TAB 2: ŞİRKET GENEL ANALİZLERİ ---
with tab2:
    st.title("📊 Şirket Genel Analiz Paneli")
    st.write("Veri setindeki müşteri davranışlarını ve risk dağılımlarını standart ölçümlerle analiz eder.")

    if not df.empty:
        # Şirket geneli özet metrikler
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Toplam Müşteri", f"{len(df):,}")
        m2.metric("Genel Terk Oranı", f"%{genel_churn_orani:.1f}")
        m3.metric("Dinamik Kritik Eşik", f"{kritik_esik:.2f} $")
        m4.metric("Kayıp Müşteri Sayısı", f"{len(df[df['Churn']=='Yes']):,}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Müşteri Değer Matrisi")
            tenure_med = df['tenure'].median()
            charge_med = df['MonthlyCharges'].median()
            # Müşterileri segmentlere ayıran fonksiyon
            def seg_f(row):
                if row['MonthlyCharges'] >= charge_med and row['tenure'] >= tenure_med: return 'VIP'
                if row['MonthlyCharges'] >= charge_med and row['tenure'] < tenure_med: return 'Riskli Yeni'
                if row['MonthlyCharges'] < charge_med and row['tenure'] >= tenure_med: return 'Sadık Eko'
                return 'Kayıp Adayı'
            temp_s = df.copy()
            temp_s['Segment'] = temp_s.apply(seg_f, axis=1)
            seg_c = temp_s['Segment'].value_counts().reset_index()
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.barplot(data=seg_c, x='Segment', y='count', palette='viridis', ax=ax1)
            st.pyplot(fig1)
            st.caption("""
            **Grafik Analizi:** Müşterileri 'Aylık Ücret' ve 'Bağlılık Süresi'ne göre 4 ana segmente ayırır. 
            * **VIP:** Yüksek gelirli ve sadık kitle. 
            * **Riskli Yeni:** Yüksek fatura ödeyen ancak henüz şirkete alışmamış, terk ihtimali en yüksek öncelikli grup. 
            * **Sadık Eko:** Düşük ücretli ama uzun süreli bağlı kitle. 
            * **Kayıp Adayı:** Hem düşük ücretli hem de yeni olan istikrarsız grup.
            """)

        with col2:
            st.subheader("📜 Sözleşme Tipi vs Terk Oranı")
            c_tr = contract_churn.rename(index={"Month-to-month": "Aylık", "One year": "1 Yıllık", "Two year": "2 Yıllık"})
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.barplot(x=c_tr.index, y=c_tr.values, palette='magma', ax=ax2)
            st.pyplot(fig2)
            st.caption(f"""
            **Grafik Analizi:** Farklı taahhüt sürelerinin müşteri tutma başarısını ölçer. 
            Genellikle **{en_riskli_sozlesme}** tipi sözleşmelerde terk oranı çok daha yüksektir. 
            Bu durum, müşterinin finansal bir bağlayıcılığı olmadığında rakip tekliflere daha hızlı yöneldiğini kanıtlar.
            """)

        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("🔗 Ek Hizmet Sahipliği Gücü")
            h_list = ['OnlineSecurity', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'OnlineBackup']
            mevcut_h = [c for c in h_list if c in df.columns]
            temp_h = df.copy()
            # Kullanılan ek servis sayısını hesaplar
            temp_h['H_Sayisi'] = temp_h[mevcut_h].apply(lambda x: x.map({'Yes': 1, 'No': 0, 'No internet service': 0}).sum(), axis=1)
            h_anlz = temp_h.groupby('H_Sayisi')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.lineplot(x=h_anlz.index, y=h_anlz.values, marker='o', color='green', ax=ax3)
            st.pyplot(fig3)
            st.caption("""
            **Grafik Analizi:** 'Ürün Yapışkanlığı' (Product Stickiness) oranını gösterir. 
            Müşterinin kullandığı ek hizmet sayısı (Güvenlik, Destek vb.) arttıkça terk oranının nasıl düştüğünü izler. 
            3 ve üzeri hizmet kullanan müşterilerin şirketten ayrılma motivasyonu teknik ve operasyonel karmaşıklık nedeniyle azalır.
            """)

        with col4:
            st.subheader("💳 Ödeme Yöntemi Bazlı Kayıplar")
            if 'PaymentMethod' in df.columns:
                pay_tr = {"Electronic check": "E-Çek", "Mailed check": "Posta", "Bank transfer (automatic)": "Banka", "Credit card (automatic)": "K.Kartı"}
                pay_data = df[df['Churn'] == 'Yes']['PaymentMethod'].value_counts().reset_index()
                pay_data['PaymentMethod'] = pay_data['PaymentMethod'].map(pay_tr).fillna(pay_data['PaymentMethod'])
                fig4, ax4 = plt.subplots(figsize=(10, 5))
                sns.barplot(data=pay_data, y='PaymentMethod', x='count', palette='flare', ax=ax4)
                st.pyplot(fig4)
                st.caption("""
                **Grafik Analizi:** Finansal operasyonların terk üzerindeki etkisidir. 
                Otomatik ödeme (Kredi Kartı/Banka) dışındaki yöntemlerde, her ay manuel işlem yapılması müşteriye ayrılma kararını hatırlatır. 
                Özellikle E-Çek gibi yöntemlerdeki yüksek kayıp, tahsilat sorunlarına veya işlem zorluğuna işaret eder.
                """)

        st.divider()
        # Fiyat hassasiyeti yoğunluk haritası
        st.subheader("⚖️ Fatura Yoğunluğu ve Karar Sınırı")
        fig5, ax5 = plt.subplots(figsize=(20, 5))
        sns.kdeplot(data=df[df['Churn']=='Yes']['MonthlyCharges'], label="Ayrılan", fill=True, color="red", ax=ax5)
        sns.kdeplot(data=df[df['Churn']=='No']['MonthlyCharges'], label="Kalan", fill=True, color="green", ax=ax5)
        ax5.axvline(kritik_esik, color='black', linestyle='--')
        ax5.legend(); st.pyplot(fig5)
        st.caption(f"""
        **Grafik Analizi:** Fiyat hassasiyetinin yoğunluk haritasıdır. 
        Kırmızı alanın (Ayrılanlar) yeşil alanı (Kalanlar) geçmeye başladığı **{kritik_esik:.2f} $** noktası, müşterinin ödediği ücretin karşılığını sorgulamaya başladığı 'Kritik Psikolojik Eşik'tir. 
        Bu eşiğin üzerindeki müşteriler rakip tekliflere en duyarlı gruptur.
        """)

# --- TAB 3: STRATEJİK YOL HARİTASI VE ROI ---
with tab3:
    st.title("🚀 Aksiyon ve Strateji Merkezi")
    if not df.empty:
        # Finansal kayıp ve kurtarma potansiyeli hesaplamaları
        risk_gelir = df[df['Churn']=='Yes']['MonthlyCharges'].sum()
        kurtarma_orani = 0.25 # %25 başarı hedefi
        aylik_kazanc = risk_gelir * kurtarma_orani
        
        tenure_med = df['tenure'].median()
        charge_med = df['MonthlyCharges'].median()
        def quick_seg(row):
            if row['MonthlyCharges'] >= charge_med and row['tenure'] >= tenure_med: return 'VIP'
            if row['MonthlyCharges'] >= charge_med and row['tenure'] < tenure_med: return 'Riskli Yeni'
            return 'Diğer'
        
        temp_strat = df.copy()
        temp_strat['Segment'] = temp_strat.apply(quick_seg, axis=1)
        segment_risk_dağılımı = temp_strat[temp_strat['Churn']=='Yes']['Segment'].value_counts(normalize=True) * 100

        # Stratejik Finansal Hedef Metrikleri
        st.subheader("💰 Stratejik Finansal Hedefler")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Altındaki Gelir (Aylık)", f"{risk_gelir:,.0f} $")
        c2.metric("Hedeflenen Kurtarma Kazancı", f"{aylik_kazanc:,.0f} $", delta=f"%{kurtarma_orani*100:.0f} Başarı")
        c3.metric("Yıllık Potansiyel Ek Gelir", f"{aylik_kazanc*12:,.0f} $")

        st.divider()

        # Veriye dayalı otomatik aksiyon önerileri
        st.subheader("🛠️ Veriye Dayalı Kurumsal Yol Haritası")
        a1, a2 = st.columns(2)
        with a1:
            risk_orani_sozlesme = contract_churn.max()
            with st.expander(f"📌 {en_riskli_sozlesme} Sözleşme Dönüşümü"):
                 st.write(f"**Durum:** Bu gruptaki terk oranı %{risk_orani_sozlesme:.1f}. Acil 12 aylık taahhüt kampanyası başlatılmalı.")
                 st.progress(int(risk_orani_sozlesme))
            
            high_ticket_churn = (df[df['MonthlyCharges'] > kritik_esik]['Churn'] == 'Yes').mean() * 100
            with st.expander(f"📌 {kritik_esik:.2f}$ Üzeri Fatura Koruması"):
                st.write(f"**Durum:** Eşik üzerindeki müşterilerde kayıp oranı %{high_ticket_churn:.1f}. Sadakat indirimi tanımlanmalı.")
                st.progress(int(high_ticket_churn))
        
        with a2:
            riskli_yeni_pay = segment_risk_dağılımı.get('Riskli Yeni', 0)
            with st.expander("📌 'Riskli Yeni' Müşteri Operasyonu"):
                st.write(f"**Durum:** Toplam kaybın %{riskli_yeni_pay:.1f}'i bu segmentten geliyor. İlk 3 ay özel destek hattı kurulmalı.")
                st.progress(int(riskli_yeni_pay))
            
            vip_pay = segment_risk_dağılımı.get('VIP', 0)
            with st.expander("📌 VIP Kayıp Önleme Programı"):
                st.write(f"**Durum:** En değerli müşterilerin %{vip_pay:.1f}'i risk altında. Özel müşteri temsilcisi atanmalı.")
                st.progress(int(vip_pay))

        st.divider()

        # Aksiyon Öncelik Matrisi Tablosu
        st.subheader("📊 Aksiyon Önceliklendirme Matrisi")
        
        oncelik_data = {
            "Aksiyon": ["Taahhüt Kampanyası", "Teknik Destek Paketi", "Sadakat İndirimi", "VIP Ataması", "Otomatik Ödeme Teşviki"],
            "Etki": ["Yüksek", "Orta", "Yüksek", "Çok Yüksek", "Orta"],
            "Uygulama Zorluğu": ["Kolay", "Zor", "Çok Kolay", "Zor", "Kolay"],
            "Öncelik": ["⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐"]
        }
        st.table(pd.DataFrame(oncelik_data))
        
        st.info("💡 **Stratejik Not:** 'Yüksek Etki' ve 'Kolay Uygulama' olan aksiyonlar (Low-Hanging Fruit) ilk çeyrek hedeflerine alınmalıdır.")

# --- TAB 4: OPERASYONEL LİSTE VE TOPLU TARAMA ---
with tab4:
        st.divider()
        st.subheader("📋 Toplu Müşteri Risk Taraması")
        if st.button("Tüm Portföyü Tara ve Risk Raporu Oluştur"):
            with st.spinner('Analiz ediliyor...'):
                # Tüm veri seti için risk skorlarını hesaplar
                risk_scores = run_batch_prediction(df, model, features)
                df['Risk_Skoru'] = risk_scores
                
                # Riskli müşterileri süzerek listeler
                riskli_liste = df[df['Risk_Skoru'] > 0.5].sort_values(by='Risk_Skoru', ascending=False)
                display_cols = ['customerID', 'tenure', 'Contract', 'InternetService', 'TechSupport', 'PaymentMethod', 'MonthlyCharges', 'Risk_Skoru']
                report_df = riskli_liste[display_cols].copy().rename(columns=column_mapping)
                
                st.success(f"Analiz Tamamlandı! {len(riskli_liste)} yüksek riskli müşteri saptandı.")
                # Renklendirilmiş interaktif tablo
                st.dataframe(
                report_df.style.background_gradient(subset=['Terk Riski (%)'], cmap='Reds')
                .format({'Terk Riski (%)': '{:.1%}', 'Aylık Ücret ($)': '{:.2f} $'}),
                use_container_width=True
                )
                
                # Rapor indirme butonu
                csv = riskli_liste.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Kritik Risk Raporunu İndir (CSV)", csv, "risk_raporu.csv", "text/csv")