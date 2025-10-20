import streamlit as st
import joblib
import os
import re

st.set_page_config(page_title="ZeroGPT Türkçe v2", page_icon="🤖", layout="centered")

st.title("🤖 ZeroGPT Türkçe v2")
st.write("""
Bu uygulama, metnin **İnsan (0)** mı yoksa **Yapay Zekâ (1)** tarafından mı yazıldığını tahmin eder.  
""")

# ---------------------------------------------------------
# Model kontrol
# ---------------------------------------------------------
MODEL_PATH = "zeroGPT_model.pkl"
VECTORIZER_PATH = "zeroGPT_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("❌ Model veya vektörleştirici bulunamadı. Lütfen önce terminalde şu komutu çalıştır:\n\n`python zeroGPTdeneme.py --save`")
    st.stop()

def clean_text(s: str) -> str:
    """Metni temizle"""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

@st.cache_resource
def load_model():
    """Model ve vektörleştiriciyi yükle"""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()

# ---------------------------------------------------------
# Kullanıcı girişi
# ---------------------------------------------------------
text = st.text_area("📝 Metni buraya yaz:", height=200, placeholder="Örnek: Yapay zekâ sistemleri üretim verimliliğini artırıyor.")

if st.button("🚀 Analiz Et"):
    if not text.strip():
        st.warning("Lütfen bir metin girin.")
    else:
        # Metni temizle ve vektörleştir
        cleaned_text = clean_text(text)
        X = vectorizer.transform([cleaned_text])
        
        # Tahmin yap
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        conf = float(prob[pred])

        label = "🧠 **Yapay Zekâ (AI)**" if pred == 1 else "👤 **İnsan (Human)**"
        color = "#ff4b4b" if pred == 1 else "#4CAF50"

        st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
        st.progress(conf)
        st.caption(f"Güven oranı: **{conf*100:.2f}%**")
        
        # Detaylı olasılıklar
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👤 İnsan", f"%{prob[0]*100:.2f}")
        with col2:
            st.metric("🤖 Yapay Zeka", f"%{prob[1]*100:.2f}")

        st.subheader("📊 Özellik Analizi (yaklaşık):")
        st.write("""
        Bu tahmin; metnin yapısı, uzunluğu, tekrar oranı, burstiness (cümle çeşitliliği) ve kelime zenginliği gibi istatistiklerle desteklenmiştir.  
        AI yazıları genelde **düşük burstiness** ve **yüksek düzenlilik** gösterir.  
        İnsan yazıları ise **değişken cümle yapısı** ve **daha yüksek entropy** taşır.
        """)

# ---------------------------------------------------------
# Bilgilendirme
# ---------------------------------------------------------
with st.expander("ℹ️ Model Hakkında"):
    st.write("""
    - **ZeroGPT_v2** modeli, Türkçe insan ve yapay zekâ metinlerini ayırmak için özel olarak eğitilmiştir.  
    - TF-IDF özelliklerine ek olarak:
        - **Perplexity** (dil karmaşıklığı)  
        - **Burstiness** (cümle uzunluğu değişkenliği)  
        - **POS çeşitliliği** (gramer yapısı)  
        - **Entropy & Stopword oranı** gibi istatistikler kullanılmıştır.  
    - Sınıflandırıcı: **GradientBoostingClassifier**
    """)

st.markdown("---")
st.write("👨‍💻 **Yusuf Serdaroğlu – ZeroGPT Türkçe v2 Denemesi**")
