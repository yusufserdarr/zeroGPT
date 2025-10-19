import streamlit as st
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# Başlık
# ---------------------------
st.set_page_config(page_title="ZeroGPT Türkçe", page_icon="🤖", layout="centered")
st.title("🤖 ZeroGPT Türkçe Tespit Aracı")
st.write("Bu uygulama, bir metnin **yapay zekâ (AI)** veya **insan** tarafından yazılıp yazılmadığını tahmin eder.")

# ---------------------------
# Modeli yükle
# ---------------------------
MODEL_PATH = "zeroGPT_model.pkl"
VECTORIZER_PATH = "zeroGPT_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("❌ Model veya vektörleştirici bulunamadı. Lütfen önce `zeroGPTdeneme.py --save` komutunu çalıştır.")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ---------------------------
# Kullanıcı girişi
# ---------------------------
text_input = st.text_area("📝 Metni buraya yaz:", height=200, placeholder="Örnek: Yapay zekâ sistemleri üretim verimliliğini artırıyor.")

if st.button("🚀 Tahmin Et"):
    if not text_input.strip():
        st.warning("Lütfen bir metin girin.")
    else:
        X = vectorizer.transform([text_input])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][pred]

        label = "🧠 **Yapay Zekâ (AI)**" if pred == 1 else "👤 **İnsan (Human)**"
        color = "red" if pred == 1 else "green"

        st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
        st.progress(prob)
        st.caption(f"Güven oranı: **{prob*100:.2f}%**")

# ---------------------------
# Bilgi alanı
# ---------------------------
with st.expander("ℹ️ Nasıl Çalışıyor?"):
    st.write("""
    Bu model, 70.000'den fazla Türkçe metin üzerinde eğitilmiş TF-IDF + Logistic Regression tabanlı bir sınıflandırıcıdır.  
    Her metindeki kelime dağılımı, bağlaç yapısı ve anlamsal yoğunluğa göre karar verir.  
    - **0:** İnsan tarafından yazılmış  
    - **1:** Yapay zekâ tarafından yazılmış
    """)

st.markdown("---")
st.write("🧩 [Yusuf Serdar • ZeroGPT Türkçe Deneme]")