import streamlit as st
import joblib
import os
import pandas as pd

st.set_page_config(page_title="ZeroGPT Türkçe v2", page_icon="🤖", layout="centered")

st.title("🤖 ZeroGPT Türkçe v2")
st.write("""
Bu uygulama, metnin **İnsan (0)** mı yoksa **Yapay Zekâ (1)** tarafından mı yazıldığını tahmin eder.  
Yeni sürüm; **perplexity, burstiness, grammar ve kelime çeşitliliği** gibi özellikleri de dikkate alır.
""")

# ---------------------------------------------------------
# Model kontrol
# ---------------------------------------------------------
PIPE_PATH = "zeroGPT_model.pkl"

if not os.path.exists(PIPE_PATH):
    st.error("❌ Model dosyası bulunamadı. Lütfen önce terminalde şu komutu çalıştır:\n\n`python3 zeroGPT_v2.py --save`")
    st.stop()

pipe = joblib.load(PIPE_PATH)

# ---------------------------------------------------------
# Kullanıcı girişi
# ---------------------------------------------------------
text = st.text_area("📝 Metni buraya yaz:", height=200, placeholder="Örnek: Yapay zekâ sistemleri üretim verimliliğini artırıyor.")

if st.button("🚀 Analiz Et"):
    if not text.strip():
        st.warning("Lütfen bir metin girin.")
    else:
        X = pd.DataFrame({"content": [text]})
        prob = pipe.predict_proba(X)[0]
        pred = int(prob.argmax())
        conf = float(prob[pred])

        label = "🧠 **Yapay Zekâ (AI)**" if pred == 1 else "👤 **İnsan (Human)**"
        color = "#ff4b4b" if pred == 1 else "#4CAF50"

        st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
        st.progress(conf)
        st.caption(f"Güven oranı: **{conf*100:.2f}%**")

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
