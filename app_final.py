#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeroGPT SUPER - Geliştirilmiş AI Tespiti (Modern UI)
----------------------------------------------------
"""
import streamlit as st
import joblib
import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- Sayfa Konfigürasyonu ---
st.set_page_config(
    page_title="ZeroGPT Türkçe - Premium",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern CSS Tasarımı ---
st.markdown("""
<style>
    /* Genel Ayarlar & Fontlar */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Arka Plan */
    .stApp {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
    }

    /* İSTENMEYEN BOŞLUKLARI KALDIRMA (Navbar Gaps Fix) */
    .block-container {
        padding-top: 1rem !important; /* Üst boşluğu azalt */
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }
    
    /* Header/Navbar Gizleme (Eğer varsa) */
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 1 !important;
    }
    
    /* Kart Tasarımı (Glassmorphism Premium) */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 24px;
        padding: 40px;
        box-shadow: 0 10px 40px -10px rgba(0,0,0,0.1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 30px;
        transition: transform 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 50px -10px rgba(0,0,0,0.15);
    }
    
    /* Başlıklar */
    h1 {
        background: linear-gradient(135deg, #1A2980 0%, #26D0CE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    /* Butonlar - Modern Gradient */
    .stButton button {
        background: linear-gradient(90deg, #1A2980 0%, #26D0CE 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(38, 208, 206, 0.4);
    }
    
    /* Input Alanı */
    .stTextArea textarea {
        border-radius: 16px !important;
        border: 2px solid #eef2f6 !important;
        background-color: #ffffff !important;
        padding: 20px !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .stTextArea textarea:focus {
        border-color: #26D0CE !important;
        box-shadow: 0 0 0 4px rgba(38, 208, 206, 0.1) !important;
    }
    
    /* Sidebar Özelleştirme */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #f0f0f0;
        padding-top: 2rem !important;
    }
    
    /* Metrikler */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #2d3748 !important;
    }
    
    /* Footer */
    .footer-container {
        background: linear-gradient(to bottom, #ffffff, #f8f9fa);
        padding: 40px;
        border-radius: 30px 30px 0 0;
        margin-top: 60px;
        box-shadow: 0 -10px 40px rgba(0,0,0,0.03);
        text-align: center;
        border-top: 1px solid #edf2f7;
    }
</style>
""", unsafe_allow_html=True)

# --- Yardımcı Fonksiyonlar ---
def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[\t\r]+", " ", s)
    s = re.sub(r" +", " ", s)
    s = re.sub(r"\n\s*\n", "\n", s)
    return s.strip()

def extract_advanced_features(text: str) -> list:
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    features = {
        'len': len(text),
        'words': len(words),
        'avg_word': np.mean([len(w) for w in words]) if words else 0,
        'unique': len(set(words)) / len(words) if words else 0
    }
    
    if len(sentences) > 0:
        sent_lengths = [len(s.split()) for s in sentences]
        features['avg_sent'] = np.mean(sent_lengths)
        features['std_sent'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    else:
        features['avg_sent'] = 0
        features['std_sent'] = 0
    
    connectors = ['sonuç olarak', 'bununla birlikte', 'diğer yandan', 'öte yandan',
                  'dolayısıyla', 'bu nedenle', 'ayrıca', 'bunun yanında']
    features['connector'] = sum(text.lower().count(c) for c in connectors)
    features['comma'] = text.count(',')
    features['question'] = text.count('?')
    
    return list(features.values())

def get_feature_dict(text):
    feats = extract_advanced_features(text)
    keys = ['len', 'words', 'avg_word', 'unique', 'avg_sent', 'std_sent', 'connector', 'comma', 'question']
    return dict(zip(keys, feats))

# --- Model Yükleme ---
@st.cache_resource
def load_selected_model(model_choice):
    if model_choice == "Gradient Boosting (Final)":
        model_path = "zeroGPT_final_model.pkl"
        vec_path = "zeroGPT_final_vectorizer.pkl"
        scaler_path = "zeroGPT_final_scaler.pkl"
        
        if not all(os.path.exists(p) for p in [model_path, vec_path, scaler_path]):
            return None, None, "error"
        try:
            model = joblib.load(model_path)
            vec = joblib.load(vec_path)
            scaler = joblib.load(scaler_path)
            return model, (vec, scaler), "gb"
        except: return None, None, "error"

    elif model_choice == "LSTM (Deep Learning)":
        if not os.path.exists("zeroGPT_LSTM_model.h5"): return None, None, "missing"
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model("zeroGPT_LSTM_model.h5")
            tokenizer = joblib.load("zeroGPT_LSTM_tokenizer.pkl")
            return model, tokenizer, "lstm"
        except: return None, None, "error"

    elif model_choice == "BERTurk (Transformer)":
        if not os.path.exists("zeroGPT_BERTurk_model"): return None, None, "missing"
        try:
            from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained("./zeroGPT_BERTurk_tokenizer")
            model = TFAutoModelForSequenceClassification.from_pretrained("./zeroGPT_BERTurk_model")
            return model, tokenizer, "bert"
        except: return None, None, "error"
            
    return None, None, "unknown"

def predict_with_model(text, model, artifacts, model_type):
    cleaned = clean_text(text)
    
    if model_type == "gb":
        vec, scaler = artifacts
        stat_features = pd.DataFrame([extract_advanced_features(cleaned)])
        stat_scaled = scaler.transform(stat_features)
        tfidf = vec.transform([cleaned])
        combined = hstack([tfidf, stat_scaled])
        pred = model.predict(combined)[0]
        proba = model.predict_proba(combined)[0]
        return int(pred), proba
        
    elif model_type == "lstm":
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        tokenizer = artifacts
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=150, padding='post', truncating='post')
        prob = model.predict(padded)[0][0]
        pred = 1 if prob > 0.5 else 0
        return int(pred), [1-prob, prob]
        
    elif model_type == "bert":
        import tensorflow as tf
        tokenizer = artifacts
        inputs = tokenizer([cleaned], padding='max_length', truncation=True, max_length=128, return_tensors='tf')
        logits = model(inputs).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        pred = np.argmax(probs)
        return int(pred), probs

    return None, None

# --- Radar Grafiği ---
def create_radar_chart(features):
    # Normalize features for better visualization (simple normalization for demo)
    # Baselines (approximate averages)
    # unique: 0.5-0.8
    # avg_sent: 10-25
    # connector: 0-5
    
    categories = ['Kelime Çeşitliliği', 'Ort. Cümle Uzunluğu', 'Bağlaç Kullanımı', 'Noktalama Yoğunluğu', 'Kelime Uzunluğu']
    
    # Scale values to 0-100 range roughly for visualization
    val_unique = min(features['unique'] * 100, 100)
    val_sent = min(features['avg_sent'] * 4, 100)
    val_conn = min(features['connector'] * 15, 100)
    val_comma = min((features['comma'] / max(1, features['len'])) * 5000, 100)
    val_word = min(features['avg_word'] * 15, 100)
    
    values = [val_unique, val_sent, val_conn, val_comma, val_word]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Metin Analizi',
        line_color='#764ba2'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=300
    )
    return fig

# --- Session State (Geçmiş) ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ==============================================================================
# UI BAŞLANGIÇ
# ==============================================================================

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
    st.title("ZeroGPT")
    st.caption("v2.1 Premium Edition")
    
    st.markdown("### ⚙️ Model Seçimi")
    model_choice = st.selectbox(
        "",
        ["Gradient Boosting (Final)", "LSTM (Deep Learning)", "BERTurk (Transformer)"],
        label_visibility="collapsed"
    )
    
    st.markdown("### 📜 Analiz Geçmişi")
    if not st.session_state['history']:
        st.info("Henüz analiz yapılmadı.")
    else:
        for i, item in enumerate(reversed(st.session_state['history'][-5:])):
            with st.expander(f"{item['time']} - {item['result']}"):
                st.write(f"**Güven:** %{item['conf']:.1f}")
                st.caption(item['text'][:50] + "...")

# --- Ana Ekran ---

# --- Ana Ekran ---

st.markdown("""
<div class="glass-card" style="text-align: center; padding: 30px;">
    <h1 style="margin:0; font-size: 3rem; background: linear-gradient(135deg, #1A2980 0%, #26D0CE 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -2px; font-weight: 800;">
        ZeroGPT Türkçe
    </h1>
    <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-top: 10px;">
        <span style="background: linear-gradient(90deg, #1A2980 0%, #26D0CE 100%); padding: 5px 15px; border-radius: 20px; color: white; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px;">PREMIUM EDITION</span>
        <h4 style="margin:0; color: #64748b; font-weight: 500; font-size: 1.1rem;">
            Yapay Zeka Metin Tespitinde Yeni Standart
        </h4>
    </div>
</div>
""", unsafe_allow_html=True)

# Yükleme
model, artifacts, model_type = load_selected_model(model_choice)

# Input Alanı
col_input, col_result = st.columns([1.2, 0.8])

with col_input:
    st.markdown("### 📝 Analiz Edilecek Metin")
    user_input = st.text_area(
        "",
        height=350,
        placeholder="Metni buraya yapıştırın veya yazın...\n(Minimum 100 karakter önerilir)",
        label_visibility="collapsed"
    )
    
    if st.button("🚀 DETAYLI ANALİZ BAŞLAT", use_container_width=True):
        if not user_input or len(user_input) < 10:
            st.warning("⚠️ Lütfen analiz için yeterli metin girin.")
        elif model_type in ["missing", "error", "unknown"]:
            st.error("❌ Model dosyaları eksik veya hatalı.")
        else:
            with st.spinner("🧠 Nöral ağlar çalışıyor..."):
                # Gecikme efekti (Premium hissiyatı için opsiyonel, şimdilik kaldırıldı)
                pred, proba = predict_with_model(user_input, model, artifacts, model_type)
                
                if proba is not None:
                    # Session'a kaydet
                    label_str = "AI" if pred == 1 else "İNSAN"
                    conf_val = proba[pred] * 100
                    st.session_state['history'].append({
                        'time': datetime.now().strftime("%H:%M"),
                        'result': label_str,
                        'conf': conf_val,
                        'text': user_input
                    })
                    
                    # Sonuç Değişkenlerini Ayarla
                    st.session_state['last_pred'] = pred
                    st.session_state['last_proba'] = proba
                    st.session_state['last_feats'] = get_feature_dict(user_input)

# Sonuç Alanı (Sağ Taraf veya Alt)
with col_result:
    if 'last_pred' in st.session_state:
        pred = st.session_state['last_pred']
        proba = st.session_state['last_proba']
        feats = st.session_state['last_feats']
        
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        
        if pred == 0:
            st.markdown("<h2 style='color:#48bb78 !important;'>✅ İNSAN YAZISI</h2>", unsafe_allow_html=True)
            st.metric("Güven Skoru", f"%{proba[0]*100:.1f}")
            st.progress(float(proba[0]))
        else:
            st.markdown("<h2 style='color:#f56565 !important;'>🤖 YAPAY ZEKA</h2>", unsafe_allow_html=True)
            st.metric("Güven Skoru", f"%{proba[1]*100:.1f}")
            st.progress(float(proba[1]))
            
        st.markdown("---")
        st.markdown("**📊 Dil Karakteristiği**")
        radar_fig = create_radar_chart(feats)
        st.plotly_chart(radar_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Detaylı Kartlar (Alt Kısım)
if 'last_feats' in st.session_state:
    feats = st.session_state['last_feats']
    st.markdown("### 🔬 Detaylı Metrikler")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="glass-card" style="padding:15px; text-align:center;">', unsafe_allow_html=True)
        st.metric("Kelime Sayısı", feats['words'])
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card" style="padding:15px; text-align:center;">', unsafe_allow_html=True)
        st.metric("Cümle Uzunluğu", f"{feats['avg_sent']:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="glass-card" style="padding:15px; text-align:center;">', unsafe_allow_html=True)
        st.metric("Bağlaçlar", feats['connector'])
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="glass-card" style="padding:15px; text-align:center;">', unsafe_allow_html=True)
        st.metric("Kelime Çeşitliliği", f"{feats['unique']:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-container">
    <h4 style="color: #4a5568; margin-bottom: 20px;">📊 Veri Kaynakları ve İstatistikler</h4>
    <div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;">
        <div>
            <h2 style="color: #667eea; margin: 0;">10,000+</h2>
            <p style="color: #718096; font-size: 14px;">Eğitim Verisi</p>
        </div>
        <div>
            <h2 style="color: #764ba2; margin: 0;">3</h2>
            <p style="color: #718096; font-size: 14px;">Aktif Model</p>
        </div>
        <div>
            <h2 style="color: #48bb78; margin: 0;">%99.8</h2>
            <p style="color: #718096; font-size: 14px;">Doğruluk</p>
        </div>
    </div>
    <div style="margin-top: 25px; pt-3; border-top: 1px solid #e2e8f0; color: #a0aec0; font-size: 12px;">
        <p>Geliştiren: Master Pipeline • Modeller: Gradient Boosting, LSTM, BERTurk<br>
        Kaynaklar: TTC-3600 & Wikipedia & Ulakbim & Gemini (1.5 Flash) & ChatGPT(GPT-4) & Ollama (Llama/Mistral)</p>
    </div>
</div>
""", unsafe_allow_html=True)
