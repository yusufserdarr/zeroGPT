#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeroGPT Final - Gerçek ChatGPT Tespiti
--------------------------------------
Gerçek ChatGPT metinleriyle eğitilmiş final model
"""
import streamlit as st
import joblib
import os
import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import hstack

st.set_page_config(
    page_title="ZeroGPT Türkçe - Final",
    page_icon="🎯",
    layout="centered"
)

TR_STOPWORDS = {
    "ve","veya","ile","ama","fakat","ancak","çünkü","gibi","daha","çok",
    "az","bu","şu","o","bir","iki","üç","her","hiç","mi","mı","mu","mü",
    "de","da","ki","için","üzere","olan","olarak","sonra","önce","ise",
    "ya","ne","neden","nasıl","hangi","hem","en","ben","sen","biz","siz",
    "onlar","var","yok","diye","kadar","beri","göre","rağmen","dolayı"
}

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_advanced_features(text: str) -> dict:
    features = {}
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    features['sentence_count'] = len(sentences)
    
    if len(sentences) > 0:
        sent_lengths = [len(s.split()) for s in sentences]
        features['avg_sentence_length'] = np.mean(sent_lengths)
        features['std_sentence_length'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        features['burstiness'] = features['std_sentence_length'] / (features['avg_sentence_length'] + 1e-5)
    else:
        features['avg_sentence_length'] = 0
        features['std_sentence_length'] = 0
        features['burstiness'] = 0
    
    words = re.findall(r'\w+', text.lower(), flags=re.UNICODE)
    if len(words) > 0:
        features['lexical_diversity'] = len(set(words)) / len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words])
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        features['max_word_repetition'] = most_common_count / len(words)
        stopword_count = sum(1 for w in words if w in TR_STOPWORDS)
        features['stopword_ratio'] = stopword_count / len(words)
        digit_words = sum(1 for w in words if any(c.isdigit() for c in w))
        features['digit_word_ratio'] = digit_words / len(words)
    else:
        features['lexical_diversity'] = 0
        features['avg_word_length'] = 0
        features['max_word_repetition'] = 0
        features['stopword_ratio'] = 0
        features['digit_word_ratio'] = 0
    
    if len(text) > 0:
        char_counts = Counter(text.lower())
        total = sum(char_counts.values())
        probs = [c/total for c in char_counts.values()]
        features['char_entropy'] = -sum(p * math.log2(p + 1e-12) for p in probs)
    else:
        features['char_entropy'] = 0
    
    features['comma_count'] = text.count(',')
    features['question_mark_count'] = text.count('?')
    features['exclamation_count'] = text.count('!')
    features['quote_count'] = text.count('"') + text.count("'")
    
    if len(text) > 0:
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
    else:
        features['uppercase_ratio'] = 0
    
    connectors = ['sonuç olarak', 'bununla birlikte', 'diğer yandan', 'öte yandan', 
                  'kısacası', 'dolayısıyla', 'bu nedenle', 'ayrıca', 'bunun yanında']
    features['connector_count'] = sum(text.lower().count(c) for c in connectors)
    
    passive_markers = ['edilmek', 'yapılmak', 'olmak', 'görülmek', 'düşünülmek']
    features['passive_marker_count'] = sum(text.lower().count(p) for p in passive_markers)
    
    return features

@st.cache_resource
def load_models():
    model_path = "zeroGPT_final_model.pkl"
    vec_path = "zeroGPT_final_vectorizer.pkl"
    scaler_path = "zeroGPT_final_scaler.pkl"
    
    if not all(os.path.exists(p) for p in [model_path, vec_path, scaler_path]):
        return None, None, None
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    scaler = joblib.load(scaler_path)
    return model, vectorizer, scaler

def predict_text(text: str, model, vectorizer, scaler):
    """Metin tahmini yap - Hata yönetimi ile korumalı"""
    try:
        cleaned = clean_text(text)
        
        # Özellikleri bir kez hesapla (performans iyileştirmesi)
        features = extract_advanced_features(cleaned)
        
        # TF-IDF özellikleri
        tfidf_features = vectorizer.transform([cleaned])
        
        # İstatistiksel özellikleri kullan (tekrar hesaplama yok!)
        stat_features = pd.DataFrame([features])
        stat_features_scaled = scaler.transform(stat_features)
        
        # Birleştir ve tahmin yap
        combined = hstack([tfidf_features, stat_features_scaled])
        pred = model.predict(combined)[0]
        proba = model.predict_proba(combined)[0]
        
        return pred, proba, features
        
    except MemoryError:
        st.error("❌ **Metin çok uzun!** Lütfen daha kısa bir metin deneyin (maksimum ~5000 kelime).")
        return None, None, None
        
    except ValueError as e:
        st.error(f"❌ **Geçersiz veri hatası:** {str(e)}")
        st.info("💡 Lütfen metninizin düzgün Türkçe karakterler içerdiğinden emin olun.")
        return None, None, None
        
    except Exception as e:
        st.error(f"❌ **Beklenmeyen hata:** {str(e)}")
        st.warning("⚠️ Lütfen farklı bir metin deneyin veya sayfayı yenileyin.")
        return None, None, None

# Ana Uygulama
st.title("🎯 ZeroGPT Türkçe - Final Versiyon")
st.markdown("### Gerçek ChatGPT Metinleriyle Eğitilmiş AI Dedektörü")

# Başarı rozetleri
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("✅ Accuracy", "99.98%")
with col2:
    st.metric("🎯 Model", "Gradient Boost")
with col3:
    st.metric("📊 Özellikler", "15K+ TF-IDF")

st.markdown("---")

model, vectorizer, scaler = load_models()

if model is None:
    st.error("❌ Final model dosyaları bulunamadı!")
    st.info("Lütfen önce modeli eğitin: `python3 retrain_with_real_chatgpt.py`")
    st.stop()

# Metin girişi
user_input = st.text_area(
    "📝 Analiz edilecek metni girin:",
    height=200,
    placeholder="Örnek: Yapay zeka teknolojileri günümüzde birçok sektörde kullanılıyor..."
)

if st.button("🚀 Analiz Et", type="primary", use_container_width=True):
    if not user_input or len(user_input.strip()) < 20:
        st.warning("⚠️ Lütfen en az 20 karakter uzunluğunda bir metin girin.")
    elif len(user_input.strip().split()) < 5:
        st.warning("⚠️ Lütfen en az 5 kelime içeren bir metin girin.")
    elif len(user_input.strip().split()) > 5000:
        st.warning("⚠️ Metin çok uzun! Maksimum 5000 kelime girebilirsiniz.")
    else:
        with st.spinner("🔍 Detaylı analiz yapılıyor..."):
            pred, proba, features = predict_text(user_input, model, vectorizer, scaler)
            
            # Hata kontrolü - predict_text None döndürdüyse dur
            if pred is None or proba is None or features is None:
                st.stop()  # Hata mesajı zaten gösterildi, devam etme
            
            st.markdown("---")
            
            # Ana sonuç - büyük ve belirgin
            if pred == 0:
                st.success("# ✅ İNSAN YAZISI")
                st.metric("Güven Oranı", f"%{proba[0]*100:.1f}", delta="İnsan")
            else:
                st.error("# 🤖 YAPAY ZEKA YAZISI")
                st.metric("Güven Oranı", f"%{proba[1]*100:.1f}", delta="AI")
            
            # Olasılık çubukları
            st.markdown("### 📊 Detaylı Olasılıklar")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**👤 İnsan**")
                st.progress(proba[0])
                st.write(f"**%{proba[0]*100:.2f}**")
            
            with col2:
                st.markdown("**🤖 Yapay Zeka**")
                st.progress(proba[1])
                st.write(f"**%{proba[1]*100:.2f}**")
            
            # Özellik analizi
            st.markdown("---")
            st.markdown("### 🔬 Özellik Analizi")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📏 Kelime", f"{features['word_count']:.0f}")
                st.metric("📝 Cümle", f"{features['sentence_count']:.0f}")
            
            with col2:
                st.metric("🎯 Çeşitlilik", f"{features['lexical_diversity']:.2f}")
                st.metric("💥 Burstiness", f"{features['burstiness']:.2f}")
            
            with col3:
                st.metric("🔢 Entropi", f"{features['char_entropy']:.2f}")
                st.metric("🔗 Bağlaç", f"{features['connector_count']:.0f}")
            
            with col4:
                st.metric("📌 Stopword", f"{features['stopword_ratio']:.2f}")
                st.metric("❓ Soru", f"{features['question_mark_count']:.0f}")
            
            # Yorum
            st.markdown("---")
            with st.expander("💡 Bu Sonuç Ne Anlama Geliyor?"):
                if pred == 1:
                    st.info("""
                    **AI Tespit İşaretleri:**
                    - Düzenli ve tutarlı cümle yapısı (düşük burstiness)
                    - Formal ve akademik dil kullanımı
                    - Bağlaç yoğunluğu ("sonuç olarak", "bununla birlikte")
                    - Yapılandırılmış paragraf düzeni
                    - Tekrarlayan kalıplar ve kelime seçimleri
                    """)
                else:
                    st.success("""
                    **İnsan Yazı İşaretleri:**
                    - Değişken cümle uzunlukları (yüksek burstiness)
                    - Doğal dil akışı ve spontane ifadeler
                    - Kişisel vurgular ve ünlemler
                    - Konuşma diline yakın üslup
                    - Özgün kelime seçimleri
                    """)

# Örnek metinler
st.markdown("---")
with st.expander("📚 Örnek Metinlerle Test Edin"):
    st.markdown("**AI Metni:**")
    st.code("""Teknoloji, insanlığın ilerlemesini hızlandıran en güçlü araçlardan biridir. 
Günümüzde yapay zekâ, otomasyon ve büyük veri analizleri sayesinde üretimden 
sağlığa kadar her alanda verimlilik artışı sağlanmaktadır.""")
    
    st.markdown("**İnsan Metni:**")
    st.code("""Bugün arkadaşımla sahilde yürüdük, hava çok güzeldi yaa! 
Deniz kenarında oturup sohbet ettik. Akşam da güzel bir yemek yedik.""")

# Model bilgisi
st.markdown("---")
with st.expander("ℹ️ Model Hakkında"):
    st.markdown("""
    ### 🎓 ZeroGPT Final Model
    
    **Eğitim Verisi:**
    - 📊 20,500+ Türkçe metin
    - ✅ Gerçek ChatGPT örnekleri dahil
    - 🔄 Dengeli veri seti (%50 AI, %50 İnsan)
    
    **Teknik Özellikler:**
    - 🚀 Gradient Boosting Classifier (200 trees)
    - 📈 15,000 TF-IDF özellikleri (1-3 gram)
    - 🔬 20+ istatistiksel özellik
    - 🎯 %99.98 test accuracy
    
    **Tespit Edilen Özellikler:**
    - Burstiness (cümle varyasyonu)
    - Lexical Diversity (kelime çeşitliliği)
    - Character Entropy (bilgi yoğunluğu)
    - Connector Usage (bağlaç kullanımı)
    - Syntactic Patterns (sözdizimi kalıpları)
    
    **Version:** 1.0 Final (Gerçek ChatGPT ile eğitilmiş)
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>🎯 ZeroGPT Türkçe Final v1.0 - Gerçek ChatGPT Dedektörü</div>",
    unsafe_allow_html=True
)

