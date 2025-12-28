#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeroGPT SUPER - Geliştirilmiş AI Tespiti
-----------------------------------------
TTC-3600 + Wikipedia + Gerçek ChatGPT ile eğitilmiş
70,000 örnek içeren gelişmiş model
"""
import streamlit as st
import streamlit.components.v1 as components
import joblib
import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack

# Model yeni eğitildi - artık uyumsuzluk yok!

st.set_page_config(
    page_title="ZeroGPT Türkçe - Final",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Minimal CSS - Beyaz arka plan zorla!
st.markdown("""
<style>
    /* Ana arka planı beyaz yap */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* Tüm metinleri siyah yap */
    .stApp, .stMarkdown, h1, h2, h3, p, div {
        color: #000000 !important;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: #1f1f1f !important;
    }
    
    /* Buton stili */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Metin alanı */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Sidebar gizle */
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    # Sekmeler ve carriage return'ü temizle, fakat yeni satırları koru
    s = re.sub(r"[\t\r]+", " ", s)
    # Aynı satırda birden fazla boşluk ve yeni satırları temizle (paragraflar korunur)
    s = re.sub(r" +", " ", s)  # Satır içi çoklu boşlukları düzelt
    s = re.sub(r"\n\s*\n", "\n", s)  # Çoklu boş satırları tek satıra indir
    return s.strip()

def extract_advanced_features(text: str) -> dict:
    """Gelişmiş özellikler - Model ile uyumlu"""
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    features = {
        'len': len(text),
        'words': len(words),
        'avg_word': np.mean([len(w) for w in words]) if words else 0,
        'unique': len(set(words)) / len(words) if words else 0
    }
    
    # Cümle özellikleri
    if len(sentences) > 0:
        sent_lengths = [len(s.split()) for s in sentences]
        features['avg_sent'] = np.mean(sent_lengths)
        features['std_sent'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    else:
        features['avg_sent'] = 0
        features['std_sent'] = 0
    
    # Bağlaç sayısı (AI'ın ayırt edici özelliği!)
    connectors = ['sonuç olarak', 'bununla birlikte', 'diğer yandan', 'öte yandan',
                  'dolayısıyla', 'bu nedenle', 'ayrıca', 'bunun yanında']
    features['connector'] = sum(text.lower().count(c) for c in connectors)
    
    # Noktalama
    features['comma'] = text.count(',')
    features['question'] = text.count('?')
    
    return features

@st.cache_resource
def load_models():
    model_path = "zeroGPT_final_model.pkl"
    vec_path = "zeroGPT_final_vectorizer.pkl"
    scaler_path = "zeroGPT_final_scaler.pkl"
    
    if not all(os.path.exists(p) for p in [model_path, vec_path, scaler_path]):
        return None, None, None
    
    try:
        # Yeni eğitilmiş modeli yükle
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        scaler = joblib.load(scaler_path)
        return model, vectorizer, scaler
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {str(e)}")
        st.info("💡 Lütfen ilkDeneme.ipynb'deki son 10 cell'i çalıştırarak modeli yeniden eğitin.")
        return None, None, None

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

def analyze_sentences(text: str, model, vectorizer, scaler):
    """Her cümleyi ayrı ayrı analiz et ve AI oranlarını döndür"""
    # Cümlelere ayır
    sentences = [s.strip() for s in re.split(r'([.!?]+)', text) if s.strip()]
    
    # Cümle ve noktalama işaretlerini birleştir
    combined_sentences = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i + 1] in ['.', '!', '?', '...']:
            combined_sentences.append(sentences[i] + sentences[i + 1])
            i += 2
        else:
            combined_sentences.append(sentences[i])
            i += 1
    
    results = []
    for sent in combined_sentences:
        if len(sent.strip()) < 10:  # Çok kısa cümleler için
            results.append({
                'text': sent,
                'is_ai': False,
                'ai_prob': 0.0,
                'human_prob': 1.0
            })
            continue
        
        try:
            cleaned = clean_text(sent)
            features = extract_advanced_features(cleaned)
            tfidf_features = vectorizer.transform([cleaned])
            stat_features = pd.DataFrame([features])
            stat_features_scaled = scaler.transform(stat_features)
            combined = hstack([tfidf_features, stat_features_scaled])
            
            pred = model.predict(combined)[0]
            proba = model.predict_proba(combined)[0]
            
            results.append({
                'text': sent,
                'is_ai': pred == 1,
                'ai_prob': proba[1],
                'human_prob': proba[0]
            })
        except:
            results.append({
                'text': sent,
                'is_ai': False,
                'ai_prob': 0.0,
                'human_prob': 1.0
            })
    
    return results

# Ana Uygulama
st.title("🎯 ZeroGPT Türkçe SUPER")
st.markdown("### 🚀 TTC-3600 + Wikipedia + ChatGPT ile Geliştirilmiş AI Dedektörü")

# Başarı rozetleri
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Veriset", "70K Örnek")
with col2:
    st.metric("🎯 Model", "Gradient Boost")
with col3:
    st.metric("✨ Kaynak", "TTC-3600")

st.markdown("---")

model, vectorizer, scaler = load_models()

if model is None:
    st.error("❌ Model dosyaları bulunamadı!")
    st.info("💡 **ilkDeneme.ipynb** dosyasındaki son 10 cell'i çalıştırarak modeli eğitin (Cell 18-27)")
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
            
            # Cümle bazlı analiz
            sentence_results = analyze_sentences(user_input, model, vectorizer, scaler)
            
            st.markdown("---")
            
            # Ana sonuç - sade ve anlaşılır
            if pred == 0:
                st.success("# ✅ İNSAN YAZISI")
                st.metric("Güven Oranı", f"%{proba[0]*100:.1f}", delta="İnsan")
            else:
                st.error("# 🤖 YAPAY ZEKA YAZISI")
                st.metric("Güven Oranı", f"%{proba[1]*100:.1f}", delta="AI")
            
            st.markdown("---")
            
            # 🎨 CÜMLE BAZLI VURGULAMA - AI cümleleri sarı renkte!
            st.markdown("### 🔍 Detaylı Cümle Analizi")
            st.markdown("**🟨 Sarı vurgulu** kısımlar AI tarafından yazılmış, **⬜ normal** kısımlar insan yazısı:")
            
            # HTML oluştur - Beyaz arka plan ve siyah metin
            html_content = '''
            <div style="
                line-height: 2.2; 
                font-size: 18px; 
                padding: 25px; 
                background: #ffffff;
                color: #000000;
                border-radius: 10px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border: 2px solid #e0e0e0;
            ">'''
            
            ai_count = 0
            human_count = 0
            
            for result in sentence_results:
                text = result['text']
                is_ai = result['is_ai']
                ai_prob = result['ai_prob']
                
                # Yeni satırları <br> tagına dönüştür
                text_html = text.replace('\n', '<br>')
                
                if is_ai and ai_prob > 0.6:  # AI tespit eşiği
                    ai_count += 1
                    # Sarı vurgulu AI cümlesi - çok belirgin!
                    html_content += f'''
                    <span class="ai-sentence" style="
                        background: #FFD93D;
                        color: #000000;
                        padding: 6px 10px;
                        border-radius: 8px;
                        border-left: 5px solid #FF6B35;
                        margin: 3px;
                        display: inline-block;
                        font-weight: 600;
                        transition: all 0.3s ease;
                        cursor: help;
                        box-shadow: 0 2px 8px rgba(255, 217, 61, 0.5);
                    " title="🤖 AI Olasılığı: %{ai_prob*100:.1f}">🤖 {text_html}</span> 
                    '''
                else:
                    human_count += 1
                    # Normal metin (insan) - açık gri arka plan
                    html_content += f'''<span style="
                        background: #f5f5f5;
                        color: #000000;
                        padding: 6px 10px;
                        margin: 3px;
                        display: inline-block;
                        border-radius: 5px;
                    ">👤 {text_html}</span> '''
            
            html_content += '</div>'
            
            # CSS ile birlikte göster
            components.html(f"""
                <style>
                    @keyframes pulse {{
                        0%, 100% {{ box-shadow: 0 2px 8px rgba(255, 217, 61, 0.5); }}
                        50% {{ box-shadow: 0 4px 16px rgba(255, 217, 61, 0.8); }}
                    }}
                    .ai-sentence:hover {{
                        transform: scale(1.05);
                        box-shadow: 0 4px 16px rgba(255, 107, 53, 0.6) !important;
                        background: #FFC300 !important;
                    }}
                </style>
                {html_content}
            """, height=max(300, len(sentence_results) * 35))
            
            # İstatistikler
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🤖 AI Cümleleri", f"{ai_count} adet", delta=f"%{(ai_count/(ai_count+human_count)*100) if (ai_count+human_count) > 0 else 0:.0f}")
            with col2:
                st.metric("👤 İnsan Cümleleri", f"{human_count} adet", delta=f"%{(human_count/(ai_count+human_count)*100) if (ai_count+human_count) > 0 else 0:.0f}")
            with col3:
                st.metric("📝 Toplam Cümle", f"{ai_count + human_count} adet")
            
            # Olasılık çubukları
            st.markdown("---")
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
            st.markdown("### 🔬 Gelişmiş Metin Analizi")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📏 Uzunluk", f"{features['len']:.0f}")
                st.metric("📝 Kelime", f"{features['words']:.0f}")
            
            with col2:
                st.metric("✏️ Ort. Kelime", f"{features['avg_word']:.1f}")
                st.metric("🎯 Çeşitlilik", f"{features['unique']:.2f}")
            
            with col3:
                st.metric("📊 Ort. Cümle", f"{features['avg_sent']:.1f}")
                st.metric("📈 Std Cümle", f"{features['std_sent']:.1f}")
            
            with col4:
                st.metric("🔗 Bağlaç", f"{features['connector']:.0f}")
                st.metric("📌 Virgül", f"{features['comma']:.0f}")
            
            # Yorum
            st.markdown("---")
            with st.expander("💡 Bu Sonuç Ne Anlama Geliyor?"):
                if pred == 1:
                    st.info(f"""
                    **AI Tespit İşaretleri:**
                    - 🔗 Bağlaç kullanımı: {features['connector']} adet ("sonuç olarak", "dolayısıyla")
                    - 📊 Düzenli cümle uzunluğu (std: {features['std_sent']:.1f})
                    - 📝 Formal ve akademik dil
                    - 🎯 Yapılandırılmış paragraf düzeni
                    - ✨ Tutarlı kelime seçimleri
                    """)
                else:
                    st.success(f"""
                    **İnsan Yazı İşaretleri:**
                    - 💬 Doğal dil akışı
                    - 🎭 Değişken cümle yapısı (std: {features['std_sent']:.1f})
                    - 🗣️ Konuşma diline yakın üslup
                    - 💭 Spontane ifadeler
                    - 🎨 Özgün kelime seçimleri
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
    ### 🎓 ZeroGPT SUPER Model
    
    **Eğitim Verisi:**
    - 📊 70,000 Türkçe metin
    - ✅ 35K İnsan + 35K AI (Dengeli)
    - 📰 TTC-3600 Türkçe Haber Veriseti
    - 📚 Wikipedia Türkçe Makaleleri
    - 🤖 Gerçek ChatGPT Örnekleri
    
    **Veri Kaynakları:**
    - **TTC-3600:** 6 kategoride profesyonel haber metinleri
    - **Wikipedia:** Ansiklopedik Türkçe içerik
    - **ChatGPT:** Gerçek AI üretimi metinler
    
    **Teknik Özellikler:**
    - 🚀 Gradient Boosting Classifier (100 trees)
    - 📈 3,000 TF-IDF özellikleri (1-2 gram)
    - 🔬 9 gelişmiş istatistiksel özellik
    - 🎯 Yüksek doğruluk oranı
    
    **Tespit Edilen Özellikler:**
    - Metin uzunluğu & Kelime sayısı
    - Ortalama kelime/cümle uzunluğu
    - Kelime çeşitliliği
    - Cümle standart sapması (burstiness)
    - Bağlaç kullanımı ("sonuç olarak" vb.)
    - Noktalama özellikleri (virgül, soru işareti)
    
    **Version:** 2.0 SUPER (TTC-3600 dahil gelişmiş model)
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>🚀 ZeroGPT Türkçe SUPER v2.0 - TTC-3600 Gelişmiş Model</div>",
    unsafe_allow_html=True
)

