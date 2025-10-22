#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeroGPT Web API - Flask Backend
Modern web arayüzü için API
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app)

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
    # Sekmeler ve carriage return'ü temizle, fakat yeni satırları koru
    s = re.sub(r"[\t\r]+", " ", s)
    # Aynı satırda birden fazla boşluk ve yeni satırları temizle (paragraflar korunur)
    s = re.sub(r" +", " ", s)  # Satır içi çoklu boşlukları düzelt
    s = re.sub(r"\n\s*\n", "\n", s)  # Çoklu boş satırları tek satıra indir
    return s.strip()

def clean_sentence(s: str) -> str:
    """Cümle analizi için - TÜM boşlukları kaldır"""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[\t\r\n]+", " ", s)  # Newline'ları DA kaldır!
    s = re.sub(r" +", " ", s)
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

# Model yükleme
model = joblib.load("zeroGPT_final_model.pkl")
vectorizer = joblib.load("zeroGPT_final_vectorizer.pkl")
scaler = joblib.load("zeroGPT_final_scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text or len(text.strip()) < 20:
            return jsonify({'error': 'Metin çok kısa'}), 400
        
        # Genel analiz
        cleaned = clean_text(text)
        features = extract_advanced_features(cleaned)
        tfidf_features = vectorizer.transform([cleaned])
        stat_features = pd.DataFrame([features])
        stat_features_scaled = scaler.transform(stat_features)
        combined = hstack([tfidf_features, stat_features_scaled])
        
        pred = model.predict(combined)[0]
        proba = model.predict_proba(combined)[0]
        
        # Cümle bazlı analiz - Her cümleyi GERÇEKTEN analiz et
        sentences = [s.strip() for s in re.split(r'([.!?]+)', text) if s.strip()]
        combined_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1] in ['.', '!', '?', '...']:
                combined_sentences.append(sentences[i] + sentences[i + 1])
                i += 2
            else:
                combined_sentences.append(sentences[i])
                i += 1
        
        sentence_results = []
        errors_log = []
        
        # Genel sonucu baz al - eşik değeri ayarla
        overall_ai_prob = float(proba[1])
        # Eğer genel %70+ AI ise, cümlelerde eşiği düşür
        sentence_threshold = 0.3 if overall_ai_prob > 0.7 else 0.5
        
        for idx, sent in enumerate(combined_sentences):
            if len(sent.strip()) < 10:
                continue
            
            try:
                # CONTEXT EKLE: Önceki + mevcut + sonraki cümle
                context_parts = []
                if idx > 0:  # Önceki cümle varsa ekle
                    context_parts.append(combined_sentences[idx - 1])
                context_parts.append(sent)  # Mevcut cümle
                if idx < len(combined_sentences) - 1:  # Sonraki cümle varsa ekle
                    context_parts.append(combined_sentences[idx + 1])
                
                # Context'li metni birleştir
                context_text = ' '.join(context_parts)
                cleaned_context = clean_sentence(context_text)
                
                # Context ile analiz et
                sent_features = extract_advanced_features(cleaned_context)
                sent_tfidf = vectorizer.transform([cleaned_context])
                sent_stat = pd.DataFrame([sent_features])
                sent_stat_scaled = scaler.transform(sent_stat)
                sent_combined = hstack([sent_tfidf, sent_stat_scaled])
                
                # Model tahmini
                sent_pred = model.predict(sent_combined)[0]
                sent_proba = model.predict_proba(sent_combined)[0]
                
                # Dinamik eşik kullan
                sentence_results.append({
                    'text': sent,
                    'is_ai': float(sent_proba[1]) > sentence_threshold,
                    'ai_prob': float(sent_proba[1])
                })
            except Exception as e:
                # Hata olursa: genel olasılığı kullan
                sentence_results.append({
                    'text': sent,
                    'is_ai': bool(pred == 1),
                    'ai_prob': overall_ai_prob
                })
        
        return jsonify({
            'prediction': int(pred),
            'probabilities': {
                'human': float(proba[0]),
                'ai': float(proba[1])
            },
            'features': features,
            'sentences': sentence_results,
            'debug': {
                'total_sentences': len(combined_sentences),
                'analyzed_sentences': len(sentence_results),
                'errors': len([s for s in sentence_results if s.get('ai_prob') == 0.0]),
                'error_messages': errors_log
            }
        })
        
    except Exception as e:
        import traceback
        print("❌ HATA:", str(e))
        traceback.print_exc()
        return jsonify({'error': f'Analiz hatası: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

