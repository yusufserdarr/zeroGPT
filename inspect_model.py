#!/usr/bin/env python3
"""Model içeriğini görüntüle"""
import joblib
import json

print("=" * 60)
print("🔍 MODEL ANALİZİ")
print("=" * 60)

# Model yükle
model = joblib.load("zeroGPT_final_model.pkl")
vectorizer = joblib.load("zeroGPT_final_vectorizer.pkl")
scaler = joblib.load("zeroGPT_final_scaler.pkl")

print("\n📦 MODEL TİPİ:")
print(f"   {type(model).__name__}")

print("\n⚙️ MODEL PARAMETRELERİ:")
params = model.get_params()
for key, value in params.items():
    print(f"   {key}: {value}")

print("\n📊 VECTORİZER BİLGİSİ:")
print(f"   Toplam feature: {len(vectorizer.vocabulary_)}")
print(f"   Max features: {vectorizer.max_features}")
print(f"   N-gram range: {vectorizer.ngram_range}")

print("\n🔢 SCALER BİLGİSİ:")
print(f"   Feature sayısı: {scaler.n_features_in_}")

print("\n✅ MODEL ÖZETİ:")
print(f"   TF-IDF Features: {len(vectorizer.vocabulary_)}")
print(f"   Stat Features: {scaler.n_features_in_}")
print(f"   Toplam: {len(vectorizer.vocabulary_) + scaler.n_features_in_}")

print("\n" + "=" * 60)

