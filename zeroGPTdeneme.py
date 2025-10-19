#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeroGPTdeneme.py
----------------
Türkçe metinlerde "İnsan (0) vs Yapay Zekâ (1)" sınıflandırması için
hızlı eğitim ve tahmin betiği.

Kullanım:
  Eğitim + değerlendirme + model kaydetme:
    python zeroGPTdeneme.py --human temizlenmis_turkce_veri.csv --ai ai_texts_35k.csv --save

  Sadece tahmin:
    python zeroGPTdeneme.py --predict "Yapay zekâ sistemleri veri temelli kararları hızlandırıyor."

Notlar:
- Varsayılan dosya adları aynı klasörde ise argümansız da çalışır.
- Çıktılar: zeroGPT_model.pkl, zeroGPT_vectorizer.pkl, final_dataset.csv
"""
import argparse
import os
import sys
import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

DEFAULT_HUMAN = "temizlenmis_turkce_veri.csv"
DEFAULT_AI = "ai_texts_35k.csv"
MODEL_PATH = "zeroGPT_model.pkl"
VECTORIZER_PATH = "zeroGPT_vectorizer.pkl"
FINAL_DATASET = "final_dataset.csv"
RANDOM_STATE = 42

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\\S+|www\\.\\S+", " ", s)
    s = re.sub(r"[\\t\\r\\n]+", " ", s)
    s = re.sub(r"\\s+", " ", s)
    return s.strip()

def load_dataset(path: str, assume_label: Optional[int] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[HATA] Dosya bulunamadı: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    # 'content' sütunu zorunlu
    if "content" not in df.columns:
        print(f"[HATA] '{path}' içinde 'content' sütunu yok. Sütunlar: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    # Label yoksa atayalım
    if "label" not in df.columns:
        if assume_label is None:
            print(f"[UYARI] '{path}' için 'label' sütunu yok, 0 atanıyor.", file=sys.stderr)
            df["label"] = 0
        else:
            df["label"] = assume_label
    # Temizlik
    df["content"] = df["content"].astype(str).map(clean_text)
    df = df[df["content"].str.split().str.len() > 3].reset_index(drop=True)
    return df[["content", "label"]]

def build_final_dataset(human_path: str, ai_path: str) -> pd.DataFrame:
    human = load_dataset(human_path, assume_label=0)
    ai = load_dataset(ai_path, assume_label=1)
    final = pd.concat([human.assign(label=0), ai.assign(label=1)], ignore_index=True)
    final = final.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    final.to_csv(FINAL_DATASET, index=False, encoding="utf-8-sig")
    return final

def train_and_eval(df: pd.DataFrame) -> Tuple[LogisticRegression, TfidfVectorizer]:
    X = df["content"].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced"  # dengesizlik varsa yardımcı
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("\\n=== DEĞERLENDİRME ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    return model, vectorizer

def save_artifacts(model, vectorizer):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[KAYIT] Model -> {MODEL_PATH}")
    print(f"[KAYIT] Vektörleştirici -> {VECTORIZER_PATH}")

def predict_text(text: str):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("[HATA] Model ve/veya vektörleştirici bulunamadı. Önce eğitin: zeroGPTdeneme.py --save", file=sys.stderr)
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    X = vectorizer.transform([clean_text(text)])
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0][pred]
    label = "İNSAN (0)" if pred == 0 else "YAPAY ZEKÂ (1)"
    if proba is not None:
        print(f"Tahmin: {label} | Güven: {proba*100:.1f}%")
    else:
        print(f"Tahmin: {label}")

def main():
    parser = argparse.ArgumentParser(description="ZeroGPT Türkçe Mini — Eğitim ve Tahmin")
    parser.add_argument("--human", type=str, default=DEFAULT_HUMAN, help="İnsan yazısı CSV (vars: temizlenmis_turkce_veri.csv)")
    parser.add_argument("--ai", type=str, default=DEFAULT_AI, help="AI yazısı CSV (vars: ai_texts_300.csv)")
    parser.add_argument("--save", action="store_true", help="Model ve vektörleştiriciyi kaydet")
    parser.add_argument("--predict", type=str, default=None, help="Tek metin için hızlı tahmin")
    args = parser.parse_args()

    if args.predict:
        predict_text(args.predict)
        return

    # Eğitim akışı
    print("[BİLGİ] Final veri seti hazırlanıyor...")
    df = build_final_dataset(args.human, args.ai)
    print(f"[BİLGİ] Örnek sayısı: {len(df)} | Sınıf dağılımı:\\n{df['label'].value_counts()}")

    print("[BİLGİ] Eğitim başlıyor...")
    model, vectorizer = train_and_eval(df)

    if args.save:
        save_artifacts(model, vectorizer)

if __name__ == "__main__":
    main()
