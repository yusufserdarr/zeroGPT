#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeroGPT_v2 — Türkçe AI vs İnsan Tespiti (Gelişmiş)
--------------------------------------------------
• TF-IDF + Zengin özellikler (burstiness, entropi, stopword oranı, tekrar oranı, vb.)
• Opsiyonel: spaCy (POS çeşitliliği), transformers (XLM-R ile pseudo-perplexity)
• Eğitim + değerlendirme + model/vektör kaydetme + tek cümle tahmini

Kullanım:
  # Eğitim + kaydetme (varsayılan dosya adlarıyla)
  python3 zeroGPT_v2.py --human temizlenmis_turkce_veri.csv --ai ai_texts_35k.csv --save

  # Tek cümle tahmini
  python3 zeroGPT_v2.py --predict "Bugün arkadaşımla kahve içtim ve sahilde yürüdüm."

Gerekli olabilecek paketler:
  pip install pandas scikit-learn joblib numpy
  # opsiyonel:
  pip install spacy
  python -m spacy download tr_core_news_sm
  pip install transformers torch sentencepiece
"""

import argparse
import os
import re
import sys
import math
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# ------------------------------------------------------------
# Konfig
# ------------------------------------------------------------
DEFAULT_HUMAN = "temizlenmis_turkce_veri.csv"
DEFAULT_AI    = "ai_texts_35k.csv"
MODEL_PATH    = "zeroGPT_v2_model.pkl"
VECT_PATH     = "zeroGPT_v2_vectorizer.pkl"   # TF-IDF bileşeni ayrıca kaydedilir
PIPE_PATH     = "zeroGPT_v2_pipeline.pkl"     # Tüm pipeline (önerilen)
FINAL_DATASET = "final_dataset_v2.csv"
RANDOM_STATE  = 42

# Basit Türkçe stopword listesi (kısa bir çekirdek set; istersen genişletebilirsin)
TR_STOPWORDS = {
    "ve","veya","ile","ama","fakat","ancak","çünkü","gibi","daha","çok",
    "az","bu","şu","o","bir","iki","üç","her","hiç","mi","mı","mu","mü",
    "de","da","ki","için","üzere","olan","olarak","sonra","önce","ise",
    "ya","ya da","ne","neden","nasıl","hangi","hem","en","çok","az","ben",
    "sen","o","biz","siz","onlar","var","yok","diye"
}

# ------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[ \t\r\n]+", " ", s)
    return s.strip()

def sentence_split(text: str) -> List[str]:
    # Nokta/soru/ünlem ile kabaca cümle bölme
    return [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]

def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    from collections import Counter
    c = Counter(text)
    total = sum(c.values())
    probs = [v/total for v in c.values()]
    return float(-sum(p*math.log(p+1e-12, 2) for p in probs))

def word_stats(text: str) -> Dict[str, float]:
    tokens = [w for w in re.findall(r"\w+", text, flags=re.UNICODE)]
    n = len(tokens)
    if n == 0:
        return dict(
            avg_word_len=0, type_token_ratio=0, top1_ratio=0,
            digit_ratio=0, uppercase_ratio=0, stopword_ratio=0
        )
    # ortalama kelime uzunluğu
    avg_word_len = sum(len(t) for t in tokens)/n
    # type-token ratio (çeşitlilik)
    ttr = len(set(t.lower() for t in tokens))/n
    # en sık kelimenin oranı (tekrar hissi)
    from collections import Counter
    cnt = Counter([t.lower() for t in tokens])
    top1_ratio = max(cnt.values())/n
    # rakam ve büyük harf oranı
    total_chars = max(1, len(text))
    digit_ratio = sum(ch.isdigit() for ch in text)/total_chars
    uppercase_ratio = sum(ch.isupper() for ch in text)/total_chars
    # stopword oranı
    sw = sum(1 for t in tokens if t.lower() in TR_STOPWORDS)
    stop_ratio = sw/n
    return dict(
        avg_word_len=avg_word_len,
        type_token_ratio=ttr,
        top1_ratio=top1_ratio,
        digit_ratio=digit_ratio,
        uppercase_ratio=uppercase_ratio,
        stopword_ratio=stop_ratio
    )

def burstiness(text: str) -> Tuple[float, float]:
    sents = sentence_split(text)
    if not sents:
        return 0.0, 0.0
    lengths = [len(re.findall(r"\w+", s)) for s in sents]
    mu = float(np.mean(lengths))
    sigma = float(np.std(lengths))
    return mu, sigma  # (ortalama cümle uzunluğu, değişkenlik=burstiness)

# Opsiyonel: spaCy POS çeşitliliği (kurulu değilse atla)
def pos_diversity(text: str) -> Tuple[float, int]:
    try:
        import spacy
        nlp = spacy.load("tr_core_news_sm")  # yoksa except'e düşer
    except Exception:
        return 0.0, 0  # yoksa sıfırla geç
    doc = nlp(text)
    tags = [t.pos_ for t in doc if not t.is_space]
    if not tags:
        return 0.0, 0
    # POS entropisi
    from collections import Counter
    c = Counter(tags)
    total = sum(c.values())
    probs = [v/total for v in c.values()]
    pos_entropy = float(-sum(p*math.log(p+1e-12, 2) for p in probs))
    # POS bigram çeşitliliği
    bigrams = list(zip(tags, tags[1:]))
    uniq_bigrams = len(set(bigrams))
    return pos_entropy, uniq_bigrams

# Opsiyonel: Pseudo-Perplexity (XLM-R masked LM ile)
# Kurulu değilse 0 döner; var ise maske-olabilirlik skoru üretir (küçük metinlerde pahalı olabilir).
def pseudo_perplexity(text: str, max_len: int = 256) -> float:
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch
    except Exception:
        return 0.0
    try:
        model_name = "xlm-roberta-base"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForMaskedLM.from_pretrained(model_name)
        mdl.eval()
        with torch.no_grad():
            tokens = tok.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)
            if len(tokens) < 5:
                return 0.0
            losses = []
            for i in range(1, len(tokens)-1):
                masked = tokens.copy()
                masked[i] = tok.mask_token_id
                inp = torch.tensor([masked])
                labels = torch.tensor([tokens])
                out = mdl(inp, labels=labels)
                # out.loss ~ korsan/ortalama; token bazlı negatif log olasılığına yaklaşım
                losses.append(out.loss.item())
            ppl = float(math.exp(np.mean(losses)))
            # Sıkılandırma: çok uç değerlere karşı sınırlayalım
            return min(ppl, 1e4)
    except Exception:
        return 0.0

def engineered_features_series(text: str) -> Dict[str, float]:
    t = clean_text(text)
    ce = char_entropy(t)
    ws = word_stats(t)
    mu, sigma = burstiness(t)
    pos_ent, pos_bi = pos_diversity(t)        # 0'a düşebilir (opsiyonel)
    ppl = pseudo_perplexity(t)                # 0'a düşebilir (opsiyonel)
    feats = {
        "char_entropy": ce,
        "avg_sent_len": mu,
        "burstiness": sigma,
        "pos_entropy": pos_ent,
        "pos_bigram_uniq": pos_bi,
        "pseudo_perplexity": ppl,
        **ws
    }
    return feats

def build_dense_features(X: pd.Series) -> pd.DataFrame:
    rows = [engineered_features_series(x) for x in X.tolist()]
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Veri yükleme ve hazırlama
# ------------------------------------------------------------
def load_dataset(path: str, assume_label: Optional[int] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[HATA] Dosya yok: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    if "content" not in df.columns:
        print(f"[HATA] 'content' sütunu bulunamadı. Sütunlar: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    if "label" not in df.columns:
        df["label"] = 0 if assume_label is None else assume_label
    df["content"] = df["content"].astype(str).map(clean_text)
    # çok kısa olanları at
    df = df[df["content"].str.split().str.len() > 3].reset_index(drop=True)
    return df[["content","label"]]

def build_final_dataset(human_path: str, ai_path: str) -> pd.DataFrame:
    human = load_dataset(human_path, assume_label=0)
    ai    = load_dataset(ai_path, assume_label=1)
    final = pd.concat([human.assign(label=0), ai.assign(label=1)], ignore_index=True)
    final = final.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    final.to_csv(FINAL_DATASET, index=False, encoding="utf-8-sig")
    return final

# ------------------------------------------------------------
# Model kurulum
# ------------------------------------------------------------
def build_pipeline() -> Pipeline:
    # TF-IDF metin vektörü
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95
    )

    # Dense özellikler (FunctionTransformer ile)
    dense_builder = FunctionTransformer(build_dense_features, validate=False)
    dense_pipe = Pipeline(steps=[
        ("dense_build", dense_builder),
        ("scale", StandardScaler(with_mean=False))  # DataFrame -> sparse ile uyum için with_mean=False
    ])

    # ColumnTransformer: hem TF-IDF hem dense
    # Not: FunctionTransformer doğrudan X (Series) beklediği için 2 kez aynı girdi veriyoruz.
    union = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf, "content"),
            ("dense", dense_pipe, "content"),
        ],
        remainder="drop"
    )

    # Sınıflandırıcı: GradientBoosting (iyi genelleme), LR fallback
    clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    pipe = Pipeline(steps=[
        ("union", union),
        ("clf", clf)
    ])
    return pipe

def train_eval_save(df: pd.DataFrame, save: bool) -> Pipeline:
    X = df[["content"]]  # ColumnTransformer 'content' kolonu ister
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print("\n=== DEĞERLENDİRME (v2) ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred, digits=4))

    if save:
        joblib.dump(pipe, PIPE_PATH)
        # Ayrı ayrı kaydetmek istersen (örnek): TF-IDF ve model
        # ama ColumnTransformer içinde oldukları için komple pipeline kaydı önerilir.
        print(f"[KAYIT] Pipeline -> {PIPE_PATH}")

    return pipe

# ------------------------------------------------------------
# Tahmin
# ------------------------------------------------------------
def predict_text(text: str):
    if not os.path.exists(PIPE_PATH):
        print("[HATA] Pipeline bulunamadı. Önce eğitin: python3 zeroGPT_v2.py --save", file=sys.stderr)
        sys.exit(1)
    pipe: Pipeline = joblib.load(PIPE_PATH)
    X = pd.DataFrame({"content": [text]})
    prob = pipe.predict_proba(X)[0]
    pred = int(np.argmax(prob))
    label = "İNSAN (0)" if pred == 0 else "YAPAY ZEKÂ (1)"
    conf  = prob[pred]
    print(f"Tahmin: {label} | Güven: {conf*100:.1f}%")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ZeroGPT_v2 — Perplexity/Burstiness/Pattern destekli Türkçe tespit")
    ap.add_argument("--human", type=str, default=DEFAULT_HUMAN, help="İnsan yazısı CSV (content[,label])")
    ap.add_argument("--ai",    type=str, default=DEFAULT_AI,    help="AI yazısı CSV (content[,label])")
    ap.add_argument("--save",  action="store_true", help="Model/pipeline kaydet")
    ap.add_argument("--predict", type=str, default=None, help="Tek metin için hızlı tahmin")
    args = ap.parse_args()

    if args.predict:
        predict_text(args.predict)
        return

    print("[BİLGİ] Final veri seti hazırlanıyor (v2)...")
    df = build_final_dataset(args.human, args.ai)
    print(f"[BİLGİ] Örnek sayısı: {len(df)} | Sınıf dağılımı:\n{df['label'].value_counts()}")

    print("[BİLGİ] Eğitim başlıyor (v2)...")
    _ = train_eval_save(df, save=args.save)

if __name__ == "__main__":
    main()
