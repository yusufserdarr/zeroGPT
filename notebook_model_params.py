"""
Notebook için Model Parametreleri
Kullanıcının istediği: 200 features (LR, NB, SVM) ve 1000 features (RF)
80'lere düşürmek için agresif model parametreleri
"""

# TF-IDF PARAMETRELERİ
TFIDF_LR_NB_SVM = {
    'max_features': 200,
    'ngram_range': (1, 1),
    'min_df': 5,
    'max_df': 0.9
}

TFIDF_RF = {
    'max_features': 1000,
    'ngram_range': (1, 1),
    'min_df': 3,
    'max_df': 0.95
}

# MODEL PARAMETRELERİ (Agresif - 80'lere düşürmek için)
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 100,
    'C': 0.01,
    'random_state': 42
}

NAIVE_BAYES_PARAMS = {
    'alpha': 5.0
}

SVM_PARAMS = {
    'kernel': 'linear',
    'C': 0.01,
    'random_state': 42
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 20,
    'max_depth': 3,
    'random_state': 42,
    'n_jobs': -1
}

print("="*80)
print("📋 NOTEBOOK İÇİN MODEL PARAMETRELERİ")
print("="*80)
print("\nTF-IDF (LR, NB, SVM):")
for k, v in TFIDF_LR_NB_SVM.items():
    print(f"  {k}: {v}")

print("\nTF-IDF (Random Forest):")
for k, v in TFIDF_RF.items():
    print(f"  {k}: {v}")

print("\nLogistic Regression:")
for k, v in LOGISTIC_REGRESSION_PARAMS.items():
    print(f"  {k}: {v}")

print("\nNaive Bayes:")
for k, v in NAIVE_BAYES_PARAMS.items():
    print(f"  {k}: {v}")

print("\nSVM:")
for k, v in SVM_PARAMS.items():
    print(f"  {k}: {v}")

print("\nRandom Forest:")
for k, v in RANDOM_FOREST_PARAMS.items():
    print(f"  {k}: {v}")

print("\n" + "="*80)
print("✅ Bu parametrelerle notebook'taki modelleri güncelleyin!")
print("="*80)

