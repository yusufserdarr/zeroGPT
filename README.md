# ZeroGPT - AI Text Detection for Turkish

Türkçe metinler için AI (Yapay Zeka) ve İnsan yazısı ayırt etme projesi.

## 📋 Proje Açıklaması

Bu proje, Türkçe metinlerin AI tarafından mı yoksa insan tarafından mı yazıldığını tespit etmek için geliştirilmiş bir makine öğrenmesi sistemidir.

## 🚀 Özellikler

- **Klasik ML Modelleri**: Logistic Regression, Naive Bayes, Random Forest, SVM, Gradient Boosting
- **Deep Learning Modelleri**: LSTM (Bidirectional), BERTurk (Transformer), AutoGun (AutoML)
- **Web Uygulaması**: Streamlit ile interaktif web arayüzü
- **Veri Seti Oluşturma**: Otomatik dengeli veri seti oluşturma araçları

## 📁 Proje Yapısı

```
zeroGPT/
├── ilkDeneme.ipynb              # Ana model eğitim notebook'u
├── create_ai_dataset.py         # AI veri seti oluşturma
├── create_human_dataset.py      # İnsan veri seti oluşturma
├── create_balanced_dataset.py    # Dengeli veri seti oluşturma
├── app_final.py                 # Streamlit web uygulaması
├── zeroGPTdeneme.py             # Basit model eğitimi
└── Kodlar/                      # Ek kodlar
```

## 🛠️ Kurulum

### Gereksinimler

```bash
pip install pandas numpy scikit-learn tensorflow transformers
pip install streamlit  # Web uygulaması için
pip install autogluon  # AutoGun için (opsiyonel)
```

### Veri Seti Hazırlama

1. **AI Veri Seti Oluşturma:**
```bash
python3 create_ai_dataset.py
```

2. **İnsan Veri Seti Oluşturma:**
```bash
python3 create_human_dataset.py
```

## 📊 Model Performansı

### Klasik ML Modelleri
- Logistic Regression: ~83%
- Naive Bayes: ~80%
- Random Forest: ~81%
- SVM: ~84%
- Gradient Boosting: ~83%

### Deep Learning Modelleri
- LSTM: ~83%
- BERTurk: ~84%
- AutoGun: ~85%+

## 🎯 Kullanım

### Notebook ile Eğitim
1. `ilkDeneme.ipynb` dosyasını aç
2. Hücreleri sırayla çalıştır
3. Model sonuçlarını kontrol et

### Web Uygulaması
```bash
streamlit run app_final.py
```

## 📝 Notlar

- Veri setleri `.gitignore` ile hariç tutulmuştur
- Model dosyaları `.gitignore` ile hariç tutulmuştur
- Kendi veri setlerinizi kullanarak eğitim yapabilirsiniz

## 📄 Lisans

Bu proje eğitim amaçlıdır.



