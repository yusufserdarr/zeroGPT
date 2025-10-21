#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerçek ChatGPT Metinleri Toplama Aracı
--------------------------------------
ChatGPT'den aldığınız metinleri bu dosyaya ekleyin.
"""

import pandas as pd
import os

# ChatGPT'den aldığınız gerçek Türkçe metinleri buraya ekleyin
chatgpt_texts = [
    # Genel bilgi soruları
    """Yapay zeka, bilgisayarların insan benzeri düşünme ve öğrenme yeteneklerini 
    simüle etmesini sağlayan bir teknoloji dalıdır. Makine öğrenmesi ve derin öğrenme 
    gibi alt dalları sayesinde, bilgisayarlar verilerden öğrenerek kararlar alabilir 
    ve tahminlerde bulunabilir. Günümüzde sağlık, finans, eğitim ve birçok sektörde 
    kullanılmaktadır.""",
    
    """Türkiye'nin başkenti Ankara'dır. 1923 yılında Cumhuriyet'in ilanından sonra 
    İstanbul yerine başkent olarak seçilmiştir. Şehir, Anadolu'nun ortasında stratejik 
    bir konumda yer alır ve hem tarihî hem de modern yapılara ev sahipliği yapar.""",
    
    """Küresel ısınma, dünya genelinde ortalama sıcaklıkların artması olayıdır. 
    Başlıca nedeni sera gazı emisyonlarıdır. İklim değişikliği, buzulların erimesi, 
    deniz seviyesinin yükselmesi ve ekstrem hava olaylarının artması gibi ciddi 
    sonuçlara yol açmaktadır.""",
    
    # Öğretici/Açıklayıcı metinler
    """Python programlama dilinde değişken tanımlamak oldukça kolaydır. Örneğin, 
    'x = 5' yazarak bir tam sayı değişkeni oluşturabilirsiniz. Python, dinamik 
    tip belirleme kullanır, yani değişkenin tipini otomatik olarak algılar. 
    Bu sayede daha hızlı kod yazabilirsiniz.""",
    
    """Kahve yapmak için öncelikle taze çekilmiş kahve kullanmanız önerilir. 
    Su sıcaklığı 90-96 derece arasında olmalıdır. French press kullanıyorsanız, 
    kahveyi 4 dakika demledikten sonra prese basarak süzebilirsiniz. Böylece 
    aroması yüksek bir kahve elde edersiniz.""",
    
    # Tartışmalı/Fikir gerektiren konular
    """Uzaktan çalışmanın hem avantajları hem de dezavantajları vardır. Bir yandan 
    zaman ve maliyet tasarrufu sağlarken, diğer yandan sosyal izolasyon ve iletişim 
    sorunlarına yol açabilir. Ancak teknolojinin gelişmesiyle birlikte, video 
    konferans araçları bu sorunları minimize etmeye yardımcı olmaktadır.""",
    
    """Sosyal medyanın toplum üzerindeki etkisi tartışmalı bir konudur. Bilgiye 
    hızlı erişim ve küresel iletişim imkanı sunarken, dezenformasyon yayılımı ve 
    mahremiyet endişeleri de beraberinde getirmektedir. Kullanıcıların bilinçli 
    olması ve eleştirel düşünme becerileri geliştirmesi önemlidir.""",
    
    # Yaratıcı/Hikaye tarzı
    """Günlerden bir gün, küçük bir köyde yaşayan genç bir adam, ormanın derinliklerinde 
    eski bir harita buldu. Haritada işaretli hazine yeri, onu heyecanlandırdı. Ancak 
    yolculuk tehlikelerle doluydu. Cesaretini toplayarak yola koyuldu ve macera başladı.""",
    
    # Teknik açıklamalar
    """Blockchain teknolojisi, dağıtık bir veri tabanı sistemidir. Her işlem bloklara 
    kaydedilir ve bu bloklar zincir şeklinde birbirine bağlanır. Merkezi olmayan 
    yapısı sayesinde güvenlik ve şeffaflık sağlar. Kripto paralar, bu teknoloji 
    üzerine kuruludur.""",
    
    # Günlük konuşma tarzı
    """Dün akşam arkadaşlarla sinemaya gittik. Film gerçekten etkileyiciydi! 
    Özellikle son sahneler çok heyecanlıydı. Sonrasında bir kafede oturup filmi 
    tartıştık. Herkes farklı yorumlarda bulundu, çok eğlenceliydi.""",
]

# ÖNEMLİ: Yukarıdaki listeye kendi ChatGPT metinlerinizi ekleyin!
# Çeşitli konularda ChatGPT'ye sorular sorun ve cevapları ekleyin:
# - Bilimsel açıklamalar
# - Günlük konuşmalar
# - Teknik dökümantasyon
# - Hikayeler
# - Öneriler/Tavsiyeler

def save_chatgpt_data(texts, output_file='real_chatgpt_texts.csv'):
    """ChatGPT metinlerini CSV'ye kaydet"""
    df = pd.DataFrame({
        'content': texts,
        'label': 1  # 1 = AI
    })
    
    # Eğer dosya varsa, üzerine ekle
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ {len(df)} ChatGPT metni '{output_file}' dosyasına kaydedildi.")
    return df

def main():
    print("=" * 60)
    print("🤖 Gerçek ChatGPT Metinleri Toplama")
    print("=" * 60)
    
    if len(chatgpt_texts) < 10:
        print("\n⚠️  UYARI: Sadece", len(chatgpt_texts), "metin var!")
        print("   Daha iyi sonuçlar için en az 500-1000 metin toplamalısınız.")
        print("\n📝 Nasıl toplanır:")
        print("   1. ChatGPT'ye çeşitli sorular sorun")
        print("   2. Aldığınız cevapları chatgpt_texts listesine ekleyin")
        print("   3. Bu scripti tekrar çalıştırın")
        print("\n💡 Örnek sorular:")
        print("   - Yapay zeka nedir?")
        print("   - Python'da döngü nasıl kullanılır?")
        print("   - İklim değişikliği hakkında ne düşünüyorsun?")
        print("   - Bana bir hikaye anlat")
        print("   - Kahve nasıl yapılır?")
    
    df = save_chatgpt_data(chatgpt_texts)
    
    print("\n📊 İstatistikler:")
    print(f"   Toplam metin: {len(df)}")
    print(f"   Ortalama uzunluk: {df['content'].str.len().mean():.0f} karakter")
    
    print("\n" + "=" * 60)
    print("✅ İşlem tamamlandı!")
    print("=" * 60)
    print("\n🎯 Sonraki Adım:")
    print("   python3 retrain_with_real_data.py")

if __name__ == "__main__":
    main()

