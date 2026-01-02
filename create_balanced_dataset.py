import os
import csv
import random
import pandas as pd
import re
from collections import defaultdict

print("="*80)
print("🎯 BASİT: final_dataset_shorter.csv'den 5K AI + 5K İNSAN")
print("="*80)

def clean_text(text):
    """Metni temizle"""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def get_word_count(text):
    """Kelime sayısını hesapla"""
    return len(text.split())

# 1. final_dataset_shorter.csv'DEN VERİ TOPLA
print("\n📂 final_dataset_shorter.csv'den veri toplanıyor...")
human_data = []
ai_data = []

try:
    df = pd.read_csv('final_dataset_shorter.csv', encoding='utf-8-sig', on_bad_lines='skip', low_memory=False)
    
    content_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'content' in col_lower or 'text' in col_lower or 'sentence' in col_lower:
            content_col = col
        if 'label' in col_lower:
            label_col = col
    
    if content_col is None and len(df.columns) > 0:
        content_col = df.columns[0]
    
    if content_col and label_col:
        for _, row in df.iterrows():
            content = clean_text(row[content_col])
            if not content or len(content) < 20:
                continue
            
            word_count = get_word_count(content)
            if word_count < 5 or word_count > 200:
                continue
            
            try:
                label = int(row[label_col])
                if label == 1:
                    ai_data.append((content, word_count))
                elif label == 0:
                    human_data.append((content, word_count))
            except:
                pass
except Exception as e:
    print(f"⚠️ Hata: {e}")

print(f"   Toplanan: İnsan={len(human_data)}, AI={len(ai_data)}")

# 2. UZUNLUĞA GÖRE GRUPLA VE EŞLEŞTİR
print("\n🔢 Uzunluğa göre gruplama ve eşleştirme yapılıyor...")
random.seed(42)

human_by_length = defaultdict(list)
ai_by_length = defaultdict(list)

for content, word_count in human_data:
    length_group = (word_count // 5) * 5
    human_by_length[length_group].append(content)

for content, word_count in ai_data:
    length_group = (word_count // 5) * 5
    ai_by_length[length_group].append(content)

print(f"   İnsan uzunluk grupları: {len(human_by_length)}")
print(f"   AI uzunluk grupları: {len(ai_by_length)}")

# 3. 5K AI SEÇ VE BENZER UZUNLUKTA İNSAN EŞLEŞTİR
print("\n🎯 5K AI + 5K İnsan eşleştirmesi yapılıyor...")

selected_ai = []
selected_human = []

# Tüm AI verilerini karıştır
all_ai = [(content, word_count) for content, word_count in ai_data]
random.shuffle(all_ai)

# 5K AI seç
for content, word_count in all_ai[:5000]:
    length_group = (word_count // 5) * 5
    
    # Aynı uzunluk grubunda insan var mı?
    if length_group in human_by_length and len(human_by_length[length_group]) > 0:
        human_match = random.choice(human_by_length[length_group])
        human_by_length[length_group].remove(human_match)
        selected_ai.append(content)
        selected_human.append(human_match)
    else:
        # En yakın grubu bul
        closest_group = None
        min_diff = float('inf')
        for group in human_by_length.keys():
            diff = abs(group - length_group)
            if diff < min_diff and len(human_by_length[group]) > 0:
                min_diff = diff
                closest_group = group
        
        if closest_group is not None:
            human_match = random.choice(human_by_length[closest_group])
            human_by_length[closest_group].remove(human_match)
            selected_ai.append(content)
            selected_human.append(human_match)
    
    if len(selected_ai) >= 5000:
        break

print(f"   Seçilen: AI={len(selected_ai)}, İnsan={len(selected_human)}")

# 4. KARIŞTIR VE KAYDET
print("\n💾 Veri seti kaydediliyor...")
final_data = []
for content in selected_ai:
    final_data.append((content, 1))
for content in selected_human:
    final_data.append((content, 0))

random.shuffle(final_data)

with open('dengeli_veriset_5k.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['content', 'label'])
    for content, label in final_data:
        writer.writerow([content, label])

print(f"\n✅ BAŞARILI! Dengeli veri seti hazır!")
print(f"   Toplam: {len(final_data)} satır")
print(f"   İnsan (0): {sum(1 for _, label in final_data if label == 0)}")
print(f"   AI (1): {sum(1 for _, label in final_data if label == 1)}")
print("="*80)
