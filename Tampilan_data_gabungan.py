import os, re, html, unicodedata
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("TAHAP 1: MEMUAT DATA GABUNGAN")
print("="*70)

merged_file = "./merged_reviews/merged_reviews_master.csv"

if not os.path.exists(merged_file):
    print(f"❌ File tidak ditemukan: {merged_file}")
    print("   Jalankan script merge terlebih dahulu!")
    raise SystemExit("Script berhenti.")

df_raw = pd.read_csv(merged_file)
print(f"✓ Berhasil memuat data: {merged_file}")
print(f"  Total baris: {len(df_raw):,}")
print(f"  Total kolom: {len(df_raw.columns)}")
print(f"  Sumber data: {df_raw['source'].value_counts().to_dict()}")

stats_awal = {
    'total_rows': len(df_raw),
    'total_cols': len(df_raw.columns),
    'missing_total': df_raw.isna().sum().sum(),
    'duplicates': 0 
}

print(f"\n📊 STATISTIK AWAL DATA:")
print(f"  • Total baris: {stats_awal['total_rows']:,}")
print(f"  • Total nilai hilang: {stats_awal['missing_total']:,}")
print(f"  • Kolom: {', '.join(df_raw.columns.tolist())}")

print("\n" + "="*70)
print("TAHAP 2: ANALISIS & PENANGANAN MISSING VALUES")
print("="*70)

print("\n📋 METODOLOGI:")
print("   Missing values dapat mengganggu analisis statistik dan machine learning.")
print("   Strategi penanganan disesuaikan dengan karakteristik setiap kolom:")
print("   • Kolom kritis (content): hapus baris")
print("   • Kolom numerik (score, thumbs_up): analisis pola missing")
print("   • Kolom datetime: konversi dengan error handling")

missing_summary = pd.DataFrame({
    'Missing': df_raw.isna().sum(),
    'Percentage': (df_raw.isna().sum() / len(df_raw) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['Missing'] > 0].sort_values('Missing', ascending=False)

if len(missing_summary) > 0:
    print(f"\n📉 MISSING VALUES PER KOLOM:")
    for col, row in missing_summary.iterrows():
        print(f"  • {col}: {int(row['Missing']):,} ({row['Percentage']:.2f}%)")
else:
    print("\n✓ Tidak ada missing values ditemukan!")

df = df_raw.copy()

print("\n🔸 STEP 2A: Menangani Missing Content")
print("   ALASAN: Content adalah data utama untuk analisis sentimen.")
print("   Tanpa content, review tidak dapat dianalisis.")

before = len(df)
df = df.dropna(subset=['content'])
df = df[df['content'].astype(str).str.strip() != '']
after = len(df)
removed_content = before - after

print(f"   ❌ Baris dihapus: {removed_content:,} ({removed_content/before*100:.2f}%)")
print(f"   ✓ Baris tersisa: {after:,}")

print("\n🔸 STEP 2B: Analisis Missing pada Kolom Numerik")
print("   ALASAN: Missing values pada rating dapat mengindikasikan review tanpa rating.")

if 'score' in df.columns:
    missing_score = df['score'].isna().sum()
    print(f"\n   • Missing score: {missing_score:,} ({missing_score/len(df)*100:.2f}%)")
    if missing_score > 0:
        print("     → Score kosong dipertahankan sebagai 'tidak ada rating'")

if 'thumbs_up' in df.columns:
    missing_thumbs = df['thumbs_up'].isna().sum()
    print(f"   • Missing thumbs_up: {missing_thumbs:,} ({missing_thumbs/len(df)*100:.2f}%)")
    if missing_thumbs > 0:
        print("     → Diisi dengan 0 (asumsi: tidak ada vote)")
        df['thumbs_up'] = df['thumbs_up'].fillna(0)

print("\n" + "="*70)
print("TAHAP 3: PEMBERSIHAN TEKS (TEXT CLEANING)")
print("="*70)

print("\n📋 METODOLOGI:")
print("   Text cleaning bertujuan menstandarkan format teks untuk analisis:")
print("   • HTML decoding: menghilangkan entitas HTML (&amp;, &lt;, dll)")
print("   • URL removal: link tidak relevan untuk analisis sentimen")
print("   • Whitespace normalization: konsistensi spasi dan baris baru")
print("   • Unicode normalization: standarisasi karakter unicode")
print("   • Emoji removal: opsional, tergantung kebutuhan analisis")

def clean_text(text):
    """
    Fungsi pembersihan teks komprehensif
    """
    if pd.isna(text):
        return np.nan

    text = str(text)

    text = html.unescape(text)

    text = re.sub(r'http\S+|www\.\S+', '', text)

    text = ''.join(ch for ch in text if ch.isprintable() or ch in ['\n', '\t'])

    text = unicodedata.normalize('NFKC', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  
    u"\U0001F300-\U0001F5FF"  
    u"\U0001F680-\U0001F6FF"  
    u"\U0001F1E0-\U0001F1FF"  
    u"\U00002702-\U000027B0"  
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

print("\n🔸 STEP 3A: Membersihkan Teks Content")

sample_before = df['content'].head(3).tolist()

df['content_original'] = df['content'].copy()  
df['content'] = df['content'].astype(str).apply(clean_text)
df['content'] = df['content'].apply(lambda x: emoji_pattern.sub('', x))

sample_after = df['content'].head(3).tolist()

print("   ✓ Text cleaning selesai")
print("\n   📝 CONTOH PERUBAHAN:")
for i, (before, after) in enumerate(zip(sample_before, sample_after), 1):
    if before != after and len(before) < 100:
        print(f"   [{i}] Sebelum: {before[:80]}...")
        print(f"       Setelah:  {after[:80]}...")

before_empty = len(df)
df = df[df['content'].str.strip() != '']
after_empty = len(df)
removed_empty = before_empty - after_empty

if removed_empty > 0:
    print(f"\n   ❌ Baris dengan content kosong setelah cleaning: {removed_empty:,}")

print("\n" + "="*70)
print("TAHAP 4: KONVERSI & STANDARISASI TIPE DATA")
print("="*70)

print("\n📋 METODOLOGI:")
print("   Tipe data yang tepat penting untuk:")
print("   • Efisiensi memori dan komputasi")
print("   • Operasi matematika dan statistik yang valid")
print("   • Visualisasi dan analisis time-series")

print("\n🔸 STEP 4A: Konversi Kolom Numerik")
# Score
if 'score' in df.columns:
    before_dtype = df['score'].dtype
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    print(f"   • score: {before_dtype} → {df['score'].dtype}")
    print(f"     Range: {df['score'].min():.1f} - {df['score'].max():.1f}")
    print(f"     Mean: {df['score'].mean():.2f}")

# Thumbs up
if 'thumbs_up' in df.columns:
    before_dtype = df['thumbs_up'].dtype
    df['thumbs_up'] = pd.to_numeric(df['thumbs_up'], errors='coerce').fillna(0).astype('Int64')
    print(f"   • thumbs_up: {before_dtype} → {df['thumbs_up'].dtype}")
    print(f"     Range: {df['thumbs_up'].min()} - {df['thumbs_up'].max()}")
    print(f"     Mean: {df['thumbs_up'].mean():.2f}")

print("\n🔸 STEP 4B: Konversi Kolom Datetime")
if 'created_at' in df.columns:
    before_dtype = df['created_at'].dtype
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    print(f"   • created_at: {before_dtype} → {df['created_at'].dtype}")

    valid_dates = df['created_at'].notna().sum()
    print(f"     Valid dates: {valid_dates:,} ({valid_dates/len(df)*100:.2f}%)")

    if valid_dates > 0:
        print(f"     Range: {df['created_at'].min()} → {df['created_at'].max()}")
        span_days = (df['created_at'].max() - df['created_at'].min()).days
        print(f"     Span: {span_days:,} hari")

print("\n" + "="*70)
print("TAHAP 5: DETEKSI & PENANGANAN DUPLIKASI")
print("="*70)

print("\n📋 METODOLOGI:")
print("   Duplikasi dapat terjadi karena:")
print("   • Data scraping berulang dari sumber berbeda")
print("   • Review yang di-submit ulang")
print("   • Kesalahan dalam penggabungan data")
print("\n   Strategi: Prioritaskan review dengan timestamp terbaru")

before_dedup = len(df)

has_review_id = 'review_id' in df.columns and df['review_id'].notna().sum() > 0

if has_review_id:
    print("\n🔸 STEP 5A: Deduplikasi berdasarkan review_id")

    duplicates = df['review_id'].notna().sum() - df['review_id'].nunique()
    stats_awal['duplicates'] = duplicates

    print(f"   • Total review_id valid: {df['review_id'].notna().sum():,}")
    print(f"   • Unique review_id: {df['review_id'].nunique():,}")
    print(f"   • Duplikat terdeteksi: {duplicates:,}")

    if duplicates > 0:
        df = df.sort_values('created_at', ascending=False, na_position='last')
        df = df.drop_duplicates(subset=['review_id'], keep='first')
        print(f"   ✓ Duplikat dihapus, tersisa: {len(df):,} baris")
else:
    print("\n🔸 STEP 5A: Deduplikasi berdasarkan (user, content)")
    print("   CATATAN: review_id tidak tersedia, menggunakan kombinasi user+content")

    before_dup = len(df)
    duplicates = before_dup - df[['user', 'content']].drop_duplicates().shape[0]
    stats_awal['duplicates'] = duplicates

    print(f"   • Duplikat terdeteksi: {duplicates:,}")

    if duplicates > 0:
        df = df.sort_values('created_at', ascending=False, na_position='last')
        df = df.drop_duplicates(subset=['user', 'content'], keep='first')
        print(f"   ✓ Duplikat dihapus, tersisa: {len(df):,} baris")

after_dedup = len(df)
removed_dup = before_dedup - after_dedup

print(f"\n📊 HASIL DEDUPLIKASI:")
print(f"   • Baris sebelum: {before_dedup:,}")
print(f"   • Baris sesudah: {after_dedup:,}")
print(f"   • Dihapus: {removed_dup:,} ({removed_dup/before_dedup*100:.2f}%)")

print("\n" + "="*70)
print("TAHAP 6: FEATURE ENGINEERING & DETEKSI OUTLIER")
print("="*70)

print("\n📋 METODOLOGI:")
print("   Feature engineering menciptakan variabel baru untuk analisis:")
print("   • content_len: panjang teks untuk analisis detail/kedalaman review")
print("   • Outlier detection: identifikasi nilai ekstrem menggunakan percentile")
print("   • Capping: membatasi nilai ekstrem untuk menghindari bias statistik")

print("\n🔸 STEP 6A: Membuat Fitur Panjang Teks")
df['content_len'] = df['content'].astype(str).str.len()

print(f"   ✓ Feature 'content_len' dibuat")
print(f"   📏 Statistik panjang teks:")
print(f"      • Min: {df['content_len'].min():,} karakter")
print(f"      • Mean: {df['content_len'].mean():.1f} karakter")
print(f"      • Median: {df['content_len'].median():.1f} karakter")
print(f"      • Max: {df['content_len'].max():,} karakter")
print(f"      • Std: {df['content_len'].std():.1f}")

print("\n🔸 STEP 6B: Deteksi Outlier - Thumbs Up")
if 'thumbs_up' in df.columns:
    p25 = df['thumbs_up'].quantile(0.25)
    p50 = df['thumbs_up'].quantile(0.50)
    p75 = df['thumbs_up'].quantile(0.75)
    p95 = df['thumbs_up'].quantile(0.95)
    p99 = df['thumbs_up'].quantile(0.99)

    print(f"   📊 Distribusi thumbs_up:")
    print(f"      • Q1 (25%): {p25:.0f}")
    print(f"      • Median (50%): {p50:.0f}")
    print(f"      • Q3 (75%): {p75:.0f}")
    print(f"      • P95: {p95:.0f}")
    print(f"      • P99: {p99:.0f}")

    df['thumbs_up_capped'] = df['thumbs_up'].clip(upper=p99)
    outliers = (df['thumbs_up'] > p99).sum()

    print(f"\n   🎯 Outlier Detection:")
    print(f"      • Review dengan thumbs_up > P99: {outliers:,} ({outliers/len(df)*100:.2f}%)")
    print(f"      • Nilai di-cap ke: {p99:.0f}")
    print(f"      • Feature 'thumbs_up_capped' dibuat untuk analisis robust")

print("\n🔸 STEP 6C: Deteksi Outlier - Panjang Teks")
len_p999 = df['content_len'].quantile(0.999)
df['is_very_long'] = df['content_len'] > len_p999

very_long_count = df['is_very_long'].sum()
print(f"   📊 Review sangat panjang (> P99.9):")
print(f"      • Threshold: {len_p999:.0f} karakter")
print(f"      • Jumlah: {very_long_count:,} ({very_long_count/len(df)*100:.2f}%)")
print(f"      • Feature 'is_very_long' dibuat untuk flagging")

short_threshold = 10
very_short = (df['content_len'] < short_threshold).sum()
print(f"\n   📊 Review sangat pendek (< {short_threshold} karakter):")
print(f"      • Jumlah: {very_short:,} ({very_short/len(df)*100:.2f}%)")

print("\n" + "="*70)
print("TAHAP 7: PERSIAPAN DATA FINAL")
print("="*70)

keep_cols = [
    c for c in [
        'review_id', 'user', 'content', 'score',
        'thumbs_up', 'thumbs_up_capped', 'app_version',
        'created_at', 'source', 'content_len', 'is_very_long'
    ] if c in df.columns
]

df_cleaned = df[keep_cols].copy()
df_cleaned = df_cleaned.reset_index(drop=True)

print(f"✓ Kolom final dipilih: {len(keep_cols)} kolom")
print(f"  {', '.join(keep_cols)}")

print("\n" + "="*70)
print("TAHAP 8: MENYIMPAN DATA BERSIH")
print("="*70)

output_folder = "./cleaned_reviews"
os.makedirs(output_folder, exist_ok=True)

csv_path = os.path.join(output_folder, "tiktok_reviews_cleaned.csv")
parquet_path = os.path.join(output_folder, "tiktok_reviews_cleaned.parquet")

df_cleaned.to_csv(csv_path, index=False)
df_cleaned.to_parquet(parquet_path, index=False)

print(f"✓ Data berhasil disimpan:")
print(f"  • CSV: {csv_path}")
print(f"  • Parquet: {parquet_path}")
print(f"  • Total baris: {len(df_cleaned):,}")

print("\n" + "="*70)
print("TAHAP 9: RINGKASAN & REFLEKSI ANALITIS")
print("="*70)

stats_akhir = {
    'total_rows': len(df_cleaned),
    'total_cols': len(df_cleaned.columns),
    'missing_total': df_cleaned.isna().sum().sum(),
}

print("\n📊 PERBANDINGAN SEBELUM & SESUDAH CLEANING:\n")

print("┌─────────────────────────────────┬──────────────┬──────────────┬────────────┐")
print("│ Metrik                          │   Sebelum    │   Sesudah    │ Perubahan  │")
print("├─────────────────────────────────┼──────────────┼──────────────┼────────────┤")


pct_change_rows = ((stats_akhir['total_rows'] - stats_awal['total_rows']) / stats_awal['total_rows'] * 100)
print(f"│ Total Baris                     │ {stats_awal['total_rows']:>12,} │ {stats_akhir['total_rows']:>12,} │ {pct_change_rows:>9.2f}% │")


pct_change_miss = ((stats_akhir['missing_total'] - stats_awal['missing_total']) / max(stats_awal['missing_total'], 1) * 100)
print(f"│ Total Missing Values            │ {stats_awal['missing_total']:>12,} │ {stats_akhir['missing_total']:>12,} │ {pct_change_miss:>9.2f}% │")


print(f"│ Duplikat Dihapus                │ {stats_awal['duplicates']:>12,} │ {0:>12} │    -100.00% │")

print("└─────────────────────────────────┴──────────────┴──────────────┴────────────┘")

if 'score' in df_cleaned.columns:
    print(f"\n⭐ STATISTIK RATING:")
    print(f"   • Mean: {df_cleaned['score'].mean():.3f}")
    print(f"   • Median: {df_cleaned['score'].median():.1f}")
    print(f"   • Mode: {df_cleaned['score'].mode().values[0] if len(df_cleaned['score'].mode()) > 0 else 'N/A'}")
    print(f"   • Std Dev: {df_cleaned['score'].std():.3f}")

    rating_dist = df_cleaned['score'].value_counts().sort_index()
    print(f"\n   📊 Distribusi Rating:")
    for rating, count in rating_dist.items():
        pct = count/len(df_cleaned)*100
        bar = '█' * int(pct/2)
        print(f"      {rating:.0f}⭐: {count:>6,} ({pct:>5.2f}%) {bar}")

if 'created_at' in df_cleaned.columns and df_cleaned['created_at'].notna().sum() > 0:
    print(f"\n📅 DISTRIBUSI WAKTU:")
    df_cleaned['year'] = df_cleaned['created_at'].dt.year
    df_cleaned['month'] = df_cleaned['created_at'].dt.month

    yearly = df_cleaned['year'].value_counts().sort_index()
    print(f"   Per Tahun:")
    for year, count in yearly.items():
        if pd.notna(year):
            pct = count/len(df_cleaned)*100
            print(f"      {int(year)}: {count:>6,} ({pct:>5.2f}%)")

if 'source' in df_cleaned.columns:
    print(f"\n📁 DISTRIBUSI SUMBER DATA:")
    source_dist = df_cleaned['source'].value_counts()
    for source, count in source_dist.items():
        pct = count/len(df_cleaned)*100
        print(f"   • {source}: {count:>6,} ({pct:>5.2f}%)")

print("\n" + "="*70)
print("💡 REFLEKSI ANALITIS")
print("="*70)

print("""
1. KUALITAS DATA MENINGKAT:
   • Data mentah telah melalui 9 tahapan pembersihan sistematis
   • Missing values pada kolom kritis telah ditangani
   • Duplikasi telah dihilangkan untuk menghindari bias

2. STANDARDISASI FORMAT:
   • Semua teks telah dinormalisasi (HTML, URL, whitespace)
   • Tipe data telah dikonversi sesuai karakteristik kolom
   • Timestamp telah distandarkan untuk analisis time-series

3. OUTLIER HANDLING:
   • Outlier pada thumbs_up telah di-cap untuk analisis robust
   • Review ekstrem panjang telah di-flag untuk investigasi
   • Feature engineering menambah dimensi analisis

4. SIAP UNTUK ANALISIS:
   • Data bersih dan terstruktur untuk EDA
   • Format parquet tersedia untuk efisiensi big data
   • Feature tambahan mendukung analisis mendalam

5. TRADE-OFFS YANG DIPERTIMBANGKAN:
   • Penghapusan duplikat: kehilangan variasi temporal minimal
   • Emoji removal: fokus pada analisis tekstual, bukan visual
   • Outlier capping: preservasi distribusi tanpa bias ekstrem
""")

print("="*70)
print("✅ CLEANING SELESAI - DATA SIAP UNTUK ANALISIS")
print("="*70)
