# 📱 TikTok Reviews Data Analysis

Repository ini berisi **pipeline analisis data ulasan aplikasi TikTok**, yang digabungkan dari dua sumber utama:
- Dataset **Kaggle**
- Dataset **Google Play Store**

Tujuan utama proyek ini adalah **membersihkan, menggabungkan, dan menyiapkan data** agar siap untuk eksplorasi, analisis sentimen, dan visualisasi.

---

## 🧩 Struktur Direktori

/
├── kaggle_tiktok_reviews/ # Data mentah dari Kaggle
│ └── kaggle_display_full.csv
├── saved_reviews/ # Data hasil scraping Google Play
│ └── reviews_display_snapshot.csv
├── merged_reviews/ # Folder hasil penggabungan data
│ ├── merged_reviews_master.csv
│ ├── merged_reviews_display_snapshot.csv
│ └── merged_reviews_page_*.csv
├── cleaned_reviews/ # Hasil akhir data setelah cleaning
│ ├── tiktok_reviews_cleaned.csv
│ └── tiktok_reviews_cleaned.parquet
├── scripts/
│ ├── merged_saved_reviews.py # Script penggabungan data
│ └── display_cleaning_results.py # Script tampilan hasil cleaning
└── README.md # Dokumentasi proyek

---

## ⚙️ Instalasi & Persiapan

### 1️⃣ Clone Repository
```bash
git clone https://github.com/meyudha/Tiktok_Reviews.git
cd Tiktok_Reviews

pip install pandas ipywidgets
