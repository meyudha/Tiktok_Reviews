import os
import glob
import math
from datetime import datetime
import pandas as pd
from IPython.display import display, clear_output, Markdown
import ipywidgets as widgets
import warnings
warnings.filterwarnings("ignore")

save_folder = "./merged_reviews"
os.makedirs(save_folder, exist_ok=True)

gp_file = "./saved_reviews/reviews_display_snapshot.csv"
kaggle_file = "./kaggle_tiktok_reviews/kaggle_display_full.csv"

rows_per_page = 20000
display_limit = None


MASTER_FILE = "merged_reviews_master.csv"
DISPLAY_SNAPSHOT_FILE = "merged_reviews_display_snapshot.csv"
PAGE_FILE_PREFIX = "merged_reviews_page"


def save_single_file(df, folder, filename):
    """Simpan dataframe dengan hapus file lama terlebih dahulu"""
    if df is None or len(df) == 0:
        print("[Warn] DataFrame kosong - tidak ada yang disimpan.")
        return None

    csv_path = os.path.join(folder, filename)

    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"[Deleted] File lama dihapus: {os.path.basename(csv_path)}")

    df.to_csv(csv_path, index=False)
    print(f"[Saved] {len(df):,} rows → {os.path.basename(csv_path)}")
    return os.path.abspath(csv_path)

def harmonize_columns(df, source_name):
    """Standardisasi nama kolom"""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    column_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['reviewid', 'review_id', 'id']:
            column_map[col] = 'review_id'
        elif col_lower in ['username', 'user', 'author', 'name']:
            column_map[col] = 'user'
        elif col_lower in ['content', 'text', 'review', 'comment']:
            column_map[col] = 'content'
        elif col_lower in ['score', 'rating', 'stars']:
            column_map[col] = 'score'
        elif col_lower in ['thumbsupcount', 'thumbs_up', 'likes', 'like_count']:
            column_map[col] = 'thumbs_up'
        elif col_lower in ['appversion', 'app_version', 'version']:
            column_map[col] = 'app_version'
        elif col_lower in ['at', 'created_at', 'date', 'timestamp']:
            column_map[col] = 'created_at'

    df = df.rename(columns=column_map)


    required_cols = ['review_id', 'user', 'content', 'score', 'thumbs_up', 'app_version', 'created_at']
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

   
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['thumbs_up'] = pd.to_numeric(df['thumbs_up'], errors='coerce')

    df['source'] = source_name

    return df[['review_id', 'user', 'content', 'score', 'thumbs_up', 'app_version', 'created_at', 'source']]


print("\n" + "="*60)
print("LOADING SAVED DATA")
print("="*60)

df_gp = pd.DataFrame()
df_kaggle = pd.DataFrame()


if os.path.exists(gp_file):
    try:
        df_gp = pd.read_csv(gp_file)
        print(f"✓ Loaded Google Play: {gp_file} ({len(df_gp):,} rows)")
    except Exception as e:
        print(f"✗ Error loading Google Play: {e}")
else:
    print(f"⚠️ File Google Play tidak ditemukan: {gp_file}")


if os.path.exists(kaggle_file):
    try:
        df_kaggle = pd.read_csv(kaggle_file)
        print(f"✓ Loaded Kaggle: {kaggle_file} ({len(df_kaggle):,} rows)")
    except Exception as e:
        print(f"✗ Error loading Kaggle: {e}")
else:
    print(f"⚠️ File Kaggle tidak ditemukan: {kaggle_file}")

if len(df_gp) == 0 and len(df_kaggle) == 0:
    print("\n❌ TIDAK ADA DATA YANG DAPAT DIGABUNGKAN!")
    print("   Pastikan file berikut ada:")
    print(f"   • {gp_file}")
    print(f"   • {kaggle_file}")
    raise SystemExit("Script berhenti karena tidak ada data.")


print("\n" + "="*60)
print("HARMONIZING DATA")
print("="*60)

df_gp_harmonized = harmonize_columns(df_gp, 'google_play')
df_kaggle_harmonized = harmonize_columns(df_kaggle, 'kaggle')

print(f"✓ Google Play harmonized: {len(df_gp_harmonized):,} rows")
print(f"✓ Kaggle harmonized: {len(df_kaggle_harmonized):,} rows")


print("\n" + "="*60)
print("MERGING DATA")
print("="*60)

df_merged = pd.concat([df_gp_harmonized, df_kaggle_harmonized], ignore_index=True)


mask_bad_user = df_merged['user'].isna() | (df_merged['user'].astype(str).str.lower().str.strip().isin(['google user', 'nan', 'none', '']))
if mask_bad_user.any():
    df_merged.loc[mask_bad_user, 'user'] = df_merged.loc[mask_bad_user].apply(
        lambda r: f"anon_{r['review_id']}" if pd.notna(r['review_id']) else f"anon_idx_{int(r.name)}",
        axis=1
    )

df_merged = df_merged.sort_values('created_at', ascending=False, na_position='last').reset_index(drop=True)

total_rows = len(df_merged)
gp_count = len(df_gp_harmonized)
kaggle_count = len(df_kaggle_harmonized)

print(f"✅ Merge selesai. Total rows (DENGAN duplikat): {total_rows:,}")
print(f"   • Dari Google Play: {gp_count:,} rows")
print(f"   • Dari Kaggle: {kaggle_count:,} rows")
print(f"   • Total gabungan: {gp_count + kaggle_count:,} rows")

if df_merged['review_id'].notna().sum() > 0:
    unique_review_ids = df_merged['review_id'].notna().sum() - df_merged['review_id'].nunique()
    print(f"   • Potensi duplikat berdasarkan review_id: {unique_review_ids:,} rows")
else:
    unique_content = len(df_merged) - df_merged[['user', 'content']].drop_duplicates().shape[0]
    print(f"   • Potensi duplikat berdasarkan user+content: {unique_content:,} rows")

if total_rows > 0:
    save_single_file(df_merged, save_folder, MASTER_FILE)
else:
    print("[Warn] Tidak ada data untuk disimpan.")

if isinstance(display_limit, int) and display_limit > 0:
    df_display = df_merged.head(display_limit).reset_index(drop=True)
else:
    df_display = df_merged.copy()

total_display = len(df_display)
total_pages = max(1, math.ceil(total_display / rows_per_page))
page_idx = 0

if total_display > 0:
    save_single_file(df_display, save_folder, DISPLAY_SNAPSHOT_FILE)

prev_btn = widgets.Button(description="⬅ Prev", layout=widgets.Layout(width="90px"))
next_btn = widgets.Button(description="Next ➡", layout=widgets.Layout(width="90px"))
save_page_btn = widgets.Button(description="💾 Save Page", button_style="success")
save_all_btn = widgets.Button(description="💾 Save All", button_style="info")
show_files_btn = widgets.Button(description="📁 Files", button_style="")
info_label = widgets.Label()

def update_info():
    info_label.value = f"Page {page_idx+1}/{total_pages} — Display: {total_display:,} | Total: {total_rows:,}"

def show_page(p):
    clear_output(wait=True)
    start = p * rows_per_page
    end = min(start + rows_per_page, total_display)
    display(Markdown(f"### 📊 Merged Reviews - Page {p+1}/{total_pages}"))
    display(Markdown(f"**Rows {start+1:,}–{end:,} of {total_display:,}**"))
    if total_display > 0:
        display(df_display.iloc[start:end])
    else:
        print("No data to display.")
    display(widgets.HBox([prev_btn, next_btn, save_page_btn, save_all_btn, show_files_btn]))
    display(info_label)

def on_prev(b):
    global page_idx
    if page_idx > 0:
        page_idx -= 1
    update_info()
    show_page(page_idx)

def on_next(b):
    global page_idx
    if page_idx < total_pages - 1:
        page_idx += 1
    update_info()
    show_page(page_idx)

def on_save_page(b):
    start = page_idx * rows_per_page
    end = min(start + rows_per_page, total_display)
    sub = df_display.iloc[start:end].copy()
    fname = f"{PAGE_FILE_PREFIX}_{page_idx+1}.csv"
    save_single_file(sub, save_folder, fname)

def on_save_all(b):
    save_single_file(df_display, save_folder, DISPLAY_SNAPSHOT_FILE)

def show_saved_files(b):
    saved_files = glob.glob(os.path.join(save_folder, "*.csv"))
    print("\n" + "="*60)
    print("📁 SAVED FILES:")
    print("="*60)
    for f in sorted(saved_files):
        size_mb = os.path.getsize(f) / (1024*1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
        try:
            temp_df = pd.read_csv(f)
            rows = f"{len(temp_df):,}"
        except:
            rows = "N/A"
        print(f"✓ {os.path.basename(f)}")
        print(f"  Size: {size_mb:.2f} MB | Rows: {rows} | Modified: {mtime}")
    print("="*60 + "\n")

prev_btn.on_click(on_prev)
next_btn.on_click(on_next)
save_page_btn.on_click(on_save_page)
save_all_btn.on_click(on_save_all)
show_files_btn.on_click(show_saved_files)

update_info()
show_page(page_idx)

print("\n" + "="*60)
print("📋 SUMMARY")
print("="*60)
print(f"📂 INPUT FILES:")
print(f"   • Google Play: {gp_file}")
print(f"   • Kaggle: {kaggle_file}")
print(f"\n📊 MERGED DATA:")
print(f"   • Total rows (after dedup): {total_rows:,}")
print(f"   • Display rows: {total_display:,}")
print(f"\n💾 OUTPUT FILES:")
print(f"   • Master: {MASTER_FILE}")
print(f"   • Display snapshot: {DISPLAY_SNAPSHOT_FILE}")
print(f"   • Page files: {PAGE_FILE_PREFIX}_{{page}}.csv")
print(f"\n📁 OUTPUT FOLDER:")
print(f"   {os.path.abspath(save_folder)}")
print("="*60)
