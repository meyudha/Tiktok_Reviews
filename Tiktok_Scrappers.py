!pip install google-play-scraper

from google_play_scraper import reviews, Sort
import pandas as pd
from tabulate import tabulate
import ipywidgets as widgets
from IPython.display import display, clear_output
import math
import os
from datetime import datetime
import glob

app_id = "com.zhiliaoapp.musically"
jumlah_data = 50000
per_page = 15000
save_folder = "./saved_reviews"
os.makedirs(save_folder, exist_ok=True)
merge_existing = True
MASTER_FILE = "reviews_master_latest.csv"
RUN_SNAPSHOT_FILE = "reviews_run_snapshot.csv"
DISPLAY_SNAPSHOT_FILE = "reviews_display_snapshot.csv"
PAGE_FILE_PREFIX = "reviews_page"

def cleanup_merged_files():
    print("\n🧹 Cleaning up unwanted merged files...")
    merged_files = glob.glob("merged_*.csv")
    for f in merged_files:
        try:
            os.remove(f)
            print(f"   [✓] Deleted: {f}")
        except Exception as e:
            print(f"   [✗] Could not delete {f}: {e}")
    merged_files_in_folder = glob.glob(os.path.join(save_folder, "merged_*.csv"))
    for f in merged_files_in_folder:
        try:
            os.remove(f)
            print(f"   [✓] Deleted: {f}")
        except Exception as e:
            print(f"   [✗] Could not delete {f}: {e}")
    if not merged_files and not merged_files_in_folder:
        print("   No merged files found.")

def fetch_and_clean(app_id, jumlah_data):
    print("\n📥 Fetching reviews from Google Play...")
    print(f"   Requesting: {jumlah_data:,} reviews")
    result, _ = reviews(app_id, lang='en', country='us', sort=Sort.NEWEST, count=jumlah_data)
    df = pd.DataFrame(result)
    df_clean = df[["reviewId","userName","content","score","thumbsUpCount","reviewCreatedVersion","at","appVersion"]].copy()
    df_clean = df_clean.dropna(subset=['content'])
    df_clean = df_clean.drop_duplicates(subset=['reviewId'])
    df_clean['at'] = pd.to_datetime(df_clean['at'])
    df_clean = df_clean.sort_values('at', ascending=False).reset_index(drop=True)
    print(f"   ✓ Fetched: {len(df_clean):,} reviews (after cleaning)")
    return df_clean

def find_latest_master(folder):
    master_path = os.path.join(folder, MASTER_FILE)
    if os.path.exists(master_path):
        return master_path
    return None

def save_single_file(df, folder, filename):
    csv_path = os.path.join(folder, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    df.to_csv(csv_path, index=False)
    print(f"   [✓] Saved: {filename} ({len(df):,} rows)")
    return csv_path

def merge_with_master_keep_newest(df_new, folder):
    master_path = find_latest_master(folder)
    if master_path is None:
        print("\n📦 Creating new master file...")
        master_df = df_new.copy()
    else:
        print(f"\n📦 Merging with existing master...")
        master_df = pd.read_csv(master_path, parse_dates=['at'])
        print(f"   Old master: {len(master_df):,} rows")
        combined = pd.concat([master_df, df_new], axis=0, ignore_index=True)
        combined['at'] = pd.to_datetime(combined['at'])
        combined = combined.sort_values('at', ascending=False)
        combined = combined.drop_duplicates(subset=['reviewId'], keep='first').reset_index(drop=True)
        print(f"   New master: {len(combined):,} rows (after merge & dedup)")
        master_df = combined
    save_single_file(master_df, folder, MASTER_FILE)
    return master_df

print("="*70)
print("🚀 GOOGLE PLAY REVIEWS SCRAPER - CLEAN VERSION")
print("="*70)
cleanup_merged_files()
df_clean_run = fetch_and_clean(app_id, jumlah_data)
print("\n💾 Saving run snapshot...")
save_single_file(df_clean_run, save_folder, RUN_SNAPSHOT_FILE)

if merge_existing:
    master_df = merge_with_master_keep_newest(df_clean_run, save_folder)
else:
    print("\n⚠️  Merge disabled - master file not updated")
    master_df = None

print("\n📊 Preparing display data...")
df_display = df_clean_run.copy()
df_display = df_display.sort_values('at', ascending=False)
df_display = df_display.drop_duplicates(subset=['reviewId'], keep='first')
df_display = df_display.head(jumlah_data)
df_display = df_display.reset_index(drop=True)
print(f"   Display rows: {len(df_display):,} (expected: {jumlah_data:,})")

print("\n💾 Saving display snapshot...")
save_single_file(df_display, save_folder, DISPLAY_SNAPSHOT_FILE)

if len(df_display) != min(jumlah_data, len(df_clean_run)):
    print(f"\n⚠️  WARNING: Expected {jumlah_data:,} rows, got {len(df_display):,} rows")
else:
    print(f"\n✅ SUCCESS: Data saved exactly as requested ({len(df_display):,} rows)")

total_pages = math.ceil(len(df_display) / per_page) if len(df_display) > 0 else 1
page = 0

prev_btn = widgets.Button(description="⬅ Prev", layout=widgets.Layout(width="90px"))
next_btn = widgets.Button(description="Next ➡", layout=widgets.Layout(width="90px"))
save_page_btn = widgets.Button(description="💾 Save Page", button_style="success")
save_all_btn = widgets.Button(description="💾 Save Full", button_style="info")
show_files_btn = widgets.Button(description="📁 Show Files", button_style="")
page_label = widgets.Label()

def update_label():
    page_label.value = f"Page {page+1}/{total_pages} | Total rows: {len(df_display):,} (requested: {jumlah_data:,})"

def show_page(p):
    clear_output(wait=True)
    start = p * per_page
    end = min(start + per_page, len(df_display))
    display(page_label)
    print(f"\n{'='*70}")
    print(f"PAGE {p+1}/{total_pages} — Showing rows {start+1} to {end}")
    print('='*70 + "\n")
    display_df = df_display.iloc[start:end].copy()
    try:
        print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=True))
    except Exception:
        print(display_df.head(50).to_string())
    display(widgets.HBox([prev_btn, next_btn, save_page_btn, save_all_btn, show_files_btn]))

def on_prev(b):
    global page
    if page > 0:
        page -= 1
    update_label()
    show_page(page)

def on_next(b):
    global page
    if page < total_pages - 1:
        page += 1
    update_label()
    show_page(page)

def save_current_page(b):
    start = page * per_page
    end = min(start + per_page, len(df_display))
    sub = df_display.iloc[start:end].copy()
    fname = f"{PAGE_FILE_PREFIX}_{page+1}.csv"
    print(f"\n💾 Saving page {page+1}...")
    save_single_file(sub, save_folder, fname)

def save_full_again(b):
    print(f"\n💾 Re-saving full display dataset...")
    save_single_file(df_display, save_folder, DISPLAY_SNAPSHOT_FILE)

def show_saved_files(b):
    print("\n" + "="*70)
    print("📁 SAVED FILES IN FOLDER:")
    print("="*70)
    all_files = glob.glob(os.path.join(save_folder, "*.csv"))
    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    if not all_files:
        print("No files found.")
    else:
        for f in all_files:
            size = os.path.getsize(f)
            size_mb = size / (1024*1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
            rows_count = "N/A"
            try:
                temp_df = pd.read_csv(f)
                rows_count = f"{len(temp_df):,}"
            except:
                pass
            print(f"\n📄 {os.path.basename(f)}")
            print(f"   Size: {size_mb:.2f} MB | Rows: {rows_count} | Modified: {mtime}")
    print("\n" + "="*70)

prev_btn.on_click(on_prev)
next_btn.on_click(on_next)
save_page_btn.on_click(save_current_page)
save_all_btn.on_click(save_full_again)
show_files_btn.on_click(show_saved_files)
update_label()
show_page(page)

print("\n" + "="*70)
print("📊 SUMMARY INFORMATION")
print("="*70)
print(f"App ID              : {app_id}")
print(f"Requested Data      : {jumlah_data:,} reviews")
print(f"Fetched This Run    : {len(df_clean_run):,} reviews")
print(f"Display/Saved Data  : {len(df_display):,} reviews ✅")
if merge_existing and master_df is not None:
    print(f"Master Archive      : {len(master_df):,} reviews (all time)")
    print(f"Master File         : {MASTER_FILE}")
else:
    print("Master Archive      : Disabled")
print(f"\nFiles Saved:")
print(f"  • {DISPLAY_SNAPSHOT_FILE} → {len(df_display):,} rows (display data)")
print(f"  • {RUN_SNAPSHOT_FILE} → {len(df_clean_run):,} rows (raw run)")
if merge_existing:
    print(f"  • {MASTER_FILE} → {len(master_df):,} rows (archive)")
print(f"\nSave Folder         : {os.path.abspath(save_folder)}")
print(f"Per Page Display    : {per_page:,} rows")
print(f"Total Pages         : {total_pages}")
print("="*70)
print("\n💡 TIPS:")
print("  • Save Page    → Save current page only")
print("  • Save Full    → Re-save complete display dataset")
print("  • Show Files   → List all saved CSV files")
print("  • Mount Drive  → For permanent save: mount drive & change save_folder")
print("="*70)
cleanup_merged_files()
