!pip install kagglehub
!pip install kaggle

import json
import os
import glob
from datetime import datetime
import pandas as pd
from IPython.display import display, clear_output
import ipywidgets as widgets
import math
import shutil
import sys
import stat

kaggle_dataset = "ashishkumarak/tiktok-reviews-daily-updated"
mount_drive = False
jumlah_data = 106915
per_page = 20000
save_folder = "./kaggle_tiktok_reviews"
os.makedirs(save_folder, exist_ok=True)
kaggle_token = None
SNAPSHOT_FILE = "kaggle_display_snapshot.csv"
SNAPSHOT_PARQUET = "kaggle_display_snapshot.parquet"
FULL_DISPLAY_FILE = "kaggle_display_full.csv"
PAGE_FILE_PREFIX = "kaggle_page"

def write_kaggle_token(token_dict):
    if token_dict is None:
        return None
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    p = os.path.join(kaggle_dir, "kaggle.json")
    with open(p, "w") as f:
        json.dump(token_dict, f)
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    return p

def download_and_unzip(kaggle_id, dest_folder):
    print(f"Downloading {kaggle_id} to {dest_folder} ... (this may take a while)")
    os.makedirs(dest_folder, exist_ok=True)
    cmd = f"kaggle datasets download -d {kaggle_id} -p {dest_folder} --unzip"
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError("kaggle CLI returned non-zero exit code. Pastikan kaggle terpasang dan token valid.")
    print("Download & unzip selesai.")

def find_csv_in_folder(folder, pattern="*.csv"):
    files = glob.glob(os.path.join(folder, pattern))
    files.sort(key=os.path.getmtime, reverse=False)
    return files

def save_single_file(df, folder, filename):
    csv_path = os.path.join(folder, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"[Deleted] Old file removed: {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"[Saved] New file saved: {csv_path}")
    pq_path = None
    if "snapshot" in filename:
        try:
            pq_filename = filename.replace(".csv", ".parquet")
            pq_path = os.path.join(folder, pq_filename)
            if os.path.exists(pq_path):
                os.remove(pq_path)
            df.to_parquet(pq_path, index=False)
            print(f"[Saved] Parquet version: {pq_path}")
        except Exception as e:
            print(f"[Warning] Could not save parquet: {e}")
            pq_path = None
    return csv_path, pq_path

if mount_drive:
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        save_folder = "/content/drive/MyDrive/kaggle_tiktok_reviews"
        os.makedirs(save_folder, exist_ok=True)
        print("Google Drive mounted. Save folder set to:", save_folder)
    except Exception as e:
        print("Gagal mount Drive (not Colab or missing package). Continuing with local storage.", e)

if kaggle_token:
    token_path = write_kaggle_token(kaggle_token)
    print("Kaggle token written to:", token_path)
else:
    print("No kaggle token provided programmatically. Assuming token already available in ~/.kaggle/kaggle.json")

download_and_unzip(kaggle_dataset, save_folder)
csv_files = find_csv_in_folder(save_folder, pattern="*.csv")
csv_files = [f for f in csv_files if not any(x in os.path.basename(f) for x in [SNAPSHOT_FILE, FULL_DISPLAY_FILE, PAGE_FILE_PREFIX])]
if not csv_files:
    raise FileNotFoundError(f"Tidak menemukan file CSV di folder {save_folder} setelah download.")
selected_csv = None
for f in csv_files:
    if os.path.basename(f).lower().startswith("tiktok_reviews"):
        selected_csv = f
        break
if selected_csv is None:
    selected_csv = csv_files[-1]

print("Memuat file CSV:", selected_csv)
df_full = pd.read_csv(selected_csv)
print(f"✅ Dataset dimuat: {len(df_full):,} baris, {len(df_full.columns)} kolom")

if jumlah_data is not None:
    df = df_full.head(jumlah_data).copy()
else:
    df = df_full.copy()

snapshot_csv, snapshot_parquet = save_single_file(df, save_folder, SNAPSHOT_FILE)
rows_per_page = per_page if per_page and per_page > 0 else 20000
total_pages = math.ceil(len(df) / rows_per_page) if len(df) > 0 else 1
page = 1

prev_button = widgets.Button(description="⬅️ Previous", layout=widgets.Layout(width="120px"))
next_button = widgets.Button(description="Next ➡️", layout=widgets.Layout(width="120px"))
save_page_btn = widgets.Button(description="💾 Save current page", button_style="success")
save_full_btn = widgets.Button(description="💾 Save full display", button_style="info")
download_btn = widgets.Button(description="📁 Show saved files", button_style="")
page_label = widgets.Label()

def update_label():
    page_label.value = f"Page {page}/{total_pages} — showing rows {(page-1)*rows_per_page+1:,}–{min(page*rows_per_page, len(df)):,} of {len(df):,}"

def show_page(p):
    clear_output(wait=True)
    start = (p - 1) * rows_per_page
    end = min(start + rows_per_page, len(df))
    display(page_label)
    subset = df.iloc[start:end].copy()
    display(subset)
    print(f"Menampilkan baris {start+1:,} sampai {end:,} dari total {len(df):,}")
    display(widgets.HBox([prev_button, next_button, save_page_btn, save_full_btn, download_btn]))

def on_prev(b):
    global page
    if page > 1:
        page -= 1
    update_label()
    show_page(page)

def on_next(b):
    global page
    if page < total_pages:
        page += 1
    update_label()
    show_page(page)

def on_save_page(b):
    start = (page - 1) * rows_per_page
    end = min(start + rows_per_page, len(df))
    sub = df.iloc[start:end].copy()
    fname = f"{PAGE_FILE_PREFIX}_{page}.csv"
    save_single_file(sub, save_folder, fname)

def on_save_full(b):
    save_single_file(df, save_folder, FULL_DISPLAY_FILE)

def on_open_files(b):
    saved_files = [
        os.path.join(save_folder, SNAPSHOT_FILE),
        os.path.join(save_folder, SNAPSHOT_PARQUET),
        os.path.join(save_folder, FULL_DISPLAY_FILE)
    ]
    page_files = glob.glob(os.path.join(save_folder, f"{PAGE_FILE_PREFIX}_*.csv"))
    saved_files.extend(page_files)
    print("\n" + "="*60)
    print("📁 SAVED FILES:")
    print("="*60)
    for f in saved_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            size_mb = size / (1024*1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"✓ {os.path.basename(f)} ({size_mb:.2f} MB) - Modified: {mtime}")
    print("="*60 + "\n")

prev_button.on_click(on_prev)
next_button.on_click(on_next)
save_page_btn.on_click(on_save_page)
save_full_btn.on_click(on_save_full)
download_btn.on_click(on_open_files)

update_label()
show_page(page)

print("\n" + "="*60)
print("RINGKASAN:")
print("="*60)
print(f"- Kaggle dataset: {kaggle_dataset}")
print(f"- CSV loaded: {selected_csv}")
print(f"- Rows loaded into df (for display): {len(df):,} (requested: {jumlah_data})")
print(f"- Save folder: {os.path.abspath(save_folder)}")
print(f"- Snapshot file: {SNAPSHOT_FILE} (otomatis dibuat)")
print("\nFITUR SAVE:")
print("- Save current page → File: kaggle_page_{page}.csv (hapus lama, simpan baru)")
print("- Save full display → File: kaggle_display_full.csv (hapus lama, simpan baru)")
print("- Setiap run ulang, file lama akan dihapus dan diganti dengan data baru")
print("="*60)
