"""
One-line monthly mean rating (Plotly) — menggunakan:
./cleaned_reviews/tiktok_reviews_cleaned.csv

Output:
 - plot_monthly_mean_rating_single.html (interactive Plotly)
"""
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ---------- CONFIG ----------
DATA_PATH = "./cleaned_reviews/tiktok_reviews_cleaned.csv"
OUT_HTML = "plot_monthly_mean_rating_single.html"
PLOT_WIDTH = 1100
PLOT_HEIGHT = 550

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File tidak ditemukan: {DATA_PATH}")

print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)[:20]}{'...' if len(df.columns)>20 else ''}")

score_cols = [c for c in df.columns if c.lower() in ('score','rating','ratings','rate','rating_value','score_value')]
date_cols  = [c for c in df.columns if c.lower() in ('created_at','created','date','timestamp','time','review_time')]

if not score_cols:
    raise SystemExit("Kolom score/rating tidak ditemukan. Pastikan ada kolom 'score' atau 'rating'.")
if not date_cols:
    raise SystemExit("Kolom tanggal tidak ditemukan. Pastikan ada kolom 'created_at' atau serupa.")

score_col = score_cols[0]
date_col  = date_cols[0]
print(f"Using score column: {score_col}")
print(f"Using date column:  {date_col}")

df[score_col] = pd.to_numeric(df[score_col], errors='coerce')

df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=False)

df = df.dropna(subset=[date_col, score_col])
if df.empty:
    raise SystemExit("Setelah pembersihan, tidak ada data tersisa (date/score all NaN).")

df['year_month'] = df[date_col].dt.to_period('M').dt.to_timestamp('M')

monthly = (
    df
    .groupby('year_month', as_index=False)[score_col]
    .mean()
    .rename(columns={score_col: 'mean_score'})
    .sort_values('year_month')
)

print(f"Total bulan terdata: {len(monthly)} (dari {monthly['year_month'].min().date()} s.d {monthly['year_month'].max().date()})")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=monthly['year_month'],
    y=monthly['mean_score'],
    mode='lines+markers',
    line=dict(color='black', width=3),
    marker=dict(size=6, color='darkorange'),
    name='Rata-rata Rating Bulanan',
    hovertemplate='Bulan: %{x|%Y-%m}<br>Mean Rating: %{y:.3f}<extra></extra>'
))

fig.update_layout(
    title="Rata-rata Rating TikTok per Bulan",
    xaxis_title="Bulan",
    yaxis_title="Rata-rata Rating",
    yaxis=dict(range=[0, 5.1], tick0=0, dtick=0.5),
    template="plotly_white",
    hovermode="x unified",
    width=PLOT_WIDTH,
    height=PLOT_HEIGHT,
    margin=dict(l=60, r=20, t=80, b=60)
)

fig.write_html(OUT_HTML)
print("Saved:", OUT_HTML)

try:
    fig.show()
except Exception as e:
    print("Could not show figure inline (non-notebook environment). Open the HTML file to view. Error:", e)

print("\nSampel data bulanan (first 12 rows):")
print(monthly.head(12).to_string(index=False))
