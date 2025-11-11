"""
Stacked area (sentiment proportion per month) with major-version shown in tooltip.
Data forced from: ./cleaned_reviews/tiktok_reviews_cleaned.csv
Output: stacked_area_sentiment_with_versions_tooltip.html
"""
import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

DATA_PATH = "./cleaned_reviews/tiktok_reviews_cleaned.csv"
OUT_HTML = "stacked_area_sentiment_with_versions_tooltip.html"
PLOT_WIDTH = 1150
PLOT_HEIGHT = 600

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)[:30]}{'...' if len(df.columns)>30 else ''}")

score_candidates = [c for c in df.columns if c.lower() in (
    'score','rating','ratings','rate','rating_value','score_value','rating_score','stars')]

date_candidates = [c for c in df.columns if c.lower() in (
    'created_at','created','date','timestamp','time','review_time','created_time')]

version_candidates = [c for c in df.columns if c.lower() in (
    'app_version','version','app_ver','ver','version_name')]

if not score_candidates:
    raise SystemExit("Kolom score/rating tidak ditemukan. Pastikan dataset memiliki kolom score/rating.")
if not date_candidates:
    raise SystemExit("Kolom tanggal tidak ditemukan. Pastikan dataset memiliki kolom created_at atau serupa.")

score_col = score_candidates[0]
date_col = date_candidates[0]
version_col = version_candidates[0] if version_candidates else None

print("Using score column:", score_col)
print("Using date column: ", date_col)
print("Using version column:", version_col if version_col else "(none found)")

df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# drop rows without essential fields
df = df.dropna(subset=[score_col, date_col])
if df.empty:
    raise SystemExit("After dropna, no rows remain. Check data.")

# label sentiment
def label_sentiment(s):
    if s <= 2: return 'Negative'
    elif s == 3: return 'Neutral'
    else: return 'Positive'
df['sentiment'] = df[score_col].apply(label_sentiment)

def extract_major_version(val):
    if pd.isna(val):
        return np.nan
    s = str(val)
    # common version patterns: 1.2.3, v1.2, 12, 1-2 etc. pick first integer token
    m = re.search(r'\bv?(\d+)(?:\.\d+)?', s, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except:
            return np.nan
    # fallback: any integer in the string
    m2 = re.search(r'(\d+)', s)
    return int(m2.group(1)) if m2 else np.nan

if version_col and version_col in df.columns:
    df['major_version'] = df[version_col].map(extract_major_version).astype('Int64')
else:
    # try infer from other possible columns (like 'user_agent' or 'metadata') if present
    df['major_version'] = pd.NA

# month bucket (use month-end timestamp)
df['year_month'] = df[date_col].dt.to_period('M').dt.to_timestamp('M')

ver_first = (df.dropna(subset=['major_version'])
               .groupby('major_version')[date_col]
               .min()
               .reset_index()
               .rename(columns={date_col: 'first_seen'}))

if not ver_first.empty:
    ver_first['first_seen_month'] = ver_first['first_seen'].dt.to_period('M').dt.to_timestamp('M')
    # map month -> sorted list of versions
    month_to_versions = ver_first.groupby('first_seen_month')['major_version'].apply(lambda s: sorted(list(s))).to_dict()
else:
    month_to_versions = {}

sent_monthly = (df.groupby(['year_month','sentiment']).size().reset_index(name='count'))
# ensure months with zero for a sentiment are present
all_months = pd.Series(sorted(df['year_month'].unique()))
all_sentiments = ['Negative','Neutral','Positive']
idx = pd.MultiIndex.from_product([all_months, all_sentiments], names=['year_month','sentiment'])
sent_monthly = sent_monthly.set_index(['year_month','sentiment']).reindex(idx, fill_value=0).reset_index()
# compute proportions
total_per_month = sent_monthly.groupby('year_month')['count'].transform('sum')
# if a month has zero total (shouldn't normally happen), avoid div by zero
sent_monthly['prop'] = np.where(total_per_month>0, sent_monthly['count'] / total_per_month, 0.0)

pivot = sent_monthly.pivot(index='year_month', columns='sentiment', values='prop').fillna(0).reset_index().sort_values('year_month')

def versions_str_for_month(m):
    if m in month_to_versions:
        vs = month_to_versions[m]
        return ", ".join([f"v{int(v)}" for v in vs])
    else:
        return "—"

pivot['versions_list'] = pivot['year_month'].apply(versions_str_for_month)

# customdata columns: versions_list and total_count (optional)
# compute total count per month for tooltip context
month_counts = sent_monthly.groupby('year_month')['count'].sum().to_dict()
pivot['total_count'] = pivot['year_month'].map(lambda m: int(month_counts.get(m, 0)))
customdata = np.stack([pivot['versions_list'].astype(str).values, pivot['total_count'].values], axis=1)

fig = go.Figure()
colors = {'Negative':'#EF553B', 'Neutral':'#FECB52', 'Positive':'#00CC96'}
# ensure order bottom->top
for sentiment in ['Negative','Neutral','Positive']:
    if sentiment in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot['year_month'],
            y=pivot[sentiment],
            mode='lines',
            line=dict(width=0.5, color=colors[sentiment]),
            stackgroup='one',
            name=sentiment,
            customdata=customdata,
            hovertemplate='Bulan: %{x|%Y-%m}<br>Total reviews: %{customdata[1]}<br>Versi rilis: %{customdata[0]}<br>Proporsi %{y:.1%}<extra></extra>'
        ))

release_months = sorted(list(month_to_versions.keys()))
for i, m in enumerate(release_months):
    # reduce clutter: if many release months, show every 2nd (heuristic)
    if len(release_months) > 16 and i % 2 == 1:
        continue
    fig.add_vline(x=m, line=dict(color='rgba(80,80,80,0.18)', width=1, dash='dot'))

fig.update_layout(
    title="Transisi Proporsi Sentimen per Bulan (Negative → Neutral → Positive)\n(Hover menunjukkan major version yang pertama muncul di bulan tersebut)",
    xaxis_title="Bulan",
    yaxis_title="Proporsi Sentimen",
    yaxis=dict(tickformat=".0%", range=[0,1]),
    template="plotly_white",
    hovermode="x unified",
    width=PLOT_WIDTH,
    height=PLOT_HEIGHT,
    legend=dict(title='Sentiment', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=60, r=20, t=100, b=60)
)

fig.write_html(OUT_HTML)
print("Saved:", OUT_HTML)
try:
    fig.show()
except Exception as e:
    print("Could not show figure inline (non-notebook env). Open the HTML. Error:", e)

print("\nMonth range:", pivot['year_month'].min().date(), "to", pivot['year_month'].max().date())
print("Months with major-version first-seen (count):", len(month_to_versions))
if ver_first is not None and not ver_first.empty:
    display_cols = ['major_version','first_seen','first_seen_month']
    print("\nSample of major-version first-seen dates:")
    display(ver_first.sort_values('first_seen').head(20)[display_cols])
else:
    print("No major_version information available in dataset.")
