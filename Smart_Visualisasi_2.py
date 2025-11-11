"""
Visualisasi Analisis Kata Kunci Ulasan TikTok
Data source: ./cleaned_reviews/tiktok_reviews_cleaned.csv

Output:
 - bar_emotional_vs_functional_with_keywords_table.html (tabel + bar chart interaktif)
 - summary_topkeywords_prop.csv (ringkasan proporsi & kata kunci)
 - *_top_keywords.csv (daftar kata kunci top per kategori)
"""

import os, re
from collections import Counter
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown

DATA_PATH = "./cleaned_reviews/tiktok_reviews_cleaned.csv"
OUT_HTML = "bar_emotional_vs_functional_with_keywords_table.html"
TOPN = 15
EXAMPLES_PER_KEYWORD = 3

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)[:20]}{'...' if len(df.columns)>20 else ''}")

score_candidates = [c for c in df.columns if c.lower() in ('score','rating','ratings','rate','rating_value','score_value','rating_score')]
content_candidates = [c for c in df.columns if c.lower() in ('content','review','text','comment','body')]

if not score_candidates:
    raise SystemExit("Kolom score/rating tidak ditemukan.")
if not content_candidates:
    raise SystemExit("Kolom content/review tidak ditemukan.")

score_col = score_candidates[0]
content_col = content_candidates[0]
print("Using score column:", score_col)
print("Using content column:", content_col)

df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
df[content_col] = df[content_col].astype(str)
df = df.dropna(subset=[score_col, content_col])

low_df = df[df[score_col] <= 2].copy()
high_df = df[df[score_col] == 5].copy()
print(f"Total reviews: {len(df):,}; Low (<=2): {len(low_df):,}; High (5): {len(high_df):,}")

emotional_keywords = [
    "angry","anger","angri","frustrat","frustration","annoy","annoyed","hate",
    "disappoint","disappointed","furious","rage","terrible","awful","worst","sucks","upset",
    "bug","crash","crashed","lag","laggy","slow","broken","error","fault","issue","problems","problem",
    "refund","scam","fake","spam"
]
functional_keywords = [
    "entertain","entertainment","filter","music","audio","video","edit","upload","camera",
    "effect","effects","feature","features","trend","duet","sticker","sound","filtering",
    "recommend","search","profile","notification","comment","share","download","stream",
    "layout","ui","ux","bookmark"
]

def build_word_pattern(keywords):
    escaped = sorted({re.escape(k) for k in keywords}, key=lambda x: -len(x))
    pat = r'(' + r'|'.join(escaped) + r')'
    return re.compile(pat, flags=re.IGNORECASE)

pat_em = build_word_pattern(emotional_keywords)
pat_fn = build_word_pattern(functional_keywords)

def extract_matches(text, pattern):
    if not isinstance(text, str) or text.strip() == "":
        return []
    found = pattern.findall(text)
    found_norm = [f.lower() for f in found if isinstance(f, str)]
    seen = set(); uniq = []
    for t in found_norm:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq

def annotate_group(df_group):
    dfg = df_group.copy()
    dfg['em_matches'] = dfg[content_col].map(lambda t: extract_matches(t, pat_em))
    dfg['fn_matches'] = dfg[content_col].map(lambda t: extract_matches(t, pat_fn))
    dfg['has_em'] = dfg['em_matches'].map(lambda lst: len(lst) > 0)
    dfg['has_fn'] = dfg['fn_matches'].map(lambda lst: len(lst) > 0)
    return dfg

low_annot = annotate_group(low_df)
high_annot = annotate_group(high_df)

def top_keywords(df_annot, col='em_matches', topn=TOPN):
    ctr = Counter()
    for lst in df_annot[col]:
        if isinstance(lst, list):
            ctr.update(lst)
    return ctr.most_common(topn)

low_em_top = top_keywords(low_annot, 'em_matches', TOPN)
low_fn_top = top_keywords(low_annot, 'fn_matches', TOPN)
high_em_top = top_keywords(high_annot, 'em_matches', TOPN)
high_fn_top = top_keywords(high_annot, 'fn_matches', TOPN)

def compute_stats(df_annot):
    n = len(df_annot)
    em_count = int(df_annot['has_em'].sum())
    fn_count = int(df_annot['has_fn'].sum())
    return {'n': n, 'em_count': em_count, 'fn_count': fn_count,
            'em_prop': em_count/n if n>0 else 0, 'fn_prop': fn_count/n if n>0 else 0}

low_stats = compute_stats(low_annot)
high_stats = compute_stats(high_annot)

plot_df = pd.DataFrame([
    {'group': 'Low (<=2)', 'category': 'Emotional', 'prop_pct': round(low_stats['em_prop'] * 100, 2)},
    {'group': 'Low (<=2)', 'category': 'Functional', 'prop_pct': round(low_stats['fn_prop'] * 100, 2)},
    {'group': 'High (5)', 'category': 'Emotional', 'prop_pct': round(high_stats['em_prop'] * 100, 2)},
    {'group': 'High (5)', 'category': 'Functional', 'prop_pct': round(high_stats['fn_prop'] * 100, 2)},
])

def topk_to_str(counter_list, k=5):
    if not counter_list:
        return ""
    return ", ".join([kw for kw, _ in counter_list[:k]])

rows = [
    {"Group": "Low (≤2)", "Kategori": "Emotional", "Kata Kunci Teratas": topk_to_str(low_em_top, 5),
     "Proporsi (%)": plot_df.query("group=='Low (<=2)' & category=='Emotional'")['prop_pct'].values[0]},
    {"Group": "Low (≤2)", "Kategori": "Functional", "Kata Kunci Teratas": topk_to_str(low_fn_top, 5),
     "Proporsi (%)": plot_df.query("group=='Low (<=2)' & category=='Functional'")['prop_pct'].values[0]},
    {"Group": "High (5)", "Kategori": "Emotional", "Kata Kunci Teratas": topk_to_str(high_em_top, 5),
     "Proporsi (%)": plot_df.query("group=='High (5)' & category=='Emotional'")['prop_pct'].values[0]},
    {"Group": "High (5)", "Kategori": "Functional", "Kata Kunci Teratas": topk_to_str(high_fn_top, 5),
     "Proporsi (%)": plot_df.query("group=='High (5)' & category=='Functional'")['prop_pct'].values[0]},
]

table_df = pd.DataFrame(rows)

fig = make_subplots(rows=1, cols=2,
                    column_widths=[0.55, 0.45],
                    specs=[[{"type": "table"}, {"type": "xy"}]],
                    horizontal_spacing=0.08,
                    subplot_titles=("Tabel Ringkasan Kata Kunci", "Proporsi (%) per Kategori"))

# Table trace
fig.add_trace(go.Table(
    header=dict(values=list(table_df.columns),
                fill_color='#2a3f5f', font=dict(color='white', size=12), align='left'),
    cells=dict(values=[table_df[c] for c in table_df.columns],
               fill_color='#f8f9fa', align='left', font=dict(color='black', size=11))
), row=1, col=1)

# Bar chart (grouped)
cats = ["Emotional", "Functional"]
groups = ["Low (≤2)", "High (5)"]
for g in groups:
    y_vals = []
    for c in cats:
        val = table_df.query("Group==@g & Kategori==@c")["Proporsi (%)"].values[0]
        y_vals.append(val)
    fig.add_trace(go.Bar(x=cats, y=y_vals, name=g, text=[f"{v:.2f}%" for v in y_vals], textposition="outside"), row=1, col=2)

fig.update_layout(
    title_text="Perbandingan Proporsi & Kata Kunci Teratas (Low vs High Reviews)",
    barmode="group",
    template="plotly_white",
    height=550,
    width=1100,
    legend_title="Group",
    margin=dict(l=40, r=20, t=80, b=40)
)
fig.update_yaxes(title_text="Proporsi (%)", row=1, col=2)

# Save and show
fig.write_html(OUT_HTML, include_plotlyjs='cdn')
print(f"Saved visualization to: {OUT_HTML}")
try:
    fig.show()
except Exception:
    print("Open the HTML file manually to view the chart.")

table_df.to_csv("summary_topkeywords_prop.csv", index=False)
print("Saved summary table -> summary_topkeywords_prop.csv")

def save_counter(counter_list, path):
    if not counter_list:
        pd.DataFrame(columns=['keyword','count']).to_csv(path, index=False)
        return
    pd.DataFrame(counter_list, columns=['keyword','count']).to_csv(path, index=False)

save_counter(low_em_top, "low_emotional_top_keywords.csv")
save_counter(low_fn_top, "low_functional_top_keywords.csv")
save_counter(high_em_top, "high_emotional_top_keywords.csv")
save_counter(high_fn_top, "high_functional_top_keywords.csv")
print("Saved detail CSVs: *_top_keywords.csv")

print("\n✅ All done — visualisasi kata kunci selesai.")
