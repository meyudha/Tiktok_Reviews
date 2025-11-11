"""
TikTok Reviews Regression Modeling
Lokasi dataset: ./cleaned_reviews/tiktok_reviews_cleaned.csv

Algoritma:
 - Linear Regression
 - XGBoost Regressor

Metrik evaluasi:
 - Mean Squared Error (MSE)
 - Root Mean Squared Error (RMSE)
 - Coefficient of Determination (R²)

Analisis interpretatif:
 - Menjelaskan perbedaan performa kedua model.
 - Menunjukkan fitur paling berpengaruh (XGBoost feature importance).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Konfigurasi
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
RANDOM_STATE = 42
data_path = "./cleaned_reviews/tiktok_reviews_cleaned.csv"
models_dir = "./trained_models"
os.makedirs(models_dir, exist_ok=True)

print("="*80)
print("📂 MEMUAT DATA DARI:", data_path)
print("="*80)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}")

df = pd.read_csv(data_path)
print(f"Data shape: {df.shape}")
print(f"Kolom: {list(df.columns)}")
display(df.head())

print("\n== DATA PREPROCESSING ==")

# Hilangkan missing values
df = df.dropna()

# Tentukan target (misal: rating atau score)
target_candidates = [c for c in df.columns if 'rating' in c.lower() or 'score' in c.lower() or 'target' in c.lower()]
if not target_candidates:
    raise SystemExit("❌ Tidak ditemukan kolom target (rating/score). Pastikan dataset memiliki target numerik.")
target = target_candidates[0]
print(f"🎯 Target variabel: {target}")

# Pisahkan fitur numerik
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target in num_cols:
    num_cols.remove(target)
X = df[num_cols]
y = df[target]

print(f"Fitur numerik digunakan: {len(num_cols)} kolom")
print(f"Target: {target}")

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Scaling untuk model linear
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n== INISIALISASI MODEL ==")

lr = LinearRegression()
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("✓ Linear Regression siap")
print("✓ XGBoost Regressor siap")

print("\n== TRAINING MODELS ==")

# Linear Regression
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# XGBoost Regressor
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name} Performance:")
    print(f"  • MSE  = {mse:.4f}")
    print(f"  • RMSE = {rmse:.4f}")
    print(f"  • R²   = {r2:.4f}")
    return {"MSE": mse, "RMSE": rmse, "R2": r2}

metrics_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression")
metrics_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost Regressor")

comparison_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "R²"],
    "Linear Regression": [metrics_lr["MSE"], metrics_lr["RMSE"], metrics_lr["R2"]],
    "XGBoost Regressor": [metrics_xgb["MSE"], metrics_xgb["RMSE"], metrics_xgb["R2"]],
})
comparison_df["Better Model"] = comparison_df.apply(
    lambda row: "XGBoost" if (
        (row["Metric"] in ["MSE", "RMSE"] and row["XGBoost Regressor"] < row["Linear Regression"]) or
        (row["Metric"] == "R²" and row["XGBoost Regressor"] > row["Linear Regression"])
    ) else "Linear Regression", axis=1
)
print("\n📈 Perbandingan Metrik:")
display(comparison_df)

# Plot prediksi
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_lr, color="steelblue", alpha=0.6)
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title("Linear Regression - Actual vs Predicted")

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_xgb, color="forestgreen", alpha=0.6)
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title("XGBoost Regressor - Actual vs Predicted")
plt.tight_layout()
plt.show()

print("\n== INTERPRETIVE ANALYSIS ==")

diff_r2 = metrics_xgb["R2"] - metrics_lr["R2"]
diff_rmse = metrics_lr["RMSE"] - metrics_xgb["RMSE"]

if diff_r2 > 0.02:
    print("""
🔎 Interpretasi:
XGBoost memiliki performa lebih baik dibanding Linear Regression.
Kemungkinan penyebab:
• Data tidak sepenuhnya linear — XGBoost mampu menangkap hubungan non-linear.
• Adanya interaksi antar fitur (misal: jumlah likes × durasi review).
• Outlier atau distribusi skewed — XGBoost lebih robust.
""")
elif diff_r2 < -0.02:
    print("""
🔎 Interpretasi:
Linear Regression justru lebih baik dari XGBoost.
Kemungkinan penyebab:
• Hubungan antar fitur dan target bersifat linear.
• Dataset relatif kecil sehingga model kompleks (XGBoost) overfit.
• Regularisasi implicit pada Linear Regression membuat model lebih stabil.
""")
else:
    print("""
🔎 Interpretasi:
Kedua model memiliki performa yang mirip.
Kemungkinan penyebab:
• Hubungan antar fitur sebagian besar linear, tapi ada sedikit pola non-linear.
• Fitur yang digunakan sudah representatif, tanpa noise berlebih.
""")

fi = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n🏆 10 Fitur Paling Berpengaruh (XGBoost):")
display(fi.head(10))

plt.figure(figsize=(8, 5))
sns.barplot(data=fi.head(10), x="Importance", y="Feature", palette="viridis")
plt.title("Top 10 Feature Importances - XGBoost")
plt.tight_layout()
plt.show()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
lr_path = os.path.join(models_dir, f"linear_regression_{ts}.joblib")
xgb_path = os.path.join(models_dir, f"xgboost_regressor_{ts}.joblib")
joblib.dump(lr, lr_path)
joblib.dump(xgb, xgb_path)

results_json = {
    "timestamp": ts,
    "metrics": {
        "Linear Regression": metrics_lr,
        "XGBoost Regressor": metrics_xgb
    },
    "comparison": comparison_df.to_dict(orient="records"),
    "feature_importance": fi.to_dict(orient="records"),
}
with open(os.path.join(models_dir, f"regression_results_{ts}.json"), "w") as f:
    json.dump(results_json, f, indent=2)

print(f"\n✅ Models & results saved to {models_dir}/")
print(f"  • {lr_path}")
print(f"  • {xgb_path}")
print(f"  • regression_results_{ts}.json")
print("\nSelesai ✅")
