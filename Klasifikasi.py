"""
Fixed version: ensure JSON-serializable evaluation output by converting numpy objects to native Python types.
"""
import os
import json
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

RANDOM_STATE = 42
processed_dir = "./processed_model_data"
models_dir = "./trained_models"
os.makedirs(models_dir, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

def safe_np_load(path):
    """Load .npy robustly; retry with allow_pickle=True if necessary."""
    try:
        return np.load(path)
    except ValueError as e:
        msg = str(e)
        if "allow_pickle" in msg or "Object arrays cannot be loaded" in msg:
            print(f"⚠️  np.load ValueError for {path} — retrying with allow_pickle=True")
            return np.load(path, allow_pickle=True)
        raise

def load_processed_data(processed_dir):
    files = {
        'X_train': 'X_train.npy',
        'X_val':   'X_val.npy',
        'X_test':  'X_test.npy',
        'y_train': 'y_train.npy',
        'y_val':   'y_val.npy',
        'y_test':  'y_test.npy',
        'selected_features': 'selected_features.csv'
    }
    data = {}
    missing = []
    for k, fname in files.items():
        p = os.path.join(processed_dir, fname)
        if os.path.exists(p):
            if p.endswith(".npy"):
                data[k] = safe_np_load(p)
            else:
                data[k] = pd.read_csv(p)
        else:
            missing.append(p)
    if missing:
        raise FileNotFoundError(f"Missing processed files: {missing}\nEnsure the processed files exist in {processed_dir}")
    return data

def to_np(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    return np.asarray(x)

def to_serializable(obj):
    """
    Convert numpy types recursively into Python built-ins so json.dump works.
    Handles: ndarray, numpy scalar types, dicts, lists, tuples.
    """
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalar (e.g., np.int64, np.float64)
    if isinstance(obj, np.generic):
        return obj.item()
    # pandas types
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    # dict -> recurse
    if isinstance(obj, dict):
        return {to_serializable(k): to_serializable(v) for k, v in obj.items()}
    # list/tuple -> recurse
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    # other basic types are fine
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # fallback: try to cast
    try:
        return obj.tolist()
    except Exception:
        try:
            return obj.__dict__
        except Exception:
            return str(obj)

def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def interpret_confusion(cm, classes):
    insights = []
    for i, cls in enumerate(classes):
        total = int(cm[i, :].sum())
        correct = int(cm[i, i])
        acc = (correct / total) if total > 0 else 0.0
        row = cm[i, :].copy()
        row[i] = -1
        most_confused_idx = int(np.argmax(row))
        most_confused_label = classes[most_confused_idx]
        confused_count = int(cm[i, most_confused_idx])
        if confused_count > 0:
            insights.append(
                f"Class {cls}: {correct}/{total} correct ({acc*100:.1f}%). "
                f"Most confused with class {most_confused_label} ({confused_count} cases)."
            )
        else:
            insights.append(f"Class {cls}: {correct}/{total} correct ({acc*100:.1f}%). Low confusion.")
    return insights

print("\n== LOADING PROCESSED DATA ==")
try:
    data = load_processed_data(processed_dir)
except Exception as e:
    raise SystemExit(f"Failed to load processed data: {e}")

X_train = to_np(data['X_train'])
X_val   = to_np(data['X_val'])
X_test  = to_np(data['X_test'])
y_train = to_np(data['y_train'])
y_val   = to_np(data['y_val'])
y_test  = to_np(data['y_test'])
sel_df  = data.get('selected_features', None)

label_mapping = None
if y_train.dtype.kind in ('U','S','O'):
    unique_labels = np.unique(y_train)
    label_mapping = {lab: int(i) for i, lab in enumerate(unique_labels)}
    print("ℹ️  Converting string labels to integers:", label_mapping)
    y_train = np.array([label_mapping[x] for x in y_train])
    y_val   = np.array([label_mapping.get(x, -1) for x in y_val])
    y_test  = np.array([label_mapping.get(x, -1) for x in y_test])
else:
    if not np.issubdtype(y_train.dtype, np.integer):
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)

if isinstance(sel_df, pd.DataFrame) and 'feature' in sel_df.columns:
    selected_features = sel_df['feature'].tolist()
else:
    selected_features = [f"f_{i}" for i in range(X_train.shape[1])]

print(f"✓ Data shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

is_sparse = hasattr(X_train, "toarray") or "scipy" in str(type(X_train))
scaler = None
if not is_sparse:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
else:
    print("⚠️  Sparse-like input detected — skipping StandardScaler.")

svc = LinearSVC(
    penalty='l2', loss='squared_hinge', C=1.0,
    max_iter=4000, random_state=RANDOM_STATE,
    class_weight='balanced'
)
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_split=5, min_samples_leaf=2,
    max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
)

print("\n== TRAINING ==")
t0 = time()
svc.fit(X_train, y_train)
svc_time = time() - t0
t0 = time()
rf.fit(X_train, y_train)
rf_time = time() - t0

def get_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    prec_per, rec_per, f1_per, support_per = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {
        'accuracy': acc,
        'precision_weighted': float(prec_w),
        'recall_weighted': float(rec_w),
        'f1_weighted': float(f1_w),
        'precision_per_class': [float(x) for x in prec_per],
        'recall_per_class': [float(x) for x in rec_per],
        'f1_per_class': [float(x) for x in f1_per],
        'support_per_class': [int(x) for x in support_per]
    }

results = {}
for name, model in [('LinearSVC', svc), ('RandomForest', rf)]:
    y_pred = model.predict(X_test)
    metrics = get_metrics(y_test, y_pred)
    classes = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cr_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results[name] = {
        'metrics': metrics,
        'confusion_matrix': cm,             # will sanitize later
        'classes': classes,
        'classification_report': cr_dict    # contains numpy types possibly
    }
    print(f"\n-- {name} --")
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1(weighted): {metrics['f1_weighted']:.4f}")
    plot_confusion_matrix(cm, classes, title=f"{name} - Confusion Matrix")
    for line in interpret_confusion(cm, classes):
        print("  •", line)

fi = rf.feature_importances_
fi_df = pd.DataFrame({'feature': selected_features, 'importance': fi}).sort_values('importance', ascending=False)

out_json = {
    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    'models': {
        'LinearSVC': {
            'path': None,
            'train_time_s': svc_time,
            'metrics': results['LinearSVC']['metrics'],
            'confusion_matrix': results['LinearSVC']['confusion_matrix'],
            'classification_report': results['LinearSVC']['classification_report']
        },
        'RandomForest': {
            'path': None,
            'train_time_s': rf_time,
            'metrics': results['RandomForest']['metrics'],
            'confusion_matrix': results['RandomForest']['confusion_matrix'],
            'classification_report': results['RandomForest']['classification_report'],
            'feature_importance': fi_df.to_dict(orient='records')
        }
    },
    'selected_features': selected_features,
    'label_mapping': label_mapping
}

ts = out_json['timestamp']
svc_path = os.path.join(models_dir, f"linear_svc_{ts}.joblib")
rf_path  = os.path.join(models_dir, f"random_forest_{ts}.joblib")
joblib.dump(svc, svc_path)
joblib.dump(rf, rf_path)
out_json['models']['LinearSVC']['path'] = svc_path
out_json['models']['RandomForest']['path'] = rf_path
print(f"\nSaved models to: {svc_path} and {rf_path}")

sanitized = to_serializable(out_json)

out_path = os.path.join(models_dir, f"evaluation_results_{ts}.json")
with open(out_path, "w") as f:
    json.dump(sanitized, f, indent=2)

print(f"Saved evaluation JSON -> {out_path}")
print("\nDone ✅")
