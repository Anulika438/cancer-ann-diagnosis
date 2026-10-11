"""
Module 5 Assignment - Cancer Diagnosis (risk level) using an Artificial Neural Network (TensorFlow/Keras)

Dataset : Cancer Patients and Air Pollution (Kaggle) - cancer_patient_data_sets.xlsx
Task    : Multiclass classification of cancer risk Level (Low / Medium / High)
Run     : python ann_cancer_diagnosis.py   (expects the xlsx in the same folder or /mnt/user-data/uploads)
Outputs : figures/*.png, results.json
"""
import os, json, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

SEED = 42
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
DATA = next(p for p in [os.path.join(HERE, "cancer_patient_data_sets.xlsx"),
                        "/mnt/user-data/uploads/cancer_patient_data_sets.xlsx"] if os.path.exists(p))
CLASSES = ["Low", "Medium", "High"]          # ordinal order -> 0,1,2
results = {}

# ------------------------------------------------------------------ 1. DATA COLLECTION
raw = pd.read_excel(DATA)
print("Raw shape:", raw.shape)
results["raw_shape"] = list(raw.shape)

# ------------------------------------------------------------------ 2. PREPROCESSING
df = raw.copy()
df.columns = [c.strip() for c in df.columns]
df = df.drop(columns=["Patient Id"])                     # identifier, not a predictor
results["missing_values"] = int(df.isna().sum().sum())   # 0 -> nothing to impute
print("Missing values:", results["missing_values"])

features = [c for c in df.columns if c != "Level"]
# Duplicate feature profiles: 848/1000 rows repeat another patient's profile exactly (same label).
n_dup = int(df.duplicated().sum())
conflicts = int(df.drop_duplicates().duplicated(subset=features).sum())
df = df.drop_duplicates().reset_index(drop=True)         # prevents train/test leakage
print(f"Duplicate rows removed: {n_dup}; label conflicts: {conflicts}; unique profiles: {len(df)}")
results.update(duplicates_removed=n_dup, label_conflicts=conflicts, unique_profiles=len(df))

# Outliers: IQR rule on every feature; only Age has meaningful spread. Winsorise (clip) rather than drop,
# because with 152 samples we cannot afford to lose rows.
q1, q3 = df[features].quantile(.25), df[features].quantile(.75)
lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
out_counts = ((df[features] < lo) | (df[features] > hi)).sum()
results["outliers_clipped"] = {k: int(v) for k, v in out_counts.items() if v > 0}
print("Outliers clipped:", results["outliers_clipped"])
df[features] = df[features].clip(lo, hi, axis=1)

y = df["Level"].map({c: i for i, c in enumerate(CLASSES)}).values
X = df[features].values.astype("float32")
results["class_counts"] = df["Level"].value_counts().to_dict()

# Holdout test set (20%) - untouched until final evaluation
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
results["n_dev"], results["n_test"] = len(y_dev), len(y_test)

# ------------------------------------------------------------------ 3. MODEL BUILDING
def build_baseline(n_in):
    """Simple ANN: 2 hidden layers, ReLU, softmax output."""
    return keras.Sequential([
        layers.Input((n_in,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax")])

def build_improved(n_in):
    """Regularised ANN: L2 weight decay + BatchNorm + Dropout (small data -> overfitting risk)."""
    reg = regularizers.l2(1e-3)
    return keras.Sequential([
        layers.Input((n_in,)),
        layers.Dense(64, activation="relu", kernel_regularizer=reg), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_regularizer=reg), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(16, activation="relu", kernel_regularizer=reg),
        layers.Dense(3, activation="softmax")])

# ------------------------------------------------------------------ 4. TRAINING
def fit(builder, Xtr, ytr, Xva, yva, lr, epochs=200, improved=False):
    keras.backend.clear_session(); tf.random.set_seed(SEED)
    m = builder(Xtr.shape[1])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)]
    if improved:
        cbs.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5))
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=0, callbacks=cbs)
    return m, h

def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred), precision=p, recall=r, f1=f)

# Cross-validation on the development set (5-fold) - scaler fitted inside each fold (no leakage)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)
cv = {"baseline": [], "improved": []}
for tr, va in rskf.split(X_dev, y_dev):
    sc = StandardScaler().fit(X_dev[tr]); Xtr, Xva = sc.transform(X_dev[tr]), sc.transform(X_dev[va])
    for name, b, lr, imp in [("baseline", build_baseline, 1e-2, False), ("improved", build_improved, 5e-3, True)]:
        m, _ = fit(b, Xtr, y_dev[tr], Xva, y_dev[va], lr, improved=imp)
        cv[name].append(metrics(y_dev[va], m(Xva, training=False).numpy().argmax(1)))
        print("CV fold done:", name, round(cv[name][-1]["accuracy"],3), flush=True)
cv_summary = {n: {k: (float(np.mean([d[k] for d in v])), float(np.std([d[k] for d in v]))) for k in v[0]}
              for n, v in cv.items()}
results["cv_5fold"] = cv_summary
print("CV summary:", json.dumps(cv_summary, indent=1))

# Final models: hold 15% of dev set for early stopping/validation curves, evaluate on the untouched test set
X_tr, X_va, y_tr, y_va = train_test_split(X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=SEED)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_va_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_test)

final, hist, test_res = {}, {}, {}
for name, b, lr, imp in [("baseline", build_baseline, 1e-2, False), ("improved", build_improved, 5e-3, True)]:
    m, h = fit(b, X_tr_s, y_tr, X_va_s, y_va, lr, improved=imp)
    pred = m(X_te_s, training=False).numpy().argmax(1)
    final[name], hist[name] = m, h.history
    test_res[name] = dict(**metrics(y_test, pred), epochs_run=len(h.history["loss"]), params=int(m.count_params()),
                          report=classification_report(y_test, pred, target_names=CLASSES, output_dict=True, zero_division=0),
                          confusion=confusion_matrix(y_test, pred).tolist())
    print(f"\n=== {name.upper()} (holdout test, n={len(y_test)}) ===")
    print(classification_report(y_test, pred, target_names=CLASSES, zero_division=0))
results["holdout_test"] = test_res

# ------------------------------------------------------------------ DEMONSTRATION: effect of duplicate leakage
# Naive random split on all 1000 original rows (what happens if duplicates are NOT removed)
rawX = raw.drop(columns=["Patient Id", "Level"]).values.astype("float32")
rawy = raw["Level"].map({c: i for i, c in enumerate(CLASSES)}).values
a, b_, c, d = train_test_split(rawX, rawy, test_size=0.2, stratify=rawy, random_state=SEED)
sc = StandardScaler().fit(a)
m, _ = fit(build_baseline, sc.transform(a), c, sc.transform(b_), d, 1e-2)
leak_acc = float(accuracy_score(d, m(sc.transform(b_), training=False).numpy().argmax(1)))
test_rows = pd.DataFrame(b_); train_rows = pd.DataFrame(a)
overlap = float(test_rows.merge(train_rows.drop_duplicates(), how="left", indicator=True)["_merge"].eq("both").mean())
results["leakage_demo"] = dict(naive_split_accuracy=leak_acc, pct_test_rows_seen_in_train=overlap)
print(f"\nNaive (leaky) split accuracy: {leak_acc:.3f}; test rows also in train: {overlap:.1%}")

# Permutation importance on the improved model (holdout test set)
base_acc = accuracy_score(y_test, final["improved"](X_te_s, training=False).numpy().argmax(1))
rng = np.random.default_rng(SEED); imp_scores = {}
for j, f in enumerate(features):
    drops = []
    for _ in range(20):
        Xp = X_te_s.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        drops.append(base_acc - accuracy_score(y_test, final["improved"](Xp, training=False).numpy().argmax(1)))
    imp_scores[f] = float(np.mean(drops))
top = dict(sorted(imp_scores.items(), key=lambda kv: -kv[1])[:10])
results["permutation_importance_top10"] = top

# ------------------------------------------------------------------ FIGURES
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, ls in [("baseline", "--"), ("improved", "-")]:
    ax[0].plot(hist[name]["loss"], ls, label=f"{name} train"); ax[0].plot(hist[name]["val_loss"], ls, alpha=.6, label=f"{name} val")
    ax[1].plot(hist[name]["accuracy"], ls, label=f"{name} train"); ax[1].plot(hist[name]["val_accuracy"], ls, alpha=.6, label=f"{name} val")
ax[0].set(title="Loss (cross-entropy)", xlabel="Epoch"); ax[1].set(title="Accuracy", xlabel="Epoch")
ax[0].legend(fontsize=7); ax[1].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{FIG}/training_curves.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a_, name in zip(ax, ["baseline", "improved"]):
    ConfusionMatrixDisplay(np.array(test_res[name]["confusion"]), display_labels=CLASSES).plot(ax=a_, colorbar=False, cmap="Blues")
    a_.set_title(f"{name.capitalize()} - holdout test")
plt.tight_layout(); plt.savefig(f"{FIG}/confusion_matrices.png", dpi=150); plt.close()

plt.figure(figsize=(7, 4)); plt.barh(list(top)[::-1], list(top.values())[::-1], color="#2a6f97")
plt.xlabel("Mean drop in test accuracy when shuffled"); plt.title("Permutation importance (top 10)")
plt.tight_layout(); plt.savefig(f"{FIG}/feature_importance.png", dpi=150); plt.close()

with open(os.path.join(HERE, "results.json"), "w") as fh: json.dump(results, fh, indent=2, default=float)
print("Done.")
