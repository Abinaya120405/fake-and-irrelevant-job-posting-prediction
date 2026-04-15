import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import vstack, csr_matrix, load_npz

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = r"C:\Users\abi\Downloads\irrelevent"
CLEAN_PATH      = os.path.join(BASE_DIR, "cleaned_jobs.csv")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
FEATURES_PATH   = os.path.join(BASE_DIR, "features.npz")
META_COLS_PATH  = os.path.join(BASE_DIR, "meta_features.npy")
MODEL_PATH      = os.path.join(BASE_DIR, "pac_model.pkl")
SCALER_PATH     = os.path.join(BASE_DIR, "scaler.pkl")
REPORT_PATH     = os.path.join(BASE_DIR, "evaluation_report.txt")
CM_PLOT_PATH    = os.path.join(BASE_DIR, "confusion_matrix.png")


def load_data():
    print("[INFO] Loading cleaned dataset...")
    df = pd.read_csv(CLEAN_PATH)
    df = df.reset_index(drop=True)

    if "fraudulent" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"fraudulent": "label"})
    elif "label" not in df.columns:
        raise KeyError("No 'label' or 'fraudulent' column found in cleaned_jobs.csv")

    print(f"  Loaded {len(df):,} rows")

    print("[INFO] Loading TF-IDF vectorizer...")
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("[INFO] Loading saved feature matrix...")
    X = load_npz(FEATURES_PATH)
    print(f"  Feature matrix shape: {X.shape}")

    print("[INFO] Loading meta feature column names...")
    meta_cols = list(np.load(META_COLS_PATH, allow_pickle=True))

    y = df["label"].values

    return df, X, y, vectorizer, meta_cols


def add_irrelevant_class(df_train: pd.DataFrame, X_train, *, ratio: float = 0.05):
    print("[INFO] Generating 'irrelevant' class samples...")
    X_train = X_train.tocsr()

    genuine_positions = np.where(df_train["label"].values == 0)[0]
    n_irr = max(100, int(len(genuine_positions) * ratio))
    sampled_pos = np.random.RandomState(42).choice(
        genuine_positions, size=n_irr, replace=False
    )

    irr_dense = X_train[sampled_pos].toarray()
    rng = np.random.RandomState(0)
    for row in irr_dense:
        rng.shuffle(row)

    X_irr  = csr_matrix(irr_dense)
    y_irr  = np.full(n_irr, fill_value=2, dtype=int)
    X_aug  = vstack([X_train, X_irr])
    y_aug  = np.concatenate([df_train["label"].values, y_irr])

    print(f"  Genuine    : {np.sum(y_aug == 0):,}")
    print(f"  Fake       : {np.sum(y_aug == 1):,}")
    print(f"  Irrelevant : {np.sum(y_aug == 2):,}")

    return X_aug, y_aug


def evaluate(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)

    # Dynamically detect present classes
    