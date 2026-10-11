from pathlib import Path
import urllib.request
import zipfile
import io
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

try:
    from backend.preprocess import clean
    from backend.features import get_vectorizer
except ImportError:
    from preprocess import clean
    from features import get_vectorizer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model.joblib"

LABEL_MAPPING = {
    "pants-fire": "FAKE",
    "false": "FAKE",
    "barely-true": "FAKE",
    "half-true": "REAL",
    "mostly-true": "REAL",
    "true": "REAL"
}


def ensure_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_file = DATA_DIR / "train.tsv"
    test_file = DATA_DIR / "test.tsv"

    if not train_file.exists() or not test_file.exists():
        url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
                    z.extractall(DATA_DIR)
        except Exception:
            raise RuntimeError(
                "Dataset download failed. Please provide a path to the LIAR CSV/TSV dataset."
            )

    return train_file, test_file


def train_pipeline():
    train_path, test_path = ensure_data()

    train_df = pd.read_csv(train_path, sep="\t", header=None)
    test_df = pd.read_csv(test_path, sep="\t", header=None)

    train_df = train_df.dropna(subset=[1, 2])
    test_df = test_df.dropna(subset=[1, 2])

    train_df = train_df[train_df[1].isin(LABEL_MAPPING)]
    test_df = test_df[test_df[1].isin(LABEL_MAPPING)]

    X_train_clean = [clean(t) for t in train_df[2]]
    y_train = train_df[1].map(LABEL_MAPPING).values

    X_test_clean = [clean(t) for t in test_df[2]]
    y_test = test_df[1].map(LABEL_MAPPING).values

    vectorizer = get_vectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train_clean)
    X_test_vec = vectorizer.transform(X_test_clean)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="FAKE", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="FAKE", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="FAKE", zero_division=0)

    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    joblib.dump({"vectorizer": vectorizer, "model": model}, MODEL_PATH)


if __name__ == "__main__":
    train_pipeline()
