from pathlib import Path
import joblib
import numpy as np

try:
    from backend.preprocess import clean
except ImportError:
    from preprocess import clean

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
_artifacts = None


def load_artifacts():
    global _artifacts
    if _artifacts is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train.py first.")
        data = joblib.load(MODEL_PATH)
        if isinstance(data, dict):
            _artifacts = (data["vectorizer"], data["model"])
        else:
            _artifacts = data
    return _artifacts


def predict(text: str) -> tuple[str, float]:
    vectorizer, model = load_artifacts()
    cleaned = clean(text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    idx = int(np.argmax(probs))
    label = str(model.classes_[idx])
    confidence = float(probs[idx])
    return label, confidence


def predict_proba(texts: list[str]) -> np.ndarray:
    vectorizer, model = load_artifacts()
    cleaned = [clean(t) for t in texts]
    vec = vectorizer.transform(cleaned)
    return model.predict_proba(vec)
