from typing import Any
import numpy as np

try:
    from backend.classifier import load_artifacts, predict_proba
    from backend.preprocess import clean
except ImportError:
    from classifier import load_artifacts, predict_proba
    from preprocess import clean


def _fallback_explain(text: str, top_k: int = 8) -> list[dict[str, Any]]:
    vectorizer, model = load_artifacts()
    cleaned = clean(text)
    if not cleaned:
        return []

    X = vectorizer.transform([cleaned])
    feature_indices = X.nonzero()[1]
    if len(feature_indices) == 0:
        return []

    inv_vocab = {idx: word for word, idx in vectorizer.vocabulary_.items()}
    coef = model.coef_[0]

    exps = []
    for idx in feature_indices:
        word = inv_vocab.get(idx)
        if not word:
            continue
        # In scikit-learn binary clf, class 0 is FAKE and class 1 is REAL.
        # Positive coef increases REAL probability.
        # Required: positive weight pushes FAKE, negative pushes REAL.
        weight = float(-coef[idx] * X[0, idx])
        direction = "FAKE" if weight > 0 else ("REAL" if weight < 0 else "NEUTRAL")
        exps.append({
            "word": word,
            "weight": round(weight, 4),
            "direction": direction
        })

    exps.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return exps[:top_k]


def explain(text: str) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []

    try:
        from lime.lime_text import LimeTextExplainer

        vectorizer, model = load_artifacts()
        classes = list(model.classes_)
        fake_idx = classes.index("FAKE") if "FAKE" in classes else 0

        explainer = LimeTextExplainer(class_names=classes)
        exp = explainer.explain_instance(
            text,
            predict_proba,
            num_features=8,
            num_samples=500,
            labels=[fake_idx]
        )
        weights = exp.as_list(label=fake_idx)
        results = []
        for word, weight in weights:
            weight_val = float(round(weight, 4))
            direction = "FAKE" if weight_val > 0 else ("REAL" if weight_val < 0 else "NEUTRAL")
            results.append({
                "word": word,
                "weight": weight_val,
                "direction": direction
            })
        return results[:8]
    except Exception:
        return _fallback_explain(text, top_k=8)
