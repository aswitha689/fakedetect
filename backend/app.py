from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from backend.classifier import predict
    from backend.explain import explain
    from backend.grounding import search_claims
    from backend import db
except ImportError:
    from classifier import predict
    from explain import explain
    from grounding import search_claims
    import db

app = FastAPI(title="Fake News Detection API")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Fake News Detection API is running"}


@app.on_event("startup")
def startup_event():
    db.init_db()


class PredictRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    prediction_id: int
    user_label: str
    comment: Optional[str] = None


class ApiKeyRequest(BaseModel):
    api_key: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/settings/api-key")
def get_api_key_endpoint():
    from backend.grounding import get_api_key
    key = get_api_key()
    return {"configured": bool(key)}


@app.post("/settings/api-key")
def set_api_key_endpoint(req: ApiKeyRequest):
    import os
    key = req.api_key.strip()
    os.environ["FACTCHECK_API_KEY"] = key
    env_file = Path(__file__).resolve().parent.parent / ".env"
    env_file.write_text(f"FACTCHECK_API_KEY={key}\n", encoding="utf-8")
    return {"status": "success", "configured": bool(key)}


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Lifecycle: Submitted
    stmt_id = db.insert_statement(text, status="Submitted")

    # Lifecycle: Classified
    label, confidence = predict(text)
    pred_id = db.insert_prediction(stmt_id, label, confidence)
    db.update_statement_status(stmt_id, "Classified")

    # Explanations
    explanations = explain(text)
    db.insert_explanations(pred_id, explanations)

    # Knowledge Grounding (Google Fact Check + Verified Knowledge Base)
    grounding = search_claims(text)
    if grounding.get("matches"):
        db.insert_knowledge_facts(stmt_id, grounding["matches"])

    # Lifecycle & Grounding Decision
    verified_true = any(m.get("is_verified_true") for m in grounding.get("matches", []))
    verified_false = any(m.get("is_verified_false") for m in grounding.get("matches", []))

    if verified_true:
        label = "REAL"
        confidence = 0.95
        final_status = "Fact-Check Verified"
        db.update_prediction(pred_id, label, confidence)
    elif verified_false:
        label = "FAKE"
        confidence = 0.92
        final_status = "Fact-Check Refuted"
        db.update_prediction(pred_id, label, confidence)
    elif confidence < 0.60:
        final_status = "Needs Review"
    else:
        final_status = "Explained"

    db.update_statement_status(stmt_id, final_status)

    return {
        "statement_id": stmt_id,
        "prediction_id": pred_id,
        "label": label,
        "confidence": round(confidence, 4),
        "status": final_status,
        "explanation": explanations,
        "grounding": grounding,
    }


@app.get("/history")
def history_endpoint():
    return db.get_history(limit=20)


@app.get("/statement/{statement_id}")
def statement_endpoint(statement_id: int):
    stmt = db.get_statement(statement_id)
    if not stmt:
        raise HTTPException(status_code=404, detail="Statement not found.")
    return stmt


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    fid = db.insert_feedback(req.prediction_id, req.user_label, req.comment)
    return {"status": "success", "feedback_id": fid}


@app.delete("/statement/{statement_id}")
def delete_statement_endpoint(statement_id: int):
    success = db.delete_statement(statement_id)
    if not success:
        raise HTTPException(status_code=404, detail="Statement not found.")
    return {"status": "success", "deleted_id": statement_id}
