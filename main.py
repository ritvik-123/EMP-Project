import os
import uuid
import random
import logging

from functools import lru_cache
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from google.cloud import firestore

from inference import EMPClassifier

logger = logging.getLogger("emp")

app = FastAPI(title="EMP Oppression Classifier")
templates = Jinja2Templates(directory="templates")

try:
    db = firestore.Client()
except Exception as e:
    # If Firestore itself can't even initialize (e.g. API disabled, no
    # database provisioned), don't take the whole app down with it --
    # classification should still work, just without feedback logging.
    logger.error(f"Firestore client failed to initialize: {e}")
    db = None

# Fraction of *confirmed* feedback permanently routed to the frozen eval pool.
# Assigned once, at feedback time, and never moved afterward.
EVAL_POOL_FRACTION = float(os.environ.get("EVAL_POOL_FRACTION", "0.15"))


@lru_cache(maxsize=1)
def get_classifier():
    return EMPClassifier()


class PredictionRequest(BaseModel):
    sentence: str
    use_llm_backup: bool = True


class FeedbackRequest(BaseModel):
    prediction_id: str
    is_correct: bool
    corrected_label: str | None = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/predict")
def predict(payload: PredictionRequest):
    classifier = get_classifier()
    result = classifier.classify(
        sentence=payload.sentence,
        use_llm_backup=payload.use_llm_backup
    )

    prediction_id = str(uuid.uuid4())

    if db is not None:
        try:
            db.collection("predictions").document(prediction_id).set({
                "sentence": payload.sentence,
                "predicted_label": result["final_label"],
                "source": result["source"],
                "model_version": classifier.model_version,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "feedback": None,
                "corrected_label": None,
                "pool": None,          # assigned only once feedback is confirmed
            })
        except Exception as e:
            # Logging failed -- the user still gets their classification.
            # This prediction just won't be available for feedback/retraining.
            logger.error(f"Failed to log prediction {prediction_id}: {e}")

    result["prediction_id"] = prediction_id
    return result


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    if db is None:
        return {"status": "error", "detail": "feedback storage unavailable"}

    try:
        doc_ref = db.collection("predictions").document(payload.prediction_id)
        doc = doc_ref.get()
        if not doc.exists:
            return {"status": "error", "detail": "unknown prediction_id"}

        # Pool assignment happens exactly once, the first time feedback lands,
        # and is never changed again -- this is what keeps the eval set frozen.
        existing = doc.to_dict()
        pool = existing.get("pool")
        if pool is None:
            pool = "eval" if random.random() < EVAL_POOL_FRACTION else "train"

        doc_ref.update({
            "feedback": "correct" if payload.is_correct else "incorrect",
            "corrected_label": payload.corrected_label,
            "pool": pool,
        })
        return {"status": "recorded", "pool": pool}
    except Exception as e:
        logger.error(f"Failed to record feedback for {payload.prediction_id}: {e}")
        return {"status": "error", "detail": "failed to record feedback"}


@app.get("/health")
def health():
    return {"status": "ok"}