"""SMS Spam Classifier backend.

Loads the pre-trained `model/spam_classifier.joblib` bundle
(CountVectorizer + Multinomial Naive Bayes) trained in Google Colab on the
UCI SMS Spam Collection dataset and exposes it over a REST API for inference.
No retraining happens here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# The bundle was trained with scikit-learn 1.6.1; loading with a newer
# version emits an InconsistentVersionWarning which is safe to silence here.
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

MODEL_PATH = Path(__file__).resolve().parent / "model" / "spam_classifier.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
PIPELINE = bundle["model"]
MODEL_NAME: str = bundle.get("model_name", "CountVectorizer + Multinomial Naive Bayes")
POSITIVE_CLASS: str = bundle.get("positive_class", "spam")
METRICS = {
    key: (float(value) if hasattr(value, "item") else value)
    for key, value in (bundle.get("test_metrics") or {}).items()
}
CLASSES = list(PIPELINE.classes_)


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000,
                         description="The SMS message to classify.")


class PredictResponse(BaseModel):
    message: str
    label: str
    is_spam: bool
    confidence: float
    probabilities: dict[str, float]


app = FastAPI(
    title="SMS Spam Classifier API",
    description="Classifies SMS messages as spam or ham using a pre-trained "
                "CountVectorizer + Multinomial Naive Bayes model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": True, "model_name": MODEL_NAME}


@app.get("/api/model")
def model_info() -> dict:
    return {
        "model_name": MODEL_NAME,
        "positive_class": POSITIVE_CLASS,
        "classes": CLASSES,
        "test_metrics": METRICS,
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message must not be empty.")

    probabilities = PIPELINE.predict_proba([message])[0]
    label = str(PIPELINE.predict([message])[0])
    probs = {cls: round(float(p), 6) for cls, p in zip(CLASSES, probabilities)}

    return PredictResponse(
        message=message,
        label=label,
        is_spam=label == POSITIVE_CLASS,
        confidence=probs[label],
        probabilities=probs,
    )


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("APP_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
