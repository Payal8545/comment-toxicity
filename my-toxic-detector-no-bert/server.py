from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from better_profanity import profanity
from hatesonar import Sonar
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading non-BERT toxicity models (HateSonar + profanity filter)...")
sonar = Sonar()
print("Models ready.")

DATASET_PATH = Path(__file__).with_name("dataset.csv")


def _load_dataset_model(dataset_path: Path) -> Tuple[Optional[TfidfVectorizer], Optional[LogisticRegression]]:
    if not dataset_path.exists():
        return None, None

    df = pd.read_csv(dataset_path)
    if "text" not in df.columns or "label" not in df.columns:
        return None, None

    texts = df["text"].astype(str).fillna("").tolist()
    labels = df["label"].astype(int).tolist()
    if len(texts) < 4 or len(set(labels)) < 2:
        return None, None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
    )
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, labels)
    return vectorizer, clf


print(f"Loading dataset model from {DATASET_PATH} ...")
dataset_vectorizer, dataset_clf = _load_dataset_model(DATASET_PATH)
if dataset_vectorizer is None or dataset_clf is None:
    print("No usable dataset model found; using HateSonar/profanity fallback only.")
else:
    print("Dataset model ready.")


def _scores_by_class(sonar_result: Dict[str, Any]) -> Dict[str, float]:
    return {c["class_name"]: float(c["confidence"]) for c in sonar_result["classes"]}


def is_toxic_text(text: str) -> bool:
    """Lexicon / classical ML only — no BERT or Transformers."""
    if dataset_vectorizer is not None and dataset_clf is not None:
        X = dataset_vectorizer.transform([text])
        proba = float(dataset_clf.predict_proba(X)[0][1])
        return proba >= 0.5

    if profanity.contains_profanity(text):
        return True
    result = sonar.ping(text)
    if result["top_class"] in ("hate_speech", "offensive_language"):
        return True
    s = _scores_by_class(result)
    if s.get("hate_speech", 0) >= 0.22:
        return True
    neither = s.get("neither", 1.0)
    offensive = s.get("offensive_language", 0.0)
    n = len(text.strip())
    # HateSonar often inflates "offensive_language" on long benign paragraphs (e.g. mentions of "model",
    # enthusiasm). Use a tighter "neither" gate for long text; short insults still dip "neither" enough
    # to pass the looser short-text rule.
    if n <= 90:
        if neither < 0.62 and offensive >= 0.30:
            return True
    else:
        if neither < 0.53 and offensive >= 0.36:
            return True
    return False


class BatchRequest(BaseModel):
    texts: List[str]


class SingleRequest(BaseModel):
    text: str


@app.post("/predict")
def predict_single(request: SingleRequest):
    return {"is_toxic": is_toxic_text(request.text)}


@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    if not request.texts:
        return {"results": []}
    results = []
    for t in request.texts:
        results.append({"text": t, "is_toxic": is_toxic_text(t)})
    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
