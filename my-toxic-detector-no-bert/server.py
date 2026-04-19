from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from better_profanity import profanity
from hatesonar import Sonar

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


def _scores_by_class(sonar_result: Dict[str, Any]) -> Dict[str, float]:
    return {c["class_name"]: float(c["confidence"]) for c in sonar_result["classes"]}


def is_toxic_text(text: str) -> bool:
    """Lexicon / classical ML only — no BERT or Transformers."""
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
