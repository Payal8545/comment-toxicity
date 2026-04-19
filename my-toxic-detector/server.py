from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading BERT AI model... this may take a moment on first run to download.")
# Load the pre-trained toxic BERT model
# It will automatically download a ~400MB model the first time you run this script!
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", truncation=True, max_length=512)
print("Model loaded successfully!")

class BatchRequest(BaseModel):
    texts: List[str]

class SingleRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_single(request: SingleRequest):
    # toxic-bert usually returns a list of dictionaries like: [{'label': 'toxic', 'score': 0.98}]
    prediction = toxicity_classifier(request.text)[0]
    
    # Check if the AI is fairly confident the text is toxic (score > 0.50)
    is_toxic = prediction['score'] > 0.50
    
    return {"is_toxic": is_toxic}

@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    if not request.texts:
        return {"results": []}
        
    predictions = toxicity_classifier(request.texts)
    
    results_list = []
    for i, top_pred in enumerate(predictions):
        # Depending on pipeline parsing, it might be nested
        if isinstance(top_pred, list):
            top_pred = top_pred[0]
            
        is_toxic = top_pred['score'] > 0.50
        
        results_list.append({
            "text": request.texts[i],
            "is_toxic": is_toxic
        })
        
    return {"results": results_list}