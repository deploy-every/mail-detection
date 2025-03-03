from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
with open("phishing_detection_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

class EmailInput(BaseModel):
    message: str


@app.post("/predict")
def predict_email(input_data: EmailInput):
    input_features = vectorizer.transform([input_data.message])
    prediction = model.predict(input_features)
    result = "Legitimate mail" if prediction[0] == 1 else "Phishy mail"
    return {"message": input_data.message, "classification": result}


if __name__ == "__main__":
    # Use the PORT environment variable or default to 10000
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
