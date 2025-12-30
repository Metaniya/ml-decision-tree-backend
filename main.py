from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# Create FastAPI app
app = FastAPI(title="Dog Happiness Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now (ok for project)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load Decision Tree model and scaler
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input schema
class DogData(BaseModel):
    walk_minutes: float
    play_minutes: float
    meal_quality: float

# Root endpoint
@app.get("/")
def root():
    return {"message": "Dog Happiness API is running 🐶"}

# Prediction endpoint
@app.post("/predict")
def predict(data: DogData):
    # Convert input to numpy array
    X = np.array([[data.walk_minutes, data.play_minutes, data.meal_quality]])
    # Scale the data
    X_scaled = scaler.transform(X)
    # Make prediction
    prediction = model.predict(X_scaled)
    return {"happy": int(prediction[0])}
