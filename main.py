from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI(title="Dog Happiness Prediction API")

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
