from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# Enable CORS (VERY IMPORTANT for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("gpu_power_model.pkl")

@app.post("/predict")
def predict(data: dict):
    Uavg = data["Uavg"]
    GRAMavg = data["GRAMavg"]
    GRAMmax = data["GRAMmax"]

    features = np.array([[Uavg, GRAMavg, GRAMmax]])
    prediction = model.predict(features)

    return {"Predicted_GPU_Power": float(prediction[0])}