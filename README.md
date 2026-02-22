# GPU Power Prediction System

## Overview

This project implements a Machine Learning-based GPU power consumption prediction system with a modern web interface.

The system predicts average GPU power consumption using:

- GPU Utilization (%)
- Average GPU Memory Usage (GiB)
- Maximum GPU Memory Usage (GiB)

The backend is built using FastAPI and a trained XGBoost regression model.  
The frontend is developed using React.

---

## Problem Statement

Data centers consume significant energy due to GPU-intensive workloads. Accurately predicting GPU power consumption enables improved workload scheduling, reduced operational costs, and better energy efficiency.

This project models the nonlinear relationship between GPU utilization, memory usage, and power consumption using supervised machine learning.

---

## Machine Learning Model

- Model Type: Supervised Regression  
- Algorithm: XGBoost Regressor  
- Features:
  - Uavg (GPU Utilization)
  - GRAMavg (Average GPU Memory Usage)
  - GRAMmax (Maximum GPU Memory Usage)
- Target:
  - Pavg (Average GPU Power in Watts)
- Evaluation Metric:
  - Root Mean Square Error (RMSE)

Synthetic workload data is generated using realistic statistical ranges inspired by real-world GPU cluster characteristics.

---

## Project Structure

gpu-power-prediction-system/
│
├── backend/
│   ├── app.py
│   ├── train.py
│   ├── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── package.json
│
├── README.md
├── .gitignore

---
## How to run:
cd backend
uvicorn app:app --reload

cd frontend
npm start
click on the npm link and use the website

## Complete Setup Instructions

### 1. Clone the Repository

    git clone https://github.com/yourusername/gpu-power-prediction-system.git
    cd gpu-power-prediction-system

---

## Backend Setup

Navigate to backend directory:

    cd backend

Install Python dependencies:

    pip install -r requirements.txt

If requirements.txt is not available:

    pip install fastapi uvicorn xgboost pandas scikit-learn joblib

Train the model (this generates synthetic data and saves the model file):

    python train.py

After training, a file named `gpu_power_model.pkl` will be created inside the backend directory.

Start the backend server:

    uvicorn app:app --reload

Backend runs at:

    http://localhost:8000

API documentation is available at:

    http://localhost:8000/docs

Keep this terminal running.

---

## Frontend Setup

Open a new terminal window.

Navigate to frontend directory:

    cd frontend

Install Node dependencies:

    npm install

Start the frontend application:

    npm start

Frontend runs at:

    http://localhost:3000

The frontend communicates with the backend API running on port 8000.

---

## How to Use the System

1. Ensure backend is running.
2. Ensure frontend is running.
3. Open the browser at http://localhost:3000
4. Enter:
   - GPU Utilization (%)
   - GRAM Average (GiB)
   - GRAM Maximum (GiB)
5. Click "Predict Power"
6. The system will display predicted GPU power in Watts.

---

## Example Input

GPU Utilization: 85  
GRAM Average: 4.2  
GRAM Maximum: 4.8  

Example Output:

Predicted GPU Power: Approximately 60–65 Watts

---

## Notes

- npm install is required only once per machine.
- The backend must be running before using the frontend.
- The model file (gpu_power_model.pkl) is generated after running train.py.
- The project runs entirely on localhost.
- Compatible with Python 3.9+ and Node.js 16+.

---

## Future Improvements

- Integration with real GPU workload traces
- Multi-GPU workload simulation support
- Model comparison (LightGBM, CatBoost, LSTM)
- Real-time monitoring dashboard
- Deployment to cloud infrastructure

---

## Author

Rufus Ebenezer 
B.Tech Infromation Technology  
Machine Learning Project