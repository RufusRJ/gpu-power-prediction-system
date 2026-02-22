import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

print("Starting training...")

# Generate Synthetic Dataset
np.random.seed(42)
rows = 2000

Uavg = np.random.uniform(0, 100, rows)
GRAMavg = np.random.uniform(1, 64, rows)
GRAMmax = GRAMavg + np.random.uniform(0, 8, rows)

Pavg = (
    0.6 * Uavg +
    0.05 * GRAMavg +
    0.02 * GRAMmax +
    0.3 * (Uavg * GRAMavg) / 50 +
    np.random.normal(0, 3, rows)
)

data = pd.DataFrame({
    "Uavg": Uavg,
    "GRAMavg": GRAMavg,
    "GRAMmax": GRAMmax,
    "Pavg": Pavg
})

print("Dataset created")

X = data[["Uavg", "GRAMavg", "GRAMmax"]]
y = data["Pavg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = XGBRegressor(
    n_estimators=200,   # reduced for speed
    learning_rate=0.1,
    max_depth=4,
    subsample=0.6
)

model.fit(X_train, y_train)

print("Model trained")

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5

print("RMSE:", rmse)

joblib.dump(model, "gpu_power_model.pkl")

print("Model saved successfully!")