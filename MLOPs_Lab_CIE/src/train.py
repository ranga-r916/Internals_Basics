import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import json
import os
import joblib

# -----------------------------
# MLflow setup (IMPORTANT)
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("gamelag-latency-ms")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("../data/training_data.csv")

X = df.drop("latency_ms", axis=1)
y = df["latency_ms"]

# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Metrics function
# -----------------------------
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, r2, mape

# -----------------------------
# Models
# -----------------------------
models = {
    "SVR": SVR(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

results = []

best_model_name = None
best_mae = float("inf")
best_model_obj = None

# -----------------------------
# Training loop
# -----------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("experiment_type", "baseline_comparison")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae, rmse, r2, mape = get_metrics(y_test, preds)

        # log params + metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

        # log model artifact
        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

        # track best model
        if mae < best_mae:
            best_mae = mae
            best_model_name = name
            best_model_obj = model

# -----------------------------
# Save best model (for API)
# -----------------------------
os.makedirs("../models", exist_ok=True)
joblib.dump(best_model_obj, "../models/model.pkl")

# -----------------------------
# Save JSON result
# -----------------------------
os.makedirs("../results", exist_ok=True)

output = {
    "experiment_name": "gamelag-latency-ms",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

with open("../results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Training + MLflow logging completed")
print(f"Best model: {best_model_name}")