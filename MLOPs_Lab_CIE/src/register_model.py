import mlflow
import json
import os

def register_model():
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.tracking.MlflowClient()
    
    experiment = client.get_experiment_by_name("gamelag-latency-ms")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"]
    )
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    best_mae = best_run.data.metrics["mae"]
    
    # Check what name was used for logging the model
    # I logged it as `model.name` (e.g. "RandomForest" or "SVR")
    # Let's find the artifact path
    run_name = best_run.data.tags.get("mlflow.runName", "RandomForest")
    artifact_path = run_name
    
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_name = "gamelag-latency-ms-predictor"
    
    # Register the model
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    output = {
        "registered_model_name": model_name,
        "version": int(result.version),
        "run_id": run_id,
        "source_metric": "mae",
        "source_metric_value": best_mae
    }
    
    with open("../results/step4_s6.json", "w") as f:
        json.dump(output, f, indent=4)
        
    print("Model registered and saved to step4_s6.json")

if __name__ == "__main__":
    register_model()
