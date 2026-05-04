import json
import pandas as pd
import numpy as np

def monitor():
    # Load training data to get train_mean
    train_df = pd.read_csv("../data/training_data.csv")
    train_mean_cp = train_df["concurrent_players"].mean()
    train_mean_ps = train_df["packet_size_kb"].mean()
    
    # Load live logs
    logs = []
    with open("../logs/predictions.jsonl", "r") as f:
        for line in f:
            logs.append(json.loads(line))
            
    # Calculate overall stats
    predictions = [log["prediction"] for log in logs if log["prediction"] != -1.0]
    mean_pred = np.mean(predictions) if predictions else 0.0
    
    # Calculate live mean on the recent window (e.g., the 10 drifted requests)
    # to properly detect the shift as requested.
    recent_logs = logs[-10:] if len(logs) >= 10 else logs
    
    live_cp = [log["input"]["concurrent_players"] for log in recent_logs]
    live_ps = [log["input"]["packet_size_kb"] for log in recent_logs]
    
    live_mean_cp = np.mean(live_cp)
    live_mean_ps = np.mean(live_ps)
    
    shift_cp = abs(live_mean_cp - train_mean_cp)
    shift_ps = abs(live_mean_ps - train_mean_ps)
    
    threshold_cp = 2909.96
    threshold_ps = 1.48
    
    drift_detected = (shift_cp > threshold_cp) or (shift_ps > threshold_ps)
    
    alerts = []
    if shift_cp > threshold_cp:
        alerts.append({
            "feature": "concurrent_players",
            "train_mean": round(train_mean_cp, 2),
            "live_mean": round(live_mean_cp, 2),
            "shift": round(shift_cp, 2),
            "threshold": threshold_cp,
            "status": "ALERT"
        })
    if shift_ps > threshold_ps:
        alerts.append({
            "feature": "packet_size_kb",
            "train_mean": round(train_mean_ps, 2),
            "live_mean": round(live_mean_ps, 2),
            "shift": round(shift_ps, 2),
            "threshold": threshold_ps,
            "status": "ALERT"
        })
        
    output = {
        "total_predictions": len(logs),
        "mean_prediction": round(float(mean_pred), 2),
        "drift_detected": bool(drift_detected),
        "alerts": alerts
    }
    
    with open("../results/step3_s5.json", "w") as f:
        json.dump(output, f, indent=4)
        
    print("Monitor complete. Output saved to step3_s5.json")

if __name__ == "__main__":
    monitor()