import pandas as pd
import requests
import time

def simulate():
    # Load data
    train_df = pd.read_csv("../data/training_data.csv")
    new_df = pd.read_csv("../data/new_data.csv")
    
    # 40 normal
    normal_data = train_df.head(40).drop(columns=["latency_ms"]).to_dict(orient="records")
    # if train_df has less than 40 rows, it will just take all of them. Wait, let's check size.
    # We can also sample with replacement if needed. Let's just sample 40.
    if len(normal_data) < 40:
        normal_data = train_df.sample(40, replace=True).drop(columns=["latency_ms"]).to_dict(orient="records")
        
    # 10 drifted (take the ones with highest concurrent_players to ensure drift triggers)
    drifted_data = new_df.sort_values("concurrent_players", ascending=False).head(10).drop(columns=["latency_ms"]).to_dict(orient="records")
        
    print(f"Sending {len(normal_data)} normal requests...")
    for req in normal_data:
        requests.post("http://localhost:9000/infer", json=req)
        time.sleep(0.01)
        
    print(f"Sending {len(drifted_data)} drifted requests...")
    for req in drifted_data:
        requests.post("http://localhost:9000/infer", json=req)
        time.sleep(0.01)
        
    print("Simulation complete.")

if __name__ == "__main__":
    simulate()