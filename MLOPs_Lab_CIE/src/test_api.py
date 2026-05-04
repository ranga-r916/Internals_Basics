import requests
import json
import time

def generate_step2_json():
    ping_res = requests.get("http://localhost:9000/ping").json()
    test_input = {"server_region": 2, "concurrent_players": 6624, "packet_size_kb": 4.7, "is_ranked_match": 0}
    infer_res = requests.post("http://localhost:9000/infer", json=test_input).json()

    output = {
        "health_endpoint": "/ping",
        "predict_endpoint": "/infer",
        "port": 9000,
        "health_response": ping_res,
        "test_input": test_input,
        "prediction": infer_res["prediction"]
    }
    
    with open("../results/step2_s4.json", "w") as f:
        json.dump(output, f, indent=4)
        
    print("Task 2 complete. Saved to step2_s4.json")

if __name__ == "__main__":
    generate_step2_json()
