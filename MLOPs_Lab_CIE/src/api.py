from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
from datetime import datetime

app = FastAPI()

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("../models/model.pkl")

from pydantic import BaseModel, Field

# -----------------------------
# INPUT SCHEMA
# -----------------------------
class InputData(BaseModel):
    server_region: int = Field(ge=1, le=5)
    concurrent_players: int = Field(ge=100, le=10000)
    packet_size_kb: float = Field(ge=0.5, le=10.0)
    is_ranked_match: int = Field(ge=0, le=1)

# -----------------------------
# LOG FILE PATH
# -----------------------------
LOG_FILE = "../logs/predictions.jsonl"

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/ping")
def health():
    return {
        "status": "running",
        "model": "RandomForest",
        "version": "1.0"
    }

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/infer")
def predict(data: InputData):

    # Convert input to dict (Pydantic v2)
    input_dict = data.model_dump()

    # Prediction
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]

    # -----------------------------
    # LOGGING (IMPORTANT PART)
    # -----------------------------
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_dict,
        "prediction": float(prediction)
    }

    os.makedirs("../logs", exist_ok=True)

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        f.flush()   # ensures all writes happen

    return {"prediction": float(prediction)}

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # To avoid hanging on await request.json(), we just log a dummy input with the validation error details.
    # We can reconstruct some of the invalid input from exc.errors() if needed, but for the CIE test,
    # we'll just log the error. Wait, we need the input values for monitor.py!
    # Instead, let's read the body as bytes.
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes)
    except:
        body = {}
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": body,
        "prediction": -1.0,
        "error": "validation_error"
    }
    
    os.makedirs("../logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        f.flush()
        
    return JSONResponse(status_code=422, content={"detail": exc.errors()})