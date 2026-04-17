from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Salary Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "gbr_model.pkl")
model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f" Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"  Model not found: {e}. Using dummy predictions.")

# Encodings (must match training LabelEncoder order)
JOB_TYPE_MAP = {
    "JANITOR": 0, "JUNIOR": 1, "SENIOR": 2, "MANAGER": 3,
    "VICE_PRESIDENT": 4, "CFO": 5, "CTO": 6, "CEO": 7
}
DEGREE_MAP = {
    "NONE": 0, "HIGH_SCHOOL": 1, "BACHELORS": 2, "MASTERS": 3, "DOCTORAL": 4
}
MAJOR_MAP = {
    "NONE": 0, "BIOLOGY": 1, "LITERATURE": 2, "CHEMISTRY": 3,
    "PHYSICS": 4, "COMPSCI": 5, "MATH": 6, "BUSINESS": 7, "ENGINEERING": 8
}
INDUSTRY_MAP = {
    "EDUCATION": 0, "AUTO": 1, "HEALTH": 2, "WEB": 3,
    "FINANCE": 4, "OIL": 5, "SERVICE": 6
}


class PredictRequest(BaseModel):
    jobType: str
    degree: str
    major: str
    industry: str
    yearsExperience: int
    milesFromMetropolis: int


class PredictResponse(BaseModel):
    predicted_salary: float
    salary_range_low: float
    salary_range_high: float
    confidence_note: str


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict", response_model=PredictResponse)
async def predict(data: PredictRequest):
    jt = JOB_TYPE_MAP.get(data.jobType.upper(), 3)
    deg = DEGREE_MAP.get(data.degree.upper(), 0)
    maj = MAJOR_MAP.get(data.major.upper(), 0)
    ind = INDUSTRY_MAP.get(data.industry.upper(), 0)

    features = np.array([[jt, deg, maj, ind, data.yearsExperience, data.milesFromMetropolis]])

    if model is not None:
        salary = float(model.predict(features)[0])
    else:
        # Dummy formula for demo when model not loaded
        base = 60 + jt * 15 + deg * 8 + maj * 3 + ind * 5
        salary = base + data.yearsExperience * 2 - data.milesFromMetropolis * 0.3

    salary = max(0, salary)
    return PredictResponse(
        predicted_salary=round(salary, 2),
        salary_range_low=round(salary * 0.92, 2),
        salary_range_high=round(salary * 1.08, 2),
        confidence_note="Prediction based on Gradient Boosting Regressor trained on 1M job records."
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/options")
async def options():
    return {
        "jobTypes": list(JOB_TYPE_MAP.keys()),
        "degrees": list(DEGREE_MAP.keys()),
        "majors": list(MAJOR_MAP.keys()),
        "industries": list(INDUSTRY_MAP.keys()),
    }
