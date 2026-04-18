from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Salary Predictor API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Load artifacts ─────────────────────────────────────────────────────────────
def _load(filename):
    path = os.path.join(BASE_DIR, filename)
    try:
        obj = joblib.load(path)
        print(f"✓ Loaded {filename}")
        return obj
    except Exception as e:
        print(f"✗ Could not load {filename}: {e}")
        return None


model          = _load("final_model/gbr_model.pkl")
label_encoders = _load("final_model/gbr_label_encoders.pkl")  # dict: {col: LabelEncoder}

# NOTE: gbr_scaler.pkl is intentionally NOT used.
# In the notebook, MinMaxScaler was saved AFTER train_data was already scaled to [0,1].
# That means the saved scaler has min≈0, max≈1 and applies no real transformation.
# The model was trained on data scaled from original ranges:
#   yearsExperience:     [0, 24]  → [0, 1]
#   milesFromMetropolis: [0, 100] → [0, 1]
# So we apply that transformation manually here instead.


# ── OHE column order ───────────────────────────────────────────────────────────
# Notebook used sklearn's OneHotEncoder which sorts categories alphabetically.
# This order MUST match training or predictions will be wrong.
OHE_ORDER = {
    "jobType":  ["CEO", "CFO", "CTO", "JANITOR", "JUNIOR", "MANAGER", "SENIOR", "VICE_PRESIDENT"],
    "degree":   ["BACHELORS", "DOCTORAL", "HIGH_SCHOOL", "MASTERS", "NONE"],
    "major":    ["BIOLOGY", "BUSINESS", "CHEMISTRY", "COMPSCI", "ENGINEERING", "LITERATURE", "MATH", "NONE", "PHYSICS"],
    "industry": ["AUTO", "EDUCATION", "FINANCE", "HEALTH", "OIL", "SERVICE", "WEB"],
}

# Original training data ranges (before scaling was applied in the notebook)
TRAINING_RANGES = {
    "yearsExperience":     {"min": 0, "max": 24},
    "milesFromMetropolis": {"min": 0, "max": 100},
}

CAT_COLS = ["jobType", "degree", "major", "industry"]


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    jobType:             str
    degree:              str
    major:               str
    industry:            str
    yearsExperience:     int
    milesFromMetropolis: int


class PredictResponse(BaseModel):
    predicted_salary:  float
    salary_range_low:  float
    salary_range_high: float
    confidence_note:   str


# ── Feature engineering ────────────────────────────────────────────────────────
def build_feature_vector(data: PredictRequest) -> np.ndarray:
    """
    Exact preprocessing pipeline from the notebook:
      1. One-Hot Encode 4 categorical columns (alphabetical sklearn order)
      2. Manually MinMaxScale yearsExperience [0,24]→[0,1] and
         milesFromMetropolis [0,100]→[0,1] using original training ranges
      3. Concatenate → shape (1, 31)  [29 OHE cols + 2 numerical]
    """
    values = {
        "jobType":  data.jobType.upper(),
        "degree":   data.degree.upper(),
        "major":    data.major.upper(),
        "industry": data.industry.upper(),
    }

    # 1. One-Hot Encoding
    ohe_vec = []
    for col in CAT_COLS:
        categories = OHE_ORDER[col]
        val = values[col]
        ohe_vec.extend([1 if cat == val else 0 for cat in categories])

    # 2.  Manual MinMax scaling using ORIGINAL training data ranges.
    #    The saved gbr_scaler.pkl was fit on already-scaled [0,1] data in the
    #    notebook, making it a no-op. We must scale raw inputs ourselves.
    exp_scaled   = data.yearsExperience / TRAINING_RANGES["yearsExperience"]["max"]         # [0,24]  → [0,1]
    miles_scaled = data.milesFromMetropolis / TRAINING_RANGES["milesFromMetropolis"]["max"] # [0,100] → [0,1]

    # Clamp to [0, 1] in case of out-of-range inputs
    exp_scaled   = float(np.clip(exp_scaled,   0.0, 1.0))
    miles_scaled = float(np.clip(miles_scaled, 0.0, 1.0))

    # 3. Concatenate: OHE (29) + yearsExperience + milesFromMetropolis = 31 features
    return np.array(ohe_vec + [exp_scaled, miles_scaled], dtype=float).reshape(1, -1)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict", response_model=PredictResponse)
async def predict(data: PredictRequest):
    features = build_feature_vector(data)

    if model is not None:
        salary = float(model.predict(features)[0])
    else:
        # Demo fallback — when model pkl is absent
        jt_idx  = OHE_ORDER["jobType"].index(data.jobType.upper()) if data.jobType.upper() in OHE_ORDER["jobType"] else 3
        deg_idx = OHE_ORDER["degree"].index(data.degree.upper())   if data.degree.upper()  in OHE_ORDER["degree"]  else 0
        salary  = 60 + jt_idx * 12 + deg_idx * 7 + data.yearsExperience * 2 - data.milesFromMetropolis * 0.3

    salary = max(0.0, salary)
    return PredictResponse(
        predicted_salary  = round(salary, 2),
        salary_range_low  = round(salary * 0.92, 2),
        salary_range_high = round(salary * 1.08, 2),
        confidence_note   = "GBR · 50k job records · OHE + Manual MinMaxScaler pipeline",
    )


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "model_loaded":    model is not None,
        "encoders_loaded": label_encoders is not None,
    }


@app.get("/options")
async def options():
    return {
        "jobTypes":   OHE_ORDER["jobType"],
        "degrees":    OHE_ORDER["degree"],
        "majors":     OHE_ORDER["major"],
        "industries": OHE_ORDER["industry"],
    }