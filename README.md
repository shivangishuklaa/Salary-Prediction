# 💰 SalaryLens — Salary Predictor

FastAPI app jo `gbr_model.pkl` (Gradient Boosting Regressor) ko deploy karta hai ek beautiful UI ke saath.

## 📁 File Structure
```
salary_predictor/
├── main.py           ← FastAPI backend
├── index.html        ← Frontend UI (serve hota hai FastAPI se)
├── requirements.txt  ← Python dependencies
├── gbr_model.pkl     ← (Aapka trained model — yahan rakho)
└── README.md
```

## 🚀 Setup & Run

### Step 1 — Model file rakho
Apna `gbr_model.pkl` is folder mein copy karo:
```
salary_predictor/gbr_model.pkl
```

### Step 2 — Dependencies install karo
```bash
pip install -r requirements.txt
```

### Step 3 — Server start karo
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4 — Browser mein open karo
```
http://localhost:8000
```

---

## 🔌 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Frontend UI |
| POST | `/predict` | Salary prediction |
| GET | `/health` | Server health check |
| GET | `/options` | All valid dropdown options |

### POST `/predict` — Request Body
```json
{
  "jobType": "MANAGER",
  "degree": "MASTERS",
  "major": "COMPSCI",
  "industry": "FINANCE",
  "yearsExperience": 10,
  "milesFromMetropolis": 25
}
```

### POST `/predict` — Response
```json
{
  "predicted_salary": 142.5,
  "salary_range_low": 131.1,
  "salary_range_high": 153.9,
  "confidence_note": "Prediction based on Gradient Boosting Regressor trained on 1M job records."
}
```

---

## ⚙️ Feature Encoding (training ke same order mein)

| Feature | Values |
|---------|--------|
| jobType | JANITOR, JUNIOR, SENIOR, MANAGER, VICE_PRESIDENT, CFO, CTO, CEO |
| degree | NONE, HIGH_SCHOOL, BACHELORS, MASTERS, DOCTORAL |
| major | NONE, BIOLOGY, LITERATURE, CHEMISTRY, PHYSICS, COMPSCI, MATH, BUSINESS, ENGINEERING |
| industry | EDUCATION, AUTO, HEALTH, WEB, FINANCE, OIL, SERVICE |

> ⚠️ Agar aapne training mein alag LabelEncoder order use kiya tha, toh `main.py` mein `*_MAP` dictionaries update karo.

---

## 🌐 Production Deployment (optional)

Render / Railway pe deploy karne ke liye:
```bash
# Procfile content:
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```
