# 🛡️ UPI Fraud Detection — BharatBricks Project

An end-to-end, production-grade UPI transaction fraud detection system built on **Databricks**, served via **Databricks Model Serving**, and visualised through an interactive **Streamlit** frontend.

---

## 🏗️ Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│                                                                     │
│  ┌──────────────────────┐      ┌──────────────────────────────────┐ │
│  │  Kaggle Dataset       │      │  data.gov.in (Indian Govt)       │ │
│  │  (Transaction Logs)   │      │  State-level UPI/PoS infra data  │ │
│  │  df_log               │      │  df_gov                          │ │
│  └──────────┬───────────┘      └────────────────┬─────────────────┘ │
│             │                                    │                   │
│             └──────────────┬─────────────────────┘                  │
│                            ▼                                        │
│              LEFT JOIN on state column                              │
│              → upi_fraud_enriched_final (Delta Table)               │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                             │
│                                                                     │
│  balance_diff_orig      = newbalanceOrig  - oldbalanceOrg           │
│  balance_diff_dest      = newbalanceDest  - oldbalanceDest          │
│  orig_balance_zero_flag = 1 if oldbalanceOrg == 0 else 0            │
│                                                                     │
│  Feature cols (no leakage, no synthetic state features):            │
│  step, amount, oldbalanceOrg, newbalanceOrig,                       │
│  oldbalanceDest, newbalanceDest,                                    │
│  balance_diff_orig, balance_diff_dest, orig_balance_zero_flag       │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ML PIPELINE (PySpark)                         │
│                                                                     │
│  StringIndexer  →  VectorAssembler  →  StandardScaler              │
│       │                  │                    │                     │
│  Encodes "type"    Assembles all        Zero mean,                  │
│  (TRANSFER, etc.)  features → vector   unit std                     │
│                                                │                    │
│                             RandomForestClassifier                  │
│                             numTrees=100, maxDepth=10               │
│                             weightCol=weight (class imbalance)      │
│                                                │                    │
│                         Auto-computed downsampling (5x ratio)       │
│                         3-tier threshold selection (F1 → Recall     │
│                         → min FP)                                   │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL PERSISTENCE                              │
│                                                                     │
│  Raw Spark model  →  /dbfs/models/upi_fraud_detection_model         │
│                                                                     │
│  MLflow pyfunc wrapper  →  models.default.upi_fraud_detector        │
│  (Required for Databricks Model Serving — adds python_function      │
│   flavor on top of Spark model)                                     │
│                                                                     │
│  Threshold results  →  Delta: upi_fraud_threshold_results           │
│  State summary      →  Delta: upi_fraud_state_summary               │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  DATABRICKS MODEL SERVING                           │
│                                                                     │
│  Endpoint: upi_fraud_detector                                       │
│  Compute : CPU Small, min replicas = 1, scale-to-zero = OFF         │
│  Input   : dataframe_records (JSON)                                 │
│  Output  : prob_fraud (float), fraud_flag (0 or 1)                  │
│                                                                     │
│  URL: https://<workspace>.azuredatabricks.net/                      │
│       serving-endpoints/upi_fraud_detector/invocations              │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STREAMLIT FRONTEND                             │
│                  localhost:8501                                      │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ Single Txn Score│  │  Batch CSV Score  │  │ Analytics Dashboard│ │
│  │                 │  │                  │  │                   │  │
│  │ Enter txn fields│  │ Upload CSV file  │  │ Fraud vs Safe dist│  │
│  │ → Detect Fraud  │  │ → Score Batch    │  │ Prediction history│  │
│  │ → prob + label  │  │ → Download result│  │ Summary metrics   │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
│                                                                     │
│  Feature engineering runs CLIENT-SIDE before API call               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Request Flow (Single Transaction)

```
User fills form (Streamlit UI)
         │
         ▼
Feature engineering in app.py
  balance_diff_orig, balance_diff_dest, orig_balance_zero_flag
         │
         ▼
POST /invocations
  Authorization: Bearer <DATABRICKS_TOKEN>
  Body: { "dataframe_records": [ { ...features } ] }
         │
         ▼
Databricks Model Serving Endpoint
  → pyfunc wrapper loads PipelineModel
  → StringIndexer → VectorAssembler → StandardScaler → RF
  → Returns prob_fraud + fraud_flag
         │
         ▼
Streamlit displays result
  🔴 FRAUD  (prob_fraud > threshold)
  🟢 SAFE   (prob_fraud ≤ threshold)
```

---

## 📁 Project Structure

```
bharatbricks-/
│
├── UPI_fraud_frontend/               # Streamlit frontend
│   ├── app.py                        # Main app (single + batch + dashboard)
│   ├── requirements.txt              # Python dependencies
│   └── .streamlit/
│       ├── secrets.toml              # ← your token goes here (gitignored)
│       └── secrets.toml.example      # Template (safe to commit)
│
└── Databricks Notebooks/             # Hosted on Databricks
    ├── 01_data_ingestion.ipynb       # Load df_log + df_gov from Unity Catalog
    ├── 02_feature_engineering.ipynb  # JOIN, engineer features, save Delta table
    ├── 03_model_training.ipynb       # ML pipeline, CV, threshold tuning
    ├── 04_model_registration.ipynb   # pyfunc wrapper + MLflow registration
    └── 05_inference.ipynb            # score_transactions() function
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.9986 |
| PR-AUC | 0.8144 |
| Accuracy | 95.84% |
| F1 Score | 0.9773 |
| Precision | 0.9988 |
| Recall | 0.9588 |
| False Negatives | 4 |

**Optimal threshold** selected via 3-tier fallback: maximize F1 → maximize Recall → minimize FP.

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/Vavadiya-Rudra-Bhaveshbhai/bharatbricks-.git
cd bharatbricks-/UPI_fraud_frontend
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure secrets

Create `.streamlit/secrets.toml` (never commit this):

```toml
DATABRICKS_TOKEN = "dapixxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
DATABRICKS_URL   = "https://<workspace>.azuredatabricks.net/serving-endpoints/upi_fraud_detector/invocations"
```

Use `.streamlit/secrets.toml.example` as a template.

### 5. Run the app

```bash
streamlit run app.py
```

Opens at: **http://localhost:8501**

---

## 🔐 Security Notes

- `secrets.toml` is gitignored — never commit it
- Token stays server-side (Streamlit backend) — never exposed to browser
- Rotate your Databricks PAT if accidentally pushed to GitHub
- Databricks endpoint requires `model-serving` token scope

---

## 🧩 Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Streamlit | latest |
| Pandas | latest |
| Requests | latest |
| Plotly | latest |
| Databricks Runtime | 13.x+ |
| PySpark | 3.5.x |

---

## 🌐 Backend Dependency

This frontend requires a **live Databricks Model Serving endpoint**. Ensure:

- Endpoint `upi_fraud_detector` is deployed and **Ready**
- Compute is set to **CPU Small**, min replicas = **1**, scale-to-zero = **OFF**
- Model registered as `models.default.upi_fraud_detector` (pyfunc flavor)
- Your token has `model-serving` scope

Quick health check:

```bash
curl -X POST "https://<workspace>.azuredatabricks.net/serving-endpoints/upi_fraud_detector/invocations" \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"step":1,"amount":5000,"oldbalanceOrg":10000,"newbalanceOrig":5000,"oldbalanceDest":0,"newbalanceDest":5000,"type":"TRANSFER","balance_diff_orig":-5000,"balance_diff_dest":5000,"orig_balance_zero_flag":0}]}'
```

---

## 🤝 Contribution

- Follow modular structure — keep frontend and Databricks notebooks separate
- Do not commit `secrets.toml` or any file containing tokens
- Feature engineering must match exactly between training notebook and `app.py`
- All Delta table writes use `.option("overwriteSchema", "true")`

---

## 🏁 Status

✅ Data pipeline complete  
✅ ML model trained and registered  
✅ Databricks Model Serving endpoint live  
✅ pyfunc wrapper deployed (resolves Spark model serving limitation)  
✅ Streamlit frontend — production ready  
✅ Secure token handling  
✅ Batch + real-time inference  

---

*Part of the BharatBricks project — ML systems for Indian digital payments infrastructure.*
