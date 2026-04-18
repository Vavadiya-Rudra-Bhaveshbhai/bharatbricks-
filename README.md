# 🛡️ UPI Fraud Detection — Frontend (Streamlit)

This module provides an interactive **Streamlit-based frontend** for real-time UPI fraud detection.
It connects to a **Databricks Model Serving endpoint** and performs **client-side feature engineering** before inference.

---

## 🚀 Features

* 🔍 **Single Transaction Scoring**
* 📊 **Batch CSV Scoring**
* 🗺️ **Analytics Dashboard**
* ⚙️ **System Overview Panel**
* ⚡ Real-time inference via Databricks API

---

## 🧠 Architecture

```
User Input (UI)
        ↓
Feature Engineering (Frontend)
        ↓
Databricks Model Serving Endpoint
        ↓
Prediction (Fraud / Safe)
        ↓
UI Display
```

---

## 📁 Project Structure

```
UPI_fraud_frontend/
│
├── app.py                      # Main Streamlit app
├── requirements.txt           # Python dependencies
└── .streamlit/
    └── secrets.toml.example   # Template for API token
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Vavadiya-Rudra-Bhaveshbhai/bharatbricks-.git
cd bharatbricks-/UPI_fraud_frontend
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Secrets (IMPORTANT)

Create a new file:

```
.streamlit/secrets.toml
```

Copy contents from:

```
.streamlit/secrets.toml.example
```

Update with your Databricks token:

```toml
DATABRICKS_TOKEN = "your_actual_token_here"
```

⚠️ **Never commit this file to GitHub**

---

### 5️⃣ Run the App

```bash
streamlit run app.py
```

App will open at:

```
http://localhost:8501
```

---

## 📊 Usage

### 🔍 Fraud Detection

* Enter transaction details
* Click **Detect Fraud**
* View prediction + latency

---

### 📊 Batch Scoring

* Download sample CSV
* Upload your dataset
* Click **Score Batch**

Supports:

* Raw transaction CSV
* Engineered feature CSV

---

### 🗺️ Analytics

* View prediction history
* Fraud vs Safe distribution
* Summary metrics

---

## 🔐 Security Notes

* `secrets.toml` is ignored via `.gitignore`
* Use `.streamlit/secrets.toml.example` as template
* Rotate tokens if accidentally exposed

---

## 🧩 Requirements

* Python 3.10+
* Streamlit
* Pandas
* Requests
* Plotly

---

## 🌐 Backend Dependency

This frontend requires a **live Databricks Model Serving endpoint**.

Ensure:

* Endpoint is deployed
* Token has `model-serving` scope
* URL is correctly configured in `app.py`

---

## 🤝 Contribution

* Follow modular structure
* Do not commit secrets
* Keep frontend/backend separation clean

---

## 🏁 Status

✅ Production-ready frontend
✅ Secure token handling
✅ Batch + real-time inference

---

## 📌 Notes

This frontend is part of the larger **BharatBricks** project and integrates with backend ML systems hosted on Databricks.

---
