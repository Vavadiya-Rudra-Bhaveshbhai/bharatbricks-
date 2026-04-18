import hashlib
import time

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="UPI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .title-box {
        padding: 18px 20px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(16,185,129,0.14));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 18px;
    }

    .glass-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .safe-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(5,150,105,0.12));
        border: 1px solid rgba(16,185,129,0.35);
        padding: 18px;
        border-radius: 16px;
        color: #d1fae5;
        font-weight: 700;
        text-align: center;
    }

    .fraud-box {
        background: linear-gradient(135deg, rgba(239,68,68,0.20), rgba(185,28,28,0.14));
        border: 1px solid rgba(239,68,68,0.35);
        padding: 18px;
        border-radius: 16px;
        color: #fee2e2;
        font-weight: 700;
        text-align: center;
    }

    .info-box {
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59,130,246,0.25);
        padding: 16px;
        border-radius: 14px;
        color: #dbeafe;
    }

    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #22c55e 100%);
        color: white;
        border: none;
        border-radius: 12px;
        height: 3.1em;
        font-weight: 700;
        width: 100%;
    }

    .stButton>button:hover {
        opacity: 0.92;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div {
        background: #111827 !important;
        color: white !important;
        border-radius: 10px !important;
    }

    h1, h2, h3, h4 {
        color: #f8fafc;
    }

    p, label, span, div {
        color: #e5e7eb;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.92rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS
# =========================================================
ENDPOINT_URL = "https://dbc-db8911d3-57ab.cloud.databricks.com/serving-endpoints/fraud_upi/invocations"

TYPE_MAP = {
    "TRANSFER": 0,
    "CASH_OUT": 1,
    "PAYMENT": 2,
    "CASH_IN": 3,
    "DEBIT": 4,
}

STATE_LIST = ["UP", "MH", "DL", "KA", "TN", "GJ", "RJ", "WB", "MP", "AP"]

STATE_FEATURES = {
    "UP": (800, 15.0, 0.03),
    "MH": (2000, 25.0, 0.01),
    "DL": (1500, 20.0, 0.015),
    "KA": (1200, 18.0, 0.02),
    "TN": (1000, 16.0, 0.02),
    "GJ": (900, 14.0, 0.025),
    "RJ": (700, 12.0, 0.04),
    "WB": (850, 13.0, 0.03),
    "MP": (600, 10.0, 0.05),
    "AP": (950, 17.0, 0.02),
}

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "total_pos_devices",
    "total_gov_amount_lakh",
    "fraud_rate",
    "amount_to_gov_ratio",
    "pos_density_flag",
    "high_fraud_state_flag",
    "log_gov_amount",
    "log_pos_devices",
    "balance_diff_orig",
    "balance_diff_dest",
    "orig_balance_zero_flag",
    "type_index",
]

# =========================================================
# SESSION STATE INIT
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

# =========================================================
# HELPERS
# =========================================================
def get_token():
    try:
        return st.secrets["DATABRICKS_TOKEN"]
    except Exception:
        return None


def assign_state(name_orig: str) -> str:
    h = int(hashlib.md5(name_orig.encode("utf-8")).hexdigest(), 16)
    return STATE_LIST[h % len(STATE_LIST)]


def get_state_features(state: str):
    return STATE_FEATURES.get(state, (500, 10.0, 0.02))


def build_features_full(
    step,
    txn_type,
    name_orig,
    amount,
    oldbalance_org,
    newbalance_orig,
    oldbalance_dest,
    newbalance_dest,
):
    state = assign_state(name_orig)
    total_pos_devices, total_gov_amount_lakh, fraud_rate = get_state_features(state)

    type_index = TYPE_MAP[txn_type]

    amount_to_gov_ratio = amount / (total_gov_amount_lakh + 1.0)
    pos_density_flag = 1 if total_pos_devices < 1000 else 0
    log_gov_amount = float(np.log1p(total_gov_amount_lakh))
    log_pos_devices = float(np.log1p(total_pos_devices))
    balance_diff_orig = oldbalance_org - newbalance_orig
    balance_diff_dest = newbalance_dest - oldbalance_dest
    orig_balance_zero_flag = 1 if oldbalance_org == 0 else 0
    high_fraud_state_flag = 1 if fraud_rate > 0.05 else 0

    return [
        int(step),
        float(amount),
        float(oldbalance_org),
        float(newbalance_orig),
        float(oldbalance_dest),
        float(newbalance_dest),
        int(total_pos_devices),
        float(total_gov_amount_lakh),
        float(fraud_rate),
        float(amount_to_gov_ratio),
        int(pos_density_flag),
        int(high_fraud_state_flag),
        float(log_gov_amount),
        float(log_pos_devices),
        float(balance_diff_orig),
        float(balance_diff_dest),
        int(orig_balance_zero_flag),
        int(type_index),
    ], state


def call_databricks_endpoint(features):
    token = get_token()
    if not token:
        return None, "Missing DATABRICKS_TOKEN in .streamlit/secrets.toml"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "dataframe_split": {
            "columns": FEATURE_COLUMNS,
            "data": [features],
        }
    }

    try:
        start = time.time()
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=60)
        latency = time.time() - start
        st.session_state.last_latency = latency

        if response.status_code != 200:
            return None, f"{response.status_code}: {response.text}"

        data = response.json()
        preds = data.get("predictions", [])
        if not preds:
            return None, "No predictions returned by the endpoint."

        pred = preds[0]
        if isinstance(pred, list):
            pred = pred[0]

        try:
            pred = int(round(float(pred)))
        except Exception:
            pass

        return pred, None

    except requests.RequestException as e:
        return None, f"Request error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def add_history(row):
    st.session_state.history.insert(0, row)
    st.session_state.history = st.session_state.history[:30]


def render_result(pred, state, name_orig, amount):
    label = "FRAUD" if pred == 1 else "SAFE"
    if pred == 1:
        st.markdown(
            f"""
            <div class="fraud-box">
                ⚠️ FRAUD DETECTED<br>
                <div style="font-size: 0.95rem; font-weight: 500; margin-top: 6px;">
                    User: {name_orig} | State: {state} | Amount: ₹{amount:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="safe-box">
                ✅ SAFE TRANSACTION<br>
                <div style="font-size: 0.95rem; font-weight: 500; margin-top: 6px;">
                    User: {name_orig} | State: {state} | Amount: ₹{amount:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return label


def sample_batch_df():
    rows = []

    raw_data = [
        (200, "C12345", "TRANSFER", 50000, 100000, 50000, 0, 50000),
        (301, "C67890", "CASH_OUT", 250000, 1000, 0, 0, 250000),
        (412, "C24680", "PAYMENT", 12000, 30000, 18000, 10000, 22000),
        (550, "C11111", "TRANSFER", 800000, 900000, 100000, 200000, 1000000),
        (620, "C22222", "CASH_OUT", 5000, 10000, 5000, 5000, 10000),
    ]

    for r in raw_data:
        step, nameOrig, txn_type, amount, obo, nbo, obd, nbd = r

        features, state = build_features_full(
            step=step,
            txn_type=txn_type,
            name_orig=nameOrig,
            amount=amount,
            oldbalance_org=obo,
            newbalance_orig=nbo,
            oldbalance_dest=obd,
            newbalance_dest=nbd,
        )

        row_dict = {
            # RAW INPUT (user fields)
            "step": step,
            "nameOrig": nameOrig,
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": obo,
            "newbalanceOrig": nbo,
            "oldbalanceDest": obd,
            "newbalanceDest": nbd,
            "state": state,
        }

        # ENGINEERED FEATURES (model input)
        for col, val in zip(FEATURE_COLUMNS, features):
            row_dict[col] = val

        rows.append(row_dict)

    return pd.DataFrame(rows)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🛡️ UPI Fraud Detection")
page = st.sidebar.radio(
    "Navigation",
    ["🔍 Fraud Detection", "📊 Batch Scoring", "🗺️ Analytics", "⚙️ System Overview"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
<div class="small-muted">
<b>Backend</b>: Databricks Model Serving<br>
<b>Frontend</b>: Streamlit<br>
<b>Feature Flow</b>: Frontend-engineered inputs → Databricks endpoint
</div>
""",
    unsafe_allow_html=True,
)

if get_token():
    st.sidebar.success("Databricks token loaded")
else:
    st.sidebar.error("Missing Databricks token")

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
<div class="title-box">
    <h1 style="margin:0;">🛡️ UPI Fraud Detection System</h1>
    <p style="margin:8px 0 0 0; color:#cbd5e1;">
        Real-time transaction scoring using Databricks Model Serving
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# PAGE 1: FRAUD DETECTION
# =========================================================
if page == "🔍 Fraud Detection":
    colA, colB = st.columns([1.1, 1])

    with colA:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Transaction Input")

        nameOrig = st.text_input("Customer ID (nameOrig)", value="C12345")
        txn_type = st.selectbox("Transaction Type", list(TYPE_MAP.keys()))
        step = st.number_input("Step", min_value=0, max_value=10000, value=200, step=1)
        amount = st.number_input("Amount", min_value=0.0, value=50000.0, step=1000.0)

        st.markdown("#### Balances")
        c1, c2 = st.columns(2)
        with c1:
            oldbalanceOrg = st.number_input("oldbalanceOrg", min_value=0.0, value=100000.0, step=1000.0)
            oldbalanceDest = st.number_input("oldbalanceDest", min_value=0.0, value=0.0, step=1000.0)
        with c2:
            newbalanceOrig = st.number_input("newbalanceOrig", min_value=0.0, value=50000.0, step=1000.0)
            newbalanceDest = st.number_input("newbalanceDest", min_value=0.0, value=50000.0, step=1000.0)

        st.markdown("#### Notes")
        st.caption("The app computes the same engineered features before calling the Databricks endpoint.")
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Live Scoring")

        st.info("Click the button to send engineered features to the deployed Databricks endpoint.")

        if st.button("🚀 Detect Fraud"):
            features, state = build_features_full(
                step=step,
                txn_type=txn_type,
                name_orig=nameOrig,
                amount=amount,
                oldbalance_org=oldbalanceOrg,
                newbalance_orig=newbalanceOrig,
                oldbalance_dest=oldbalanceDest,
                newbalance_dest=newbalanceDest,
            )

            with st.spinner("Scoring transaction..."):
                pred, err = call_databricks_endpoint(features)

            if err:
                st.error(err)
            else:
                label = render_result(pred, state, nameOrig, amount)

                st.markdown("#### Engineered Features Sent")
                feat_preview = pd.DataFrame(
                    [features],
                    columns=FEATURE_COLUMNS,
                )
                st.dataframe(feat_preview, use_container_width=True, hide_index=True)

                st.metric("Prediction", label)
                if st.session_state.last_latency is not None:
                    st.metric("Endpoint Latency (sec)", f"{st.session_state.last_latency:.3f}")

                st.session_state.last_prediction = {
                    "nameOrig": nameOrig,
                    "type": txn_type,
                    "amount": amount,
                    "state": state,
                    "prediction": pred,
                    "label": label,
                    "latency": st.session_state.last_latency,
                }

                add_history(
                    {
                        "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                        "nameOrig": nameOrig,
                        "type": txn_type,
                        "amount": amount,
                        "state": state,
                        "prediction": label,
                    }
                )

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.last_prediction:
        st.markdown("### Latest Result")
        st.json(st.session_state.last_prediction)
# =========================================================
# PAGE 2: BATCH SCORING
# =========================================================
elif page == "📊 Batch Scoring":
    st.subheader("Batch Scoring")
    st.write("Upload a CSV or use the sample file below.")

    sample_df = sample_batch_df()
    st.download_button(
        "⬇️ Download Sample CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_upi_transactions.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    source_df = None
    if uploaded is not None:
        try:
            source_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            source_df = None
    else:
        source_df = sample_df.copy()

    if source_df is not None:
        st.markdown("#### Input Data")
        st.dataframe(source_df, use_container_width=True, hide_index=True)

        # Detect input type
        is_raw = all(
            col in source_df.columns
            for col in [
                "step",
                "nameOrig",
                "type",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
            ]
        )

        is_engineered = all(
            col in source_df.columns
            for col in [
                "total_pos_devices",
                "total_gov_amount_lakh",
                "fraud_rate",
                "amount_to_gov_ratio",
                "pos_density_flag",
                "high_fraud_state_flag",
                "log_gov_amount",
                "log_pos_devices",
                "balance_diff_orig",
                "balance_diff_dest",
                "orig_balance_zero_flag",
                "type_index",
            ]
        )

        if not (is_raw or is_engineered):
            st.error("CSV format not recognized. Please use the sample CSV.")
        else:
            if st.button("🚀 Score Batch"):
                results = []

                with st.spinner("Scoring batch..."):
                    for idx, row in source_df.iterrows():

                        # ---------- ENGINEERED INPUT (PRIORITY) ----------
                        if is_engineered:
                            features = [row[col] for col in FEATURE_COLUMNS]
                            state = row["state"] if "state" in source_df.columns else "N/A"

                        # ---------- RAW INPUT ----------
                        elif is_raw:
                            features, state = build_features_full(
                                step=row["step"],
                                txn_type=row["type"],
                                name_orig=row["nameOrig"],
                                amount=row["amount"],
                                oldbalance_org=row["oldbalanceOrg"],
                                newbalance_orig=row["newbalanceOrig"],
                                oldbalance_dest=row["oldbalanceDest"],
                                newbalance_dest=row["newbalanceDest"],
                            )

                        pred, err = call_databricks_endpoint(features)
                        name_orig = row.get("nameOrig", f"Row_{idx}")

                        if err:
                            results.append(
                                {
                                    "nameOrig": name_orig,
                                    "state": state,
                                    "prediction": "ERROR",
                                    "details": err,
                                }
                            )
                        else:
                            results.append(
                                {
                                    "nameOrig": name_orig,
                                    "state": state,
                                    "prediction": "FRAUD" if pred == 1 else "SAFE",
                                    "details": pred,
                                }
                            )

                res_df = pd.DataFrame(results)
                st.markdown("#### Batch Results")
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                st.session_state.history = (
                    [
                        {
                            "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                            "nameOrig": r.get("nameOrig"),
                            "type": "",
                            "amount": None,
                            "state": r.get("state"),
                            "prediction": r.get("prediction"),
                        }
                        for r in results[:10]
                    ]
                    + st.session_state.history
                )

# =========================================================
# PAGE 3: ANALYTICS
# =========================================================
elif page == "🗺️ Analytics":
    st.subheader("Analytics Dashboard")

    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        col1, col2, col3 = st.columns(3)
        col1.metric("Scored Transactions", len(hist))
        col2.metric("Fraud Flags", int((hist["prediction"] == "FRAUD").sum()))
        col3.metric("Safe Transactions", int((hist["prediction"] == "SAFE").sum()))

        st.markdown("#### Recent Scores")
        st.dataframe(hist, use_container_width=True, hide_index=True)

        summary = hist["prediction"].value_counts().reset_index()
        summary.columns = ["Prediction", "Count"]

        fig = px.pie(
            summary,
            names="Prediction",
            values="Count",
            hole=0.45,
            title="Prediction Mix",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No live history yet. Score a few transactions first.")

    st.markdown("#### Example Business Insights")
    st.markdown(
        """
- High-value cash-out patterns can be routed for manual verification.
- State-aware features help contextualize risk.
- The same frontend feature engineering keeps inference aligned with training.
"""
    )

# =========================================================
# PAGE 4: SYSTEM OVERVIEW
# =========================================================
elif page == "⚙️ System Overview":
    st.subheader("System Overview")

    st.markdown(
        """
<div class="glass-card">
<h3>Architecture</h3>
<p>
User input → frontend feature engineering → Databricks Model Serving endpoint → prediction returned to UI.
</p>
<p>
Backend model remains deployed on Databricks, while Streamlit handles interaction and presentation.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### Components")
    st.markdown(
        """
- **Databricks**: model training, MLflow tracking, serving endpoint
- **Streamlit**: interactive UI
- **Requests API**: calls the live endpoint
- **Feature Engineering**: performed in the frontend before the request
"""
    )

    st.markdown("#### Endpoint Status")
    token_present = bool(get_token())
    st.success("Token loaded" if token_present else "Token missing")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("UPI Fraud Detection System | Databricks Model Serving + Streamlit")