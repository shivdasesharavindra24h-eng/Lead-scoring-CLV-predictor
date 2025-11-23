import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------- STREAMLIT PAGE CONFIG ----------------------
st.set_page_config(page_title="B2B AI Dashboard", layout="wide")

# ---------------------- CSS STYLING ----------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

body {
    background: linear-gradient(135deg, #0f051c, #020617, #001829);
    color: white;
}

h1 {
    font-size: 46px;
    text-align: center;
    font-weight: 700;
    color: #00eaff;
    margin-bottom: 15px;
}

.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
    backdrop-filter: blur(12px);
}

.metric-box {
    background: rgba(0, 229, 255, 0.1);
    border: 1px solid rgba(0, 229, 255, 0.3);
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    box-shadow: 0 0 10px rgba(0,229,255,0.5);
}

.success {color: #00ff99;}
.warning {color: #ffdd4d;}
.danger {color: #ff4d6d;}

button, .stButton>button {
    background: linear-gradient(135deg, #00eaff, #009dff, #006aff);
    color: black;
    font-size: 20px;
    font-weight: 700;
    padding: 14px;
    border-radius: 12px;
}
button:hover {
    transform: scale(1.03);
    transition: .3s;
}
</style>
""", unsafe_allow_html=True)


# ---------------------- ML MODEL TRAINING ----------------------
rng = np.random.RandomState(42)
X = rng.rand(1500,9)
y1 = (rng.rand(1500) > 0.5).astype(int)
y2 = (rng.rand(1500) > 0.6).astype(int)
y3 = (rng.rand(1500) > 0.55).astype(int)
y4 = rng.rand(1500)*120000 + 25000

lead_model = Pipeline([("s",StandardScaler()),("rf",RandomForestClassifier())]).fit(X, y1)
churn_model = Pipeline([("s",StandardScaler()),("rf",RandomForestClassifier())]).fit(X, y2)
conv_model  = Pipeline([("s",StandardScaler()),("rf",RandomForestClassifier())]).fit(X, y3)
clv_model   = Pipeline([("s",StandardScaler()),("rf",RandomForestRegressor())]).fit(X, y4)

def color(v, reverse=False):
    if reverse:
        return "success" if v < 40 else "warning" if v < 60 else "danger"
    return "success" if v >= 70 else "warning" if v >= 40 else "danger"

# ---------------------- TITLE ----------------------
st.markdown("<h1>🚀 B2B AI Marketing Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#9ca3af;font-size:18px;'>Predict lead strength, churn, conversion probability, CLV and generate strategy insights.</p>", unsafe_allow_html=True)

# ---------------------- INPUT UI ----------------------
with st.container():
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🔍 Enter Lead Details")

    col1, col2, col3 = st.columns(3)
    values = [
        col1.number_input("Company Size (1-3)", 1, 3, 2),
        col2.number_input("Industry (1-4)", 1, 4, 1),
        col3.number_input("Revenue (M$)", 0, 1000, 40),
        col1.number_input("Engagement Score", 0, 100, 55),
        col2.number_input("Email Opens (30d)", 0, 1000, 26),
        col3.number_input("Website Visits (30d)", 0, 20000, 900),
        col1.number_input("Meetings Booked", 0, 30, 4),
        col2.number_input("Days Since Last Activity", 0, 365, 12),
        col3.number_input("Competitor Interaction (0-10)", 0, 10, 3),
    ]

    run_button = st.button("Predict & Visualize 📊")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PREDICTION SECTION ----------------------
if run_button:
    Xnew = np.array([values])
    lead = lead_model.predict_proba(Xnew)[0,1]*100
    churn = churn_model.predict_proba(Xnew)[0,1]*100
    conv = conv_model.predict_proba(Xnew)[0,1]*100
    clv = clv_model.predict(Xnew)[0]

    st.subheader("📍 Prediction Results")
    colA, colB, colC, colD = st.columns(4)
    colA.markdown(f"<div class='metric-box'>Lead Score<br><span class='{color(lead)}'>{lead:.1f}%</span></div>", unsafe_allow_html=True)
    colB.markdown(f"<div class='metric-box'>Churn Risk<br><span class='{color(churn, True)}'>{churn:.1f}%</span></div>", unsafe_allow_html=True)
    colC.markdown(f"<div class='metric-box'>Conversion<br><span class='{color(conv)}'>{conv:.1f}%</span></div>", unsafe_allow_html=True)
    colD.markdown(f"<div class='metric-box'>CLV<br><span class='success'>${clv:,.0f}</span></div>", unsafe_allow_html=True)

    # ---- Chart ----
    st.subheader("📊 KPI Visualization")
    fig = go.Figure(go.Bar(
        x=["Lead Score", "Churn", "Conversion", "CLV/1000"],
        y=[lead, churn, conv, clv/1000],
        text=[f"{lead:.1f}%", f"{churn:.1f}%", f"{conv:.1f}%", f"${clv:,.0f}"],
        textposition="auto",
        marker=dict(colorscale="Turbo", color=[lead, churn, conv, clv/1000])
    ))
    fig.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Recommendations ----
    st.subheader("💡 AI-Driven Strategy Recommendations")
    recs = []
    if lead >= 60: recs.append("🚀 Assign AE & schedule demo immediately.")
    else: recs.append("📩 Move to nurture drip campaign.")
    if churn >= 60: recs.append("🛠 Trigger retention + success review.")
    if conv >= 70: recs.append("💰 Send pricing + ROI calculator + case study.")
    if clv >= 90000: recs.append("🌟 Target for ABM elite personalized campaigns.")

    for r in recs:
        st.markdown(f"✔ {r}")
