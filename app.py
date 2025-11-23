import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="B2B AI Marketing Dashboard", layout="wide")

# ------------------ Train demo models ------------------
rng = np.random.RandomState(42)
X = rng.rand(1500, 9)
y1 = (rng.rand(1500) > 0.5).astype(int)
y2 = (rng.rand(1500) > 0.6).astype(int)
y3 = (rng.rand(1500) > 0.55).astype(int)
y4 = rng.rand(1500) * 120000 + 25000

lead_model = Pipeline([("s",StandardScaler()),("rf",RandomForestClassifier())]).fit(X, y1)
churn_model = Pipeline([("s",StandardScaler()),("rf",RandomForestClassifier())]).fit(X, y2)
conv_model  = Pipeline([("s",StandardScaler()),("rf",RandomForestClassifier())]).fit(X, y3)
clv_model   = Pipeline([("s",StandardScaler()),("rf",RandomForestRegressor())]).fit(X, y4)

def color(v, reverse=False):
    if reverse:
        return "🟢 Low" if v < 40 else "🟡 Medium" if v < 60 else "🔴 High"
    return "🟢 High" if v >= 70 else "🟡 Medium" if v >= 40 else "🔴 Low"

st.title("🚀 B2B AI Marketing Intelligence Dashboard")

st.markdown("Enter customer/lead details below to predict lead score, churn risk, conversion probability and CLV.")

# ------------------ Input Fields ------------------
col1, col2, col3 = st.columns(3)

values = [
    col1.number_input("Company Size (1=Small, 2=Mid, 3=Enterprise)", min_value=1, max_value=3, value=2),
    col2.number_input("Industry (1-4)", min_value=1, max_value=4, value=1),
    col3.number_input("Revenue (M$)", 0, 1000, 20),
    col1.number_input("Engagement Score", 0, 100, 45),
    col2.number_input("Email Opens (30d)", 0, 500, 20),
    col3.number_input("Website Visits (30d)", 0, 10000, 200),
    col1.number_input("Meetings Booked", 0, 30, 1),
    col2.number_input("Days Since Last Activity", 0, 365, 14),
    col3.number_input("Competitor Interaction (0-10)", 0, 10, 5),
]

if st.button("Predict & Visualize 📊"):
    Xnew = np.array([values])

    lead = lead_model.predict_proba(Xnew)[0,1] * 100
    churn = churn_model.predict_proba(Xnew)[0,1] * 100
    conv = conv_model.predict_proba(Xnew)[0,1] * 100
    clv = clv_model.predict(Xnew)[0]

    st.subheader("📍 Prediction Results")
    st.write(f"**Lead Score:** {lead:.1f}% — {color(lead)}")
    st.write(f"**Churn Risk:** {churn:.1f}% — {color(churn, reverse=True)}")
    st.write(f"**Conversion Probability:** {conv:.1f}% — {color(conv)}")
    st.write(f"**Customer Lifetime Value (CLV):** ${clv:,.0f}")

    # Chart
    st.subheader("📊 KPI Visualization")
    fig = go.Figure(go.Bar(
        x=["Lead Score", "Churn Risk", "Conversion", "CLV/1000"],
        y=[lead, churn, conv, clv/1000],
        text=[f"{lead:.1f}%", f"{churn:.1f}%", f"{conv:.1f}%", f"${clv:,.0f}"],
        textposition="auto"
    ))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Strategy
    st.subheader("📌 AI Strategy Recommendations")
    recs = []
    if lead >= 60: recs.append("🚀 Assign AE & schedule product demo immediately.")
    else: recs.append("📩 Move lead to email nurture flow.")
    if churn >= 60: recs.append("🛠 Activate retention and success plan.")
    if conv >= 70: recs.append("💰 Share pricing & ROI calculator.")
    if clv >= 90000: recs.append("🌟 Move to ABM VIP segment.")

    for r in recs:
        st.write("- " + r)
