import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(page_title="B2B AI Dashboard", layout="wide")

# ---------- REMOVE DEFAULT STREAMLIT UI ----------
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    max-width: 100% !important;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------- CUSTOM UI CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

body {
    background: radial-gradient(circle at top left, #001a33, #000814, #00010a);
}

.dashboard-title {
    font-size: 42px;
    text-align: center;
    font-weight: 800;
    background: linear-gradient(90deg,#00eaff,#38bdf8,#0ea5e9);
    -webkit-background-clip: text;
    color: transparent;
    margin-top: 15px;
}

.glass {
  background: rgba(255,255,255,0.07);
  border-radius: 22px;
  padding: 28px;
  border: 1px solid rgba(255,255,255,0.15);
  backdrop-filter: blur(14px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.4);
  margin-bottom: 20px;
}

.metric {
  background: rgba(0,229,255,0.14);
  border: 1px solid rgba(0,229,255,0.38);
  padding: 18px;
  border-radius: 14px;
  text-align: center;
  font-size: 22px;
  font-weight: 700;
  box-shadow: 0 0 14px rgba(0,229,255,0.4);
}

.success {color:#00ff99;}
.warning {color:#ffdd4d;}
.danger {color:#ff4d6d;}

.stButton button {
    width: 100%;
    background: linear-gradient(135deg,#00eaff,#009dff,#006aff);
    font-weight: 800;
    border-radius: 14px;
    padding: 14px;
    font-size: 20px;
    border: none;
    color:black;
}
.stButton button:hover {
    transform: scale(1.03);
    transition: .25s;
}
</style>
""", unsafe_allow_html=True)

# ---------- MODEL TRAINING ----------
rng = np.random.RandomState(42)
X = rng.rand(1500, 9)
y1 = (rng.rand(1500)>0.5).astype(int)
y2 = (rng.rand(1500)>0.6).astype(int)
y3 = (rng.rand(1500)>0.55).astype(int)
y4 = rng.rand(1500)*120000 + 25000

lead_model = Pipeline([("s",StandardScaler()),("rf",RandomForestClassifier())]).fit(X,y1)
churn_model = Pipeline([("s",StandardScaler()),("rf",RandomForestClassifier())]).fit(X,y2)
conv_model  = Pipeline([("s",StandardScaler()),("rf",RandomForestClassifier())]).fit(X,y3)
clv_model   = Pipeline([("s",StandardScaler()),("rf",RandomForestRegressor())]).fit(X,y4)

def color(v, reverse=False):
    if reverse:
        return "success" if v < 40 else "warning" if v < 60 else "danger"
    return "success" if v >= 70 else "warning" if v >= 40 else "danger"

# ---------- DASHBOARD UI ----------
st.markdown("<h1 class='dashboard-title'>🚀 B2B AI Marketing Intelligence Dashboard</h1>",
            unsafe_allow_html=True)

st.write("")  # spacing
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("📌 Enter Lead / Client Data")

col1, col2, col3 = st.columns(3)
values = [
    col1.number_input("Company Size (1 = Small, 3 = Enterprise)",1,3,2),
    col2.number_input("Industry (1-4)",1,4,1),
    col3.number_input("Revenue (M$)",0,2000,50),
    col1.number_input("Engagement Score",0,100,60),
    col2.number_input("Email Opens",0,1000,40),
    col3.number_input("Website Visits",0,20000,2000),
    col1.number_input("Meetings Booked",0,30,2),
    col2.number_input("Days Since Last Activity",0,365,8),
    col3.number_input("Competitor Interaction (0-10)",0,10,3),
]
predict_btn = st.button("Predict & Show Dashboard 📊")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- PREDICTION RESULT ----------
if predict_btn:
    Xnew = np.array([values])
    lead = lead_model.predict_proba(Xnew)[0,1]*100
    churn = churn_model.predict_proba(Xnew)[0,1]*100
    conv = conv_model.predict_proba(Xnew)[0,1]*100
    clv = clv_model.predict(Xnew)[0]

    st.markdown("<h3 style='font-weight:700;text-align:center;margin-top:10px;'>📍 KPI Metrics</h3>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='metric'>Lead Score<br><span class='{color(lead)}'>{lead:.1f}%</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'>Churn Risk<br><span class='{color(churn,True)}'>{churn:.1f}%</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'>Conversion<br><span class='{color(conv)}'>{conv:.1f}%</span></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric'>CLV<br><span class='success'>${clv:,.0f}</span></div>", unsafe_allow_html=True)

    # Chart
    fig = go.Figure(go.Bar(
        x=["Lead","Churn","Conversion","CLV/1000"],
        y=[lead,churn,conv,clv/1000],
        text=[f"{lead:.1f}%",f"{churn:.1f}%",f"{conv:.1f}%",f"${clv:,.0f}"],
        textposition="auto",
        marker=dict(color=[lead,churn,conv,clv/1000], colorscale="Turbo")
    ))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 AI-Generated Strategy Actions")
    recs=[]
    if lead>=60: recs.append("🚀 Assign AE & schedule demo immediately")
    else: recs.append("📩 Move to nurture email flow")
    if churn>=60: recs.append("⚠ Deploy retention plan & feedback call")
    if conv>=70: recs.append("💰 Send pricing & ROI case study")
    if clv>=90000: recs.append("🌟 Add to ABM elite account tier")

    for r in recs: st.write(f"- {r}")
