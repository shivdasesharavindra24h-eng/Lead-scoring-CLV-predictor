import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="B2B AI Dashboard", layout="wide")

# ---- REMOVE STREAMLIT DEFAULT UI ----
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding: 0 !important; margin:0 !important; max-width:100%;}
</style>
""", unsafe_allow_html=True)

# ---- GLOBAL CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

body { background-color: black !important; }

.dashboard-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #00eaff;
    padding: 25px 0 10px 0;
}

.glass {
  background: rgba(255,255,255,0.09);
  border-radius: 22px;
  padding: 28px;
  border: 1px solid rgba(255,255,255,0.2);
  backdrop-filter: blur(14px);
  margin: 20px 40px;
}

.metric {
    background: rgba(0,229,255,0.18);
    border-radius: 14px;
    padding: 18px;
    text-align:center;
    font-size: 22px;
    font-weight:700;
    border:1px solid rgba(0,229,255,0.4);
    box-shadow:0 0 18px rgba(0,229,255,0.4);
}

.success{color:#00ff99;}
.warning{color:#ffdd4d;}
.danger{color:#ff4d6d;}

.stButton>button {
    width:100%;
    background: linear-gradient(135deg,#00eaff,#0078ff);
    border-radius:12px;
    font-size:20px;
    padding:14px;
    font-weight:700;
    color:black;
    border:none;
}
.stButton>button:hover {
    transform:scale(1.03);
    transition:.25s;
}
</style>
""", unsafe_allow_html=True)

# ----------- TRAIN MODELS -----------
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
    if reverse: return "success" if v < 40 else "warning" if v < 60 else "danger"
    return "success" if v >= 70 else "warning" if v >= 40 else "danger"

# ----------- UI -----------
st.markdown("<h1 class='dashboard-title'>🚀 B2B AI Marketing Intelligence Dashboard</h1>", unsafe_allow_html=True)

st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("📍 Enter Lead / Client Information")

c1,c2,c3 = st.columns(3)
values = [
    c1.number_input("Company Size (1-3)",1,3,2),
    c2.number_input("Industry (1-4)",1,4,1),
    c3.number_input("Revenue (M$)",0,2000,40),
    c1.number_input("Engagement Score",0,100,50),
    c2.number_input("Email Opens",0,1000,25),
    c3.number_input("Website Visits",0,20000,1800),
    c1.number_input("Meetings Booked",0,30,3),
    c2.number_input("Days Since Last Activity",0,365,10),
    c3.number_input("Competitor Interaction (0-10)",0,10,3),
]

run = st.button("Run Prediction ⚡")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- RESULT SECTION ----------
if run:
    Xnew = np.array([values])
    lead = lead_model.predict_proba(Xnew)[0,1]*100
    churn = churn_model.predict_proba(Xnew)[0,1]*100
    conv = conv_model.predict_proba(Xnew)[0,1]*100
    clv = clv_model.predict(Xnew)[0]

    st.markdown("<h3 style='text-align:center;font-weight:700;color:#00eaff;'>📊 KPI Performance</h3>", unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"<div class='metric'>Lead<br><span class='{color(lead)}'>{lead:.1f}%</span></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric'>Churn<br><span class='{color(churn,True)}'>{churn:.1f}%</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric'>Conversion<br><span class='{color(conv)}'>{conv:.1f}%</span></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric'>CLV<br><span class='success'>${clv:,.0f}</span></div>", unsafe_allow_html=True)

    # ----- Additional charts -----
    st.subheader("📈 Visual Performance Breakdown")

    fig = go.Figure(go.Bar(
        x=["Lead","Churn","Conversion","CLV/1000"],
        y=[lead, churn, conv, clv/1000],
        text=[f"{lead:.1f}%",f"{churn:.1f}%",f"{conv:.1f}%",f"${clv:,.0f}"],
        marker=dict(color=[lead,churn,conv,clv/1000], colorscale="Electric"),
        textposition="auto"
    ))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Funnel chart
    funnel = go.Figure(go.Funnel(
        y=["Awareness","Interest","Consideration","Purchase"],
        x=[100, lead, conv, conv/1.4],
        marker={"color":"#00eaff"}
    ))
    funnel.update_layout(template="plotly_dark", height=380)
    st.subheader("📉 Sales Funnel Simulation")
    st.plotly_chart(funnel, use_container_width=True)

    # Recommendations
    st.subheader("💡 AI Strategy Recommendations")
    recs=[]
    if lead>=60: recs.append("🚀 High priority: schedule demo")
    else: recs.append("📩 Move to education campaigns")
    if churn>=60: recs.append("⚠ Retention risk — CS outreach required")
    if conv>=70: recs.append("💰 Send pricing + case studies")
    if clv>=90000: recs.append("🌟 Elite ABM segment – personalise messaging")

    for r in recs:
        st.write(f"- {r}")
