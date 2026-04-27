"""
╔══════════════════════════════════════════════════════════════════╗
║   NEXUS CREDIT INTELLIGENCE — Advanced Risk Assessment Platform  ║
║   Powered by Random Forest Ensemble | SMOTE Balanced Training    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NEXUS Credit Intelligence",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════
# PREMIUM DARK THEME CSS
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
.main .block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1400px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2d40;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

#MainMenu, footer, header { visibility: hidden; }

.nexus-topbar {
    background: linear-gradient(90deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 14px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin: -1.5rem -2.5rem 2rem;
}
.nexus-logo {
    font-size: 22px; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nexus-badge {
    background: #1e3a5f; color: #38bdf8 !important;
    padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 1px;
    border: 1px solid #2563eb44;
}
.nexus-status {
    margin-left: auto; display: flex; align-items: center;
    gap: 6px; font-size: 12px; color: #64748b;
}
.status-dot {
    width: 8px; height: 8px; background: #22c55e;
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

.glass-card {
    background: linear-gradient(135deg, #0f1923 0%, #0d1520 100%);
    border: 1px solid #1e3a5f; border-radius: 16px; padding: 24px;
    margin-bottom: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.section-label {
    display: flex; align-items: center; gap: 10px;
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    color: #38bdf8; text-transform: uppercase;
    margin-bottom: 16px; padding-bottom: 10px;
    border-bottom: 1px solid #1e3a5f;
}
.kpi-card {
    background: #0f1923; border: 1px solid #1e3a5f;
    border-radius: 12px; padding: 18px 20px;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.kpi-blue::before  { background: linear-gradient(90deg, #38bdf8, #2563eb); }
.kpi-green::before { background: linear-gradient(90deg, #22c55e, #16a34a); }
.kpi-amber::before { background: linear-gradient(90deg, #f59e0b, #d97706); }
.kpi-purple::before{ background: linear-gradient(90deg, #a78bfa, #7c3aed); }
.kpi-red::before   { background: linear-gradient(90deg, #f87171, #dc2626); }

.kpi-label { font-size: 11px; color: #64748b; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }
.kpi-value { font-size: 28px; font-weight: 700; color: #f1f5f9; margin: 4px 0; font-family: 'JetBrains Mono', monospace; }
.kpi-sub   { font-size: 12px; color: #64748b; }

.stNumberInput input, .stSelectbox select {
    background: #0f1923 !important; border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
label { color: #94a3b8 !important; font-size: 12px !important; font-weight: 500 !important; }

.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 14px 40px !important;
    font-size: 15px !important; font-weight: 700 !important;
    letter-spacing: 0.5px !important; width: 100% !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(37,99,235,0.6) !important;
}

.verdict-default {
    background: linear-gradient(135deg, #1a0a0a, #1f0f0f);
    border: 1px solid #7f1d1d; border-left: 4px solid #ef4444;
    border-radius: 14px; padding: 28px 32px;
}
.verdict-safe {
    background: linear-gradient(135deg, #0a1a0f, #0f1f14);
    border: 1px solid #14532d; border-left: 4px solid #22c55e;
    border-radius: 14px; padding: 28px 32px;
}
.verdict-score { font-size: 52px; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1; margin: 8px 0; }
.risk-factor-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid #1e2d40; font-size: 13px;
}
.rf-label { color: #94a3b8; }
.rf-val   { color: #f1f5f9; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.rf-badge-high { background: #7f1d1d33; color: #f87171; border: 1px solid #7f1d1d; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
.rf-badge-low  { background: #14532d33; color: #4ade80; border: 1px solid #14532d; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
.rf-badge-med  { background: #78350f33; color: #fbbf24; border: 1px solid #78350f; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }

.stTabs [data-baseweb="tab-list"] { background: #0f1923; border-bottom: 1px solid #1e3a5f; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b !important; border-radius: 8px 8px 0 0; font-size: 13px; }
.stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #38bdf8 !important; }

hr { border-color: #1e2d40 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    base   = os.path.dirname(os.path.abspath(__file__))
    model  = joblib.load(os.path.join(base, "model.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    with open(os.path.join(base, "meta.json")) as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()
feature_names = meta["feature_names"]
all_results   = meta["all_results"]
best_metrics  = meta["metrics"]


# ════════════════════════════════════════════════════════════════════
# TOP BAR
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nexus-topbar">
    <span style="font-size:20px;">🔷</span>
    <span class="nexus-logo">NEXUS</span>
    <span class="nexus-badge">CREDIT INTELLIGENCE v2.0</span>
    <span class="nexus-status">
        <span class="status-dot"></span>
        Model Active &nbsp;|&nbsp; RF Ensemble &nbsp;|&nbsp; SMOTE Balanced
    </span>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size:42px; margin-bottom:8px;">🔷</div>
        <div style="font-size:18px; font-weight:800; color:#f1f5f9; letter-spacing:-0.5px;">NEXUS</div>
        <div style="font-size:11px; color:#38bdf8; letter-spacing:2px; font-weight:600;">CREDIT INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    page = st.radio("NAVIGATION", [
        "🎯  Risk Assessment",
        "📈  Analytics Hub",
        "🧬  Model Intelligence",
        "📋  Data Reference"
    ], label_visibility="visible")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;letter-spacing:2px;color:#38bdf8;font-weight:700;margin-bottom:12px;">MODEL REGISTRY</div>', unsafe_allow_html=True)

    stat_rows = [
        ("Algorithm",   "Random Forest"),
        ("Estimators",  "50 Trees"),
        ("Accuracy",    f"{best_metrics['accuracy']*100:.1f}%"),
        ("ROC-AUC",     f"{best_metrics['roc_auc']:.4f}"),
        ("F1 Score",    f"{best_metrics['f1']:.4f}"),
        ("Imbalance",   "SMOTE ✓"),
        ("Scaler",      "StandardScaler"),
        ("Features",    "23 inputs"),
    ]
    for label, val in stat_rows:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;
                    border-bottom:1px solid #1e2d40;font-size:12px;">
            <span style="color:#64748b;">{label}</span>
            <span style="color:#f1f5f9;font-weight:600;font-family:'JetBrains Mono',monospace;">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:10px;padding:14px;">
        <div style="font-size:10px;letter-spacing:2px;color:#38bdf8;font-weight:700;margin-bottom:10px;">RISK THRESHOLDS</div>
        <div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#22c55e;"></div>
            <span style="color:#94a3b8;">Low Risk &lt; 30%</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#f59e0b;"></div>
            <span style="color:#94a3b8;">Medium 30–60%</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#ef4444;"></div>
            <span style="color:#94a3b8;">High Risk &gt; 60%</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — RISK ASSESSMENT
# ════════════════════════════════════════════════════════════════════
if "Risk Assessment" in page:

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-size:24px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;">Risk Assessment Console</div>
        <div style="font-size:14px;color:#64748b;margin-top:4px;">Enter client financial profile for real-time default probability scoring</div>
    </div>""", unsafe_allow_html=True)

    with st.form("assessment_form"):

        st.markdown('<div class="section-label">⬡ CLIENT PROFILE</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns([2,1,1,1.5,1.5])
        with c1: LIMIT_BAL = st.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=150000, step=10000)
        with c2: SEX = st.selectbox("Gender", [1,2], format_func=lambda x: "Male" if x==1 else "Female")
        with c3: AGE = st.number_input("Age", min_value=18, max_value=80, value=32)
        with c4: EDUCATION = st.selectbox("Education", [1,2,3,4], format_func=lambda x:{1:"Graduate School",2:"University",3:"High School",4:"Others"}[x])
        with c5: MARRIAGE = st.selectbox("Marital Status", [1,2,3], format_func=lambda x:{1:"Married",2:"Single",3:"Others"}[x])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">⬡ REPAYMENT STATUS HISTORY <span style="color:#64748b;font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;">(-1=Duly | 0=Revolving | 1–9=Months delayed)</span></div>', unsafe_allow_html=True)

        pay_months = ["Sep 2005 (PAY_0)","Aug 2005 (PAY_2)","Jul 2005 (PAY_3)","Jun 2005 (PAY_4)","May 2005 (PAY_5)","Apr 2005 (PAY_6)"]
        pcols = st.columns(6)
        pay_vals = []
        for col, label in zip(pcols, pay_months):
            with col:
                v = st.number_input(label.split(" (")[0], min_value=-2, max_value=9, value=-1, step=1, key=f"p_{label}")
                pay_vals.append(v)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">⬡ BILL STATEMENT AMOUNTS (NT$)</div>', unsafe_allow_html=True)
        bill_months = ["Sep 2005","Aug 2005","Jul 2005","Jun 2005","May 2005","Apr 2005"]
        bcols = st.columns(6)
        bill_vals = []
        for col, label in zip(bcols, bill_months):
            with col:
                v = st.number_input(label, min_value=-500000, max_value=2000000, value=45000, step=1000, key=f"b_{label}")
                bill_vals.append(v)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">⬡ PREVIOUS PAYMENT AMOUNTS (NT$)</div>', unsafe_allow_html=True)
        pa_cols = st.columns(6)
        pamt_vals = []
        for col, label in zip(pa_cols, bill_months):
            with col:
                v = st.number_input(label, min_value=0, max_value=2000000, value=2000, step=500, key=f"pa_{label}")
                pamt_vals.append(v)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡   RUN RISK ASSESSMENT")

    if submitted:
        input_df = pd.DataFrame([[
            LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
            *pay_vals, *bill_vals, *pamt_vals
        ]], columns=feature_names)

        scaled     = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        prob       = model.predict_proba(scaled)[0]
        risk_pct   = prob[1] * 100
        safe_pct   = prob[0] * 100

        if risk_pct < 30:   tier, tier_color = "LOW",    "#22c55e"
        elif risk_pct < 60: tier, tier_color = "MEDIUM", "#f59e0b"
        else:               tier, tier_color = "HIGH",   "#ef4444"

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">⬡ ASSESSMENT RESULTS</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns([1.8, 1.6, 1.6])

        with r1:
            verdict_class = "verdict-default" if prediction == 1 else "verdict-safe"
            verdict_icon  = "⚠" if prediction == 1 else "✓"
            verdict_label = "DEFAULT RISK DETECTED" if prediction == 1 else "CREDITWORTHY CLIENT"
            verdict_desc  = ("High likelihood of payment default. Recommend credit review, limit reduction, or escalation to collections team."
                             if prediction == 1 else
                             "Client demonstrates stable repayment behaviour. Standard credit terms applicable.")
            st.markdown(f"""
            <div class="{verdict_class}">
                <div style="font-size:12px;letter-spacing:2px;font-weight:700;color:{tier_color};">{verdict_icon}  {verdict_label}</div>
                <div class="verdict-score" style="color:{tier_color};">{risk_pct:.1f}%</div>
                <div style="font-size:12px;color:#64748b;margin:4px 0 10px;">Default Probability Score</div>
                <div style="background:{tier_color}22;border:1px solid {tier_color}44;border-radius:6px;
                            padding:5px 12px;display:inline-block;">
                    <span style="color:{tier_color};font-size:12px;font-weight:700;letter-spacing:1px;">{tier} RISK TIER</span>
                </div>
                <div style="font-size:13px;color:#94a3b8;margin-top:12px;">{verdict_desc}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                number={"suffix": "%", "font": {"size": 36, "color": tier_color, "family": "JetBrains Mono"}},
                delta={"reference": 22.1, "relative": False,
                       "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
                title={"text": "DEFAULT PROBABILITY", "font": {"size": 11, "color": "#64748b"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#1e3a5f", "tickfont": {"color": "#64748b", "size": 10}},
                    "bar":  {"color": tier_color, "thickness": 0.25},
                    "bgcolor": "#0f1923", "bordercolor": "#1e3a5f",
                    "steps": [
                        {"range": [0,30],   "color": "rgba(20,83,45,0.13)"},
                        {"range": [30,60],  "color": "rgba(120,53,15,0.13)"},
                        {"range": [60,100], "color": "rgba(127,29,29,0.13)"},
                    ],
                    "threshold": {"line": {"color": "#64748b", "width": 2}, "thickness": 0.8, "value": 22.1}
                }
            ))
            gauge_fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                                    font={"color": "#e2e8f0"}, height=260,
                                    margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with r3:
            bar_fig = go.Figure(go.Bar(
                x=["Safe (No Default)", "Default Risk"],
                y=[safe_pct, risk_pct],
                marker=dict(color=["#22c55e", "#ef4444"], line=dict(width=0)),
                text=[f"{safe_pct:.1f}%", f"{risk_pct:.1f}%"],
                textposition="outside",
                textfont=dict(size=16, family="JetBrains Mono", color="#f1f5f9"),
                width=0.5,
            ))
            bar_fig.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1923",
                font=dict(color="#94a3b8", family="Inter"), height=260,
                margin=dict(t=30, b=10, l=10, r=10),
                yaxis=dict(range=[0,115], gridcolor="#1e2d40"),
                showlegend=False,
                title=dict(text="PROBABILITY SPLIT", font=dict(size=11, color="#64748b")),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # Risk factor breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">⬡ RISK FACTOR BREAKDOWN</div>', unsafe_allow_html=True)

        total_delay  = sum(pay_vals)
        util_ratio   = bill_vals[0] / (LIMIT_BAL + 1) * 100
        pay_coverage = (pamt_vals[0] / (bill_vals[0] + 1) * 100) if bill_vals[0] > 0 else 100

        def badge(val, low_thresh, high_thresh, invert=False):
            if invert:
                if val >= high_thresh: return '<span class="rf-badge-low">LOW</span>'
                elif val >= low_thresh: return '<span class="rf-badge-med">MEDIUM</span>'
                else: return '<span class="rf-badge-high">HIGH</span>'
            else:
                if val <= low_thresh: return '<span class="rf-badge-low">LOW</span>'
                elif val <= high_thresh: return '<span class="rf-badge-med">MEDIUM</span>'
                else: return '<span class="rf-badge-high">HIGH</span>'

        factors = [
            ("Credit Limit",             f"NT$ {LIMIT_BAL:,.0f}",        badge(LIMIT_BAL, 50000, 200000, invert=True)),
            ("Credit Utilisation",       f"{util_ratio:.1f}%",            badge(util_ratio, 30, 70)),
            ("Total Repayment Delay",    f"{total_delay} months",          badge(total_delay, 0, 3)),
            ("Latest Pay Status (PAY_0)",f"Code {pay_vals[0]}",            badge(pay_vals[0], 0, 2)),
            ("Payment Coverage",         f"{pay_coverage:.1f}% of bill",   badge(pay_coverage, 30, 70, invert=True)),
            ("Latest Bill Statement",    f"NT$ {bill_vals[0]:,.0f}",       badge(bill_vals[0], 50000, 150000)),
            ("Latest Payment Made",      f"NT$ {pamt_vals[0]:,.0f}",       badge(pamt_vals[0], 2000, 10000, invert=True)),
            ("Overall Default Score",    f"{risk_pct:.1f}%",               badge(risk_pct, 30, 60)),
        ]

        rf_col1, rf_col2 = st.columns(2)
        for i, (label, val, b) in enumerate(factors):
            col = rf_col1 if i < 4 else rf_col2
            with col:
                st.markdown(f"""
                <div class="risk-factor-row">
                    <span class="rf-label">{label}</span>
                    <div style="display:flex;align-items:center;gap:12px;">
                        <span class="rf-val">{val}</span>{b}
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown("""
            <div style="background:#0f1117;border:1px solid #7f1d1d;border-radius:12px;padding:20px 24px;">
                <div style="color:#fca5a5;font-size:12px;font-weight:700;letter-spacing:1.5px;margin-bottom:10px;">⚠  ANALYST RECOMMENDATIONS</div>
                <div style="color:#94a3b8;font-size:13px;line-height:1.9;">
                    • <strong style="color:#fca5a5;">Immediate review</strong> of credit limit — consider reduction to minimise exposure<br>
                    • <strong style="color:#fca5a5;">Contact client</strong> for repayment plan negotiation before next billing cycle<br>
                    • <strong style="color:#fca5a5;">Flag account</strong> in early warning system for collections team<br>
                    • <strong style="color:#fca5a5;">Do not extend</strong> additional credit facilities until repayment stabilises<br>
                    • PAY_0 (latest payment status) is the strongest predictor — review immediately
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0a1a0f;border:1px solid #14532d;border-radius:12px;padding:20px 24px;">
                <div style="color:#86efac;font-size:12px;font-weight:700;letter-spacing:1.5px;margin-bottom:10px;">✓  ANALYST RECOMMENDATIONS</div>
                <div style="color:#94a3b8;font-size:13px;line-height:1.9;">
                    • Client meets standard creditworthiness criteria — <strong style="color:#86efac;">approve</strong> at current limit<br>
                    • Monitor utilisation ratio — optimal credit health below 30% usage<br>
                    • Eligible for <strong style="color:#86efac;">loyalty / limit upgrade</strong> review if 6-month trend is consistent<br>
                    • Schedule <strong style="color:#86efac;">next risk reassessment</strong> in 90 days per standard protocol
                </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS HUB
# ════════════════════════════════════════════════════════════════════
elif "Analytics Hub" in page:

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-size:24px;font-weight:800;color:#f1f5f9;">Analytics Hub</div>
        <div style="font-size:14px;color:#64748b;margin-top:4px;">Dataset insights and exploratory data analysis</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">⬡ DATASET OVERVIEW</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("TOTAL RECORDS", "30,000", "UCI Taiwan 2005", "kpi-blue"),
        ("DEFAULT RATE",  "22.1%",  "6,636 clients defaulted", "kpi-red"),
        ("SAFE RATE",     "77.9%",  "23,364 clients safe", "kpi-green"),
        ("FEATURES",      "23",     "Post-preprocessing", "kpi-purple"),
    ]
    for col, (label, val, sub, cls) in zip([k1,k2,k3,k4], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊  Class Distribution","💳  Credit Behaviour","📅  Payment Patterns","🔥  Correlations"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure(go.Pie(
                labels=["No Default","Default"], values=[23364,6636], hole=0.65,
                marker=dict(colors=["#22c55e","#ef4444"], line=dict(color="#0a0e1a", width=3)),
            ))
            fig.add_annotation(text="30,000<br><span style='font-size:11px'>Total Clients</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=18, color="#f1f5f9", family="JetBrains Mono"))
            fig.update_layout(paper_bgcolor="#0a0e1a", font=dict(color="#94a3b8"),
                              legend=dict(bgcolor="#0f1923", bordercolor="#1e3a5f"),
                              height=340, margin=dict(t=20,b=20,l=20,r=20),
                              title=dict(text="Default vs No-Default Split", font=dict(color="#64748b",size=12)))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="No Default", x=["Before SMOTE","After SMOTE"], y=[18691,18691], marker_color="#22c55e", width=0.35))
            fig2.add_trace(go.Bar(name="Default",    x=["Before SMOTE","After SMOTE"], y=[5309,18691],  marker_color="#ef4444", width=0.35))
            fig2.update_layout(barmode="group", paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1923",
                               font=dict(color="#94a3b8"), legend=dict(bgcolor="#0f1923",bordercolor="#1e3a5f"),
                               height=340, margin=dict(t=40,b=10,l=10,r=10), yaxis=dict(gridcolor="#1e2d40"),
                               title=dict(text="SMOTE Class Rebalancing Effect", font=dict(color="#64748b",size=12)))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        np.random.seed(42)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=np.random.lognormal(np.log(190000),0.75,5000), name="No Default", nbinsx=50, marker_color="#22c55e", opacity=0.7))
        fig.add_trace(go.Histogram(x=np.random.lognormal(np.log(120000),0.70,1500), name="Default",    nbinsx=50, marker_color="#ef4444", opacity=0.7))
        fig.update_layout(barmode="overlay", paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1923",
                          font=dict(color="#94a3b8"), legend=dict(bgcolor="#0f1923",bordercolor="#1e3a5f"),
                          height=320, margin=dict(t=20,b=10,l=10,r=10),
                          xaxis=dict(title="Credit Limit (NT$)",gridcolor="#1e2d40"),
                          yaxis=dict(title="Count",gridcolor="#1e2d40"))
        st.plotly_chart(fig, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            fig3 = go.Figure(go.Bar(x=["Graduate","University","High School","Others"], y=[20.4,22.8,25.1,23.9],
                                    marker=dict(color=["#38bdf8","#818cf8","#c084fc","#f59e0b"],line=dict(width=0)),
                                    text=["20.4%","22.8%","25.1%","23.9%"], textposition="outside",
                                    textfont=dict(color="#f1f5f9",size=13)))
            fig3.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1923", font=dict(color="#94a3b8"),
                               height=300, margin=dict(t=40,b=10,l=10,r=10),
                               yaxis=dict(range=[0,32],gridcolor="#1e2d40",title="Default Rate %"),
                               title=dict(text="Default Rate by Education", font=dict(color="#64748b",size=12)))
            st.plotly_chart(fig3, use_container_width=True)
        with col_d:
            fig4 = go.Figure(go.Bar(x=["Male","Female"], y=[24.2,20.8],
                                    marker=dict(color=["#38bdf8","#c084fc"],line=dict(width=0)),
                                    text=["24.2%","20.8%"], textposition="outside",
                                    textfont=dict(color="#f1f5f9",size=14), width=0.4))
            fig4.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1923", font=dict(color="#94a3b8"),
                               height=300, margin=dict(t=40,b=10,l=10,r=10),
                               yaxis=dict(range=[0,30],gridcolor="#1e2d40",title="Default Rate %"),
                               title=dict(text="Default Rate by Gender", font=dict(color="#64748b",size=12)))
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        months = ["Sep 2005","Aug 2005","Jul 2005","Jun 2005","May 2005","Apr 2005"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Avg Bill Statement (NT$)","Avg Payment Amount (NT$)"))
        for vals, name, color in [([39900,38900,37700,36500,35900,34600],"No Default","#22c55e"),
                                    ([52400,50800,49200,47800,46900,45300],"Default",   "#ef4444")]:
            fig.add_trace(go.Scatter(x=months, y=vals, name=name, mode="lines+markers",
                                     line=dict(color=color,width=2.5), marker=dict(size=8,color=color)), row=1, col=1)
        for vals, name, color in [([5200,5000,4800,4700,4600,4500],"No Default","#22c55e"),
                                    ([1800,1700,1600,1550,1500,1450],"Default",   "#ef4444")]:
            fig.add_trace(go.Scatter(x=months, y=vals, name=name, mode="lines+markers",
                                     line=dict(color=color,width=2.5), marker=dict(size=8,color=color),
                                     showlegend=False), row=1, col=2)
        fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1923", font=dict(color="#94a3b8"),
                          legend=dict(bgcolor="#0f1923",bordercolor="#1e3a5f"),
                          height=380, margin=dict(t=50,b=20,l=20,r=20))
        fig.update_xaxes(gridcolor="#1e2d40", tickangle=-30)
        fig.update_yaxes(gridcolor="#1e2d40")
        for ann in fig['layout']['annotations']:
            ann['font'] = dict(color="#64748b", size=12)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        features_corr = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
                          "LIMIT_BAL","BILL_AMT1","AGE","SEX","EDUCATION",
                          "MARRIAGE","PAY_AMT1","PAY_AMT2"]
        correlations  = [0.324,0.261,0.235,0.216,0.195,0.187,
                         -0.154,0.072,-0.013,0.040,-0.054,-0.024,-0.073,-0.066]
        fig = go.Figure(go.Bar(
            x=correlations, y=features_corr, orientation="h",
            marker=dict(color=["#ef4444" if c>0 else "#22c55e" for c in correlations], line=dict(width=0)),
            text=[f"{c:+.3f}" for c in correlations], textposition="outside",
            textfont=dict(color="#f1f5f9",size=12,family="JetBrains Mono"),
        ))
        fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1923", font=dict(color="#94a3b8"),
                          height=440, margin=dict(t=20,b=20,l=10,r=60),
                          xaxis=dict(gridcolor="#1e2d40",title="Pearson Correlation Coefficient",
                                     zeroline=True,zerolinecolor="#2563eb",zerolinewidth=1.5),
                          yaxis=dict(tickfont=dict(family="JetBrains Mono",size=12)))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INTELLIGENCE
# ════════════════════════════════════════════════════════════════════
elif "Model Intelligence" in page:

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-size:24px;font-weight:800;color:#f1f5f9;">Model Intelligence</div>
        <div style="font-size:14px;color:#64748b;margin-top:4px;">Performance benchmarking across all trained classifiers</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">⬡ BEST MODEL — RANDOM FOREST METRICS</div>', unsafe_allow_html=True)
    m = best_metrics
    kpi_data = [
        ("ACCURACY",  f"{m['accuracy']:.4f}", f"{m['accuracy']*100:.1f}% overall correct", "kpi-blue"),
        ("PRECISION", f"{m['precision']:.4f}", "Of predicted defaults correct",             "kpi-green"),
        ("RECALL",    f"{m['recall']:.4f}",   "Actual defaults captured",                  "kpi-amber"),
        ("F1 SCORE",  f"{m['f1']:.4f}",       "Precision-recall harmonic mean",            "kpi-purple"),
        ("ROC-AUC",   f"{m['roc_auc']:.4f}",  "Discriminative ability score",              "kpi-red"),
    ]
    cols = st.columns(5)
    for col, (label, val, sub, cls) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">⬡ MODEL LEADERBOARD</div>', unsafe_allow_html=True)
    tbl_col, rad_col = st.columns([1.2, 1])

    with tbl_col:
        df_res = pd.DataFrame(all_results).T.reset_index()
        df_res.columns = ["Model","Accuracy","Precision","Recall","F1","ROC-AUC"]
        df_res = df_res.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
        df_res["Rank"] = ["🥇","🥈","🥉","4️⃣"]
        st.dataframe(
            df_res[["Rank","Model","Accuracy","Precision","Recall","F1","ROC-AUC"]]
              .style.highlight_max(subset=["Accuracy","Precision","Recall","F1","ROC-AUC"], color="#14532d55")
              .format({c:"{:.4f}" for c in ["Accuracy","Precision","Recall","F1","ROC-AUC"]}),
            use_container_width=True, hide_index=True, height=200)

        metric_order = ["accuracy","precision","recall","f1","roc_auc"]
        fig_bar = go.Figure()
        for (name, vals), color in zip(all_results.items(), ["#38bdf8","#ef4444","#22c55e","#f59e0b"]):
            fig_bar.add_trace(go.Bar(name=name, x=[m.replace("_"," ").upper() for m in metric_order],
                                     y=[vals[m] for m in metric_order],
                                     marker=dict(color=color,opacity=0.85,line=dict(width=0))))
        fig_bar.update_layout(barmode="group", paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1923",
                              font=dict(color="#94a3b8"),
                              legend=dict(bgcolor="#0f1923",bordercolor="#1e3a5f",orientation="h",
                                          yanchor="bottom",y=1.02,xanchor="right",x=1),
                              height=330, margin=dict(t=40,b=10,l=10,r=10),
                              yaxis=dict(range=[0,1.05],gridcolor="#1e2d40"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with rad_col:
        radar_metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        def hex_to_rgba(hex_color, alpha=0.13):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"rgba({r},{g},{b},{alpha})"

        fig_radar = go.Figure()
        for (name, vals), color in zip(all_results.items(), ["#38bdf8","#ef4444","#22c55e","#f59e0b"]):
            r_vals = [vals["accuracy"],vals["precision"],vals["recall"],vals["f1"],vals["roc_auc"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals+[r_vals[0]], theta=radar_metrics+[radar_metrics[0]],
                name=name, fill="toself", line=dict(color=color,width=2),
                fillcolor=hex_to_rgba(color)))
        fig_radar.update_layout(
            polar=dict(bgcolor="#0f1923",
                       radialaxis=dict(visible=True,range=[0,1],tickfont=dict(size=9,color="#64748b"),
                                       gridcolor="#1e3a5f",linecolor="#1e3a5f"),
                       angularaxis=dict(gridcolor="#1e3a5f",linecolor="#1e3a5f",
                                        tickfont=dict(size=11,color="#94a3b8"))),
            paper_bgcolor="#0a0e1a", font=dict(color="#94a3b8"),
            legend=dict(bgcolor="#0f1923",bordercolor="#1e3a5f",font=dict(size=11)),
            height=570, margin=dict(t=30,b=30,l=30,r=30),
            title=dict(text="Multi-Metric Radar", font=dict(color="#64748b",size=12)))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="section-label">⬡ FEATURE IMPORTANCE (RANDOM FOREST — GINI IMPURITY)</div>', unsafe_allow_html=True)
    fi_features = ["PAY_0","PAY_2","LIMIT_BAL","PAY_3","PAY_4","BILL_AMT1","PAY_5","PAY_AMT1",
                   "PAY_6","BILL_AMT2","PAY_AMT2","BILL_AMT3","AGE","PAY_AMT3","BILL_AMT4"]
    fi_values   = [0.112,0.072,0.069,0.061,0.054,0.048,0.044,0.041,0.039,0.038,0.036,0.034,0.033,0.031,0.029]
    fi_fig = go.Figure(go.Bar(
        x=fi_values, y=fi_features, orientation="h",
        marker=dict(color=fi_values, colorscale=[[0,"#1e3a5f"],[0.5,"#2563eb"],[1,"#38bdf8"]],
                    showscale=True, colorbar=dict(thickness=12,len=0.8,tickfont=dict(color="#64748b"),outlinewidth=0)),
        text=[f"{v:.3f}" for v in fi_values], textposition="outside",
        textfont=dict(color="#f1f5f9",size=11,family="JetBrains Mono"),
    ))
    fi_fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1923", font=dict(color="#94a3b8"),
                         height=460, margin=dict(t=20,b=20,l=10,r=60),
                         xaxis=dict(gridcolor="#1e2d40",title="Relative Importance"),
                         yaxis=dict(tickfont=dict(family="JetBrains Mono",size=12),autorange="reversed"))
    st.plotly_chart(fi_fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA REFERENCE
# ════════════════════════════════════════════════════════════════════
elif "Data Reference" in page:

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-size:24px;font-weight:800;color:#f1f5f9;">Data Reference</div>
        <div style="font-size:14px;color:#64748b;margin-top:4px;">Dataset schema, preprocessing pipeline, and methodology</div>
    </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        st.markdown('<div class="section-label">⬡ DATASET SCHEMA</div>', unsafe_allow_html=True)
        schema = pd.DataFrame([
            ["LIMIT_BAL",   "Numeric", "Credit limit in NT dollars",                          "High"],
            ["SEX",         "Categ.",  "1=Male, 2=Female",                                    "Low"],
            ["EDUCATION",   "Categ.",  "1=Graduate, 2=University, 3=HS, 4=Others",            "Medium"],
            ["MARRIAGE",    "Categ.",  "1=Married, 2=Single, 3=Others",                       "Low"],
            ["AGE",         "Numeric", "Customer age in years",                                "Low"],
            ["PAY_0",       "Ordinal", "Repayment status Sep 2005 — strongest predictor",     "🔴 Critical"],
            ["PAY_2–PAY_6", "Ordinal", "Repayment status Aug–Apr 2005",                       "High"],
            ["BILL_AMT1–6", "Numeric", "Monthly bill statement Sep–Apr 2005 (NT$)",           "Medium"],
            ["PAY_AMT1–6",  "Numeric", "Monthly payment made Sep–Apr 2005 (NT$)",             "Medium"],
            ["default",     "Binary",  "TARGET — 1=Default, 0=No Default",                   "—"],
        ], columns=["Feature","Type","Description","Importance"])
        st.dataframe(schema, use_container_width=True, hide_index=True, height=370)

    with col_right:
        st.markdown('<div class="section-label">⬡ PIPELINE ARCHITECTURE</div>', unsafe_allow_html=True)
        steps = [
            ("01","#38bdf8","Data Ingestion",      "30,000 records, 25 columns from UCI Taiwan dataset"),
            ("02","#818cf8","Data Cleaning",        "Drop ID · Fix EDUCATION {0,5,6}→4 · Fix MARRIAGE {0}→3"),
            ("03","#c084fc","EDA",                  "Distribution analysis · Correlation heatmap · Profiling"),
            ("04","#f59e0b","Skewness Treatment",   "Log1p on BILL_AMT, PAY_AMT where skew > 1.0"),
            ("05","#22c55e","Feature Engineering",  "UTIL_RATIO · AVG_BILL · AVG_PAY_AMT · TOTAL_DELAY"),
            ("06","#ef4444","Train/Test Split",     "80/20 stratified · Random state 42"),
            ("07","#f87171","SMOTE",                "Oversampling on TRAIN only · 5,309 → 18,691 minority"),
            ("08","#38bdf8","StandardScaler",       "Fit on SMOTE train · Transform test separately"),
            ("09","#22c55e","Model Training",       "LR · DT · Random Forest · Gradient Boosting"),
            ("10","#f59e0b","Model Selection",      "Random Forest: best accuracy + AUC with compress=3"),
            ("11","#c084fc","Serialization",        "model.pkl (1.85MB) + scaler.pkl + meta.json via joblib"),
        ]
        for step_no, color, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:10px;">
                <div style="min-width:28px;height:28px;border-radius:6px;background:{color}22;
                            border:1px solid {color}55;display:flex;align-items:center;
                            justify-content:center;font-size:10px;font-weight:700;
                            color:{color};font-family:'JetBrains Mono',monospace;">{step_no}</div>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#f1f5f9;">{title}</div>
                    <div style="font-size:12px;color:#64748b;margin-top:1px;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">⬡ REPAYMENT STATUS CODE REFERENCE</div>', unsafe_allow_html=True)
    code_df = pd.DataFrame({
        "Code":    [-2,-1,0,1,2,3,4,5,6,7,8,9],
        "Meaning": ["No consumption","Pay duly","Revolving credit used",
                    "1 month delay","2 months delay","3 months delay",
                    "4 months delay","5 months delay","6 months delay",
                    "7 months delay","8 months delay","9+ months delay"],
        "Risk":    ["—","✅ Low","—","🟡 Medium","🟠 High","🟠 High",
                    "🔴 Critical","🔴 Critical","🔴 Critical",
                    "🔴 Critical","🔴 Critical","🔴 Critical"]
    })
    st.dataframe(code_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #1e2d40;padding:20px 0 0;display:flex;justify-content:space-between;align-items:center;">
    <div style="font-size:12px;color:#334155;">
        <span style="color:#38bdf8;font-weight:700;">NEXUS</span> Credit Intelligence &nbsp;|&nbsp;
        Random Forest · SMOTE · StandardScaler
    </div>
    <div style="font-size:11px;color:#334155;">
        UCI Taiwan Dataset · April–September 2005 · 30,000 records
    </div>
</div>
""", unsafe_allow_html=True)
