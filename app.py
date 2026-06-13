import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IMDB Sentiment GRU Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS / ANIMATIONS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Root overrides ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Animated gradient hero title ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #E24B4A 0%, #FF8C8B 50%, #E24B4A 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
    margin: 0;
    line-height: 1.2;
}
@keyframes shimmer { to { background-position: 200% center; } }

.hero-sub {
    font-size: 0.95rem;
    color: #888;
    margin-top: 4px;
    letter-spacing: 0.02em;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 13px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.status-online  { background: rgba(46,204,113,0.12); color: #27ae60; border: 1px solid rgba(46,204,113,0.3); }
.status-offline { background: rgba(226,75,74,0.12);  color: #E24B4A; border: 1px solid rgba(226,75,74,0.3); }
.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%; background: currentColor;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.75)} }

/* ── Metric cards ── */
.metric-card {
    background: #1a1e26;
    border-radius: 12px;
    padding: 18px 20px;
    border-left: 4px solid #E24B4A;
    transition: transform .2s ease, box-shadow .2s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(226,75,74,.18); }
.metric-label { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 6px; }
.metric-value { font-size: 1.7rem; font-weight: 700; color: #fff; }

/* ── Info panel (sidebar) ── */
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,.07);
    font-size: .83rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #888; }
.info-val { color: #fff; font-weight: 600; }

/* ── Sentiment result cards ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.result-card {
    border-radius: 14px;
    padding: 20px 24px;
    animation: fadeSlideUp .4s ease;
    border: 1px solid transparent;
}
.result-positive {
    background: rgba(46,204,113,.08);
    border-color: rgba(46,204,113,.25);
}
.result-negative {
    background: rgba(226,75,74,.08);
    border-color: rgba(226,75,74,.25);
}
.result-emoji { font-size: 2.4rem; }
.result-title { font-size: 1.3rem; font-weight: 700; margin: 4px 0 2px; }
.result-title.pos { color: #2ecc71; }
.result-title.neg { color: #E24B4A; }
.result-desc { font-size: .85rem; color: #aaa; }

/* ── Confidence bar ── */
.conf-track {
    height: 10px;
    background: rgba(255,255,255,.08);
    border-radius: 99px;
    overflow: hidden;
    margin: 8px 0 4px;
}
.conf-fill {
    height: 100%;
    border-radius: 99px;
    transition: width .8s cubic-bezier(.4,0,.2,1);
}
.conf-pct { font-size: 1.5rem; font-weight: 700; }

/* ── Token chips ── */
.token-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.token-chip {
    font-size: .75rem;
    padding: 3px 10px;
    border-radius: 99px;
    border: 1px solid rgba(255,255,255,.12);
    color: #aaa;
    background: rgba(255,255,255,.04);
}
.token-chip-pos { background: rgba(46,204,113,.12); border-color: rgba(46,204,113,.3); color: #2ecc71; font-weight: 600; }
.token-chip-neg { background: rgba(226,75,74,.12);  border-color: rgba(226,75,74,.3);  color: #E24B4A; font-weight: 600; }

/* ── Quick-fill buttons ── */
.quick-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }

/* ── Tabs custom look ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid rgba(255,255,255,.1) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    font-size: .88rem !important;
}
.stTabs [aria-selected="true"] {
    color: #E24B4A !important;
    border-bottom: 2px solid #E24B4A !important;
    background: rgba(226,75,74,.07) !important;
}

/* ── Plotly chart border ── */
.element-container iframe { border-radius: 12px; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] > div:first-child {
    border: 2px dashed rgba(226,75,74,.35) !important;
    border-radius: 12px !important;
    background: rgba(226,75,74,.03) !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"] > div:first-child:hover {
    border-color: #E24B4A !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] .stMarkdown p { color: #ccc; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #E24B4A, #c03a39) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: .03em !important;
    transition: opacity .2s, transform .1s !important;
    padding: 10px 24px !important;
}
.stButton > button[kind="primary"]:hover { opacity: .88 !important; transform: translateY(-1px) !important; }
.stButton > button[kind="primary"]:active { transform: scale(.98) !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,.08) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading GRU model artifacts…")
def load_model_artifacts():
    arts = {"model": None, "tokenizer": None, "config": None, "status": "Missing"}
    paths = ("best_gru_model.keras", "tokenizer.pkl", "model_config.pkl")
    if not all(os.path.exists(p) for p in paths):
        arts["status"] = "Files not found"
        return arts
    try:
        arts["model"] = load_model(paths[0])
        with open(paths[1], "rb") as f: arts["tokenizer"] = pickle.load(f)
        with open(paths[2], "rb") as f: arts["config"]    = pickle.load(f)
        arts["status"] = "Success"
    except Exception as e:
        arts["status"] = f"Error: {e}"
    return arts

artifacts = load_model_artifacts()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def predict_single(text: str):
    model     = artifacts["model"]
    tokenizer = artifacts["tokenizer"]
    config    = artifacts["config"]
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=config.get("max_length", 100), padding="post", truncating="post")
    raw    = float(model.predict(padded, verbose=0)[0][0])
    is_pos = raw >= 0.5
    conf   = raw if is_pos else 1 - raw
    return is_pos, conf, raw


def get_confusion_fig():
    cm = np.array([[4439, 500], [648, 4330]])
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Negative (0)", "Positive (1)"],
        y=["Negative (0)", "Positive (1)"],
        text_auto=True,
        color_continuous_scale=[[0, "#1a1e26"], [0.5, "#7B2020"], [1, "#E24B4A"]],
    )
    fig.update_layout(
        title="Confusion Matrix",
        title_font_size=15,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def get_bar_fig():
    cats   = ["Negative (0)", "Positive (1)"]
    prec   = [0.87, 0.90]
    recall = [0.90, 0.87]
    f1     = [0.89, 0.88]
    fig = go.Figure([
        go.Bar(name="Precision", x=cats, y=prec,   marker_color="#E24B4A"),
        go.Bar(name="Recall",    x=cats, y=recall,  marker_color="#FF8C8B"),
        go.Bar(name="F1-Score",  x=cats, y=f1,      marker_color="#7B2020"),
    ])
    fig.update_layout(
        barmode="group",
        title="Classification Metrics",
        title_font_size=15,
        yaxis_range=[0, 1],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)")
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/6/69/IMDb_Logo_Alternative.svg",
        width=100,
    )
    st.markdown("### GRU Neural Engine")
    st.divider()

    if artifacts["status"] == "Success":
        st.markdown('<span class="status-badge status-online"><span class="pulse-dot"></span>ENGINE ACTIVE</span>', unsafe_allow_html=True)
        cfg = artifacts["config"]
        st.markdown(f"""
        <div style="margin-top:14px;">
            <div class="info-row"><span class="info-key">Max vocab features</span><span class="info-val">{cfg.get('max_features','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Max token length</span><span class="info-val">{cfg.get('max_length','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Embedding dim</span><span class="info-val">{cfg.get('embedding_length','N/A')}</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-offline">⚠ OFFLINE</span>', unsafe_allow_html=True)
        st.warning(
            f"**Status:** {artifacts['status']}\n\n"
            "Ensure these files are present:\n"
            "- `best_gru_model.keras`\n"
            "- `tokenizer.pkl`\n"
            "- `model_config.pkl`"
        )

    st.divider()
    st.caption("Developed by Akshit Gajera · Portfolio Core Engine")


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 0.5rem 0 1.5rem;">
    <p class="hero-title">IMDB Sentiment GRU Engine</p>
    <p class="hero-sub">Deep Sequence Recurrent Neural Network · NLP Classification Pipeline</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  Real-Time Classifier", "📊  Validation Metrics", "📁  Batch Evaluation"])


# ════════════════════════════════════════════
# TAB 1 — REAL-TIME CLASSIFIER
# ════════════════════════════════════════════
with tab1:
    st.markdown("#### Sentiment Analysis Pipeline")
    st.caption("Enter a movie review below. The GRU model will predict sentiment and confidence.")

    # Quick-fill sample chips (implemented as columns of small buttons)
    SAMPLES = {
        "⭐ Masterpiece":        "The film was an absolute masterpiece. Every scene was crafted with precision and care. The performances were stunning and the direction visionary.",
        "💀 Terrible film":     "A terrible waste of time and money. The plot made no sense, the acting was wooden, and the pacing was painfully slow.",
        "💎 Hidden gem":        "A hidden gem of the decade. The story is heartwarming and original. The chemistry between the leads is electric and the soundtrack unforgettable.",
        "😴 Boring & bland":    "Boring and completely predictable. You can see every twist coming. The dialogue felt forced and none of the characters were likeable.",
    }

    cols_q = st.columns(len(SAMPLES))
    for col, (label, text) in zip(cols_q, SAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state["review_text"] = text

    user_review = st.text_area(
        "Movie review:",
        value=st.session_state.get("review_text", ""),
        placeholder="e.g. 'The cinematography was breathtaking and performances truly unforgettable…'",
        height=160,
        key="review_input",
    )

    analyze_btn = st.button("▶  Analyze Sentiment", type="primary", use_container_width=False)

    if analyze_btn:
        if artifacts["status"] != "Success":
            st.error("Model artifacts missing or failed to load. Check the sidebar for details.")
        elif not user_review.strip():
            st.warning("Please enter a review before running analysis.")
        else:
            with st.spinner("Running text through GRU layers…"):
                is_pos, conf, raw_score = predict_single(user_review)

            pct       = round(conf * 100, 1)
            neg_pct   = round(raw_score * 100, 1) if not is_pos else round((1 - conf) * 100, 1)
            pos_pct   = round(raw_score * 100, 1) if is_pos else round((1 - conf) * 100, 1)
            sent_cls  = "Positive" if is_pos else "Negative"
            card_cls  = "result-positive" if is_pos else "result-negative"
            title_cls = "pos" if is_pos else "neg"
            emoji     = "🌟" if is_pos else "🍿"
            desc      = (
                "The GRU model registers an upbeat, commendatory review pattern."
                if is_pos else
                "The GRU model registers a critical or unfavorable review pattern."
            )
            fill_color = "#2ecc71" if is_pos else "#E24B4A"

            st.markdown(f"""
            <div class="result-card {card_cls}" style="margin-top:1.2rem;">
                <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;">
                    <div>
                        <div class="result-emoji">{emoji}</div>
                        <div class="result-title {title_cls}">{sent_cls} Sentiment</div>
                        <div class="result-desc">{desc}</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="conf-pct" style="color:{'#2ecc71' if is_pos else '#E24B4A'}">{pct}%</div>
                        <div style="font-size:.78rem;color:#888;">confidence</div>
                    </div>
                </div>

                <div style="margin-top:16px;">
                    <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#888;margin-bottom:4px;">
                        <span>Negative</span><span>Positive</span>
                    </div>
                    <div style="position:relative;height:10px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden;">
                        <div style="position:absolute;left:0;top:0;height:100%;width:{neg_pct}%;background:#E24B4A;border-radius:99px;"></div>
                        <div style="position:absolute;right:0;top:0;height:100%;width:{pos_pct}%;background:#2ecc71;border-radius:99px;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#888;margin-top:3px;">
                        <span>{neg_pct}%</span><span>{pos_pct}%</span>
                    </div>
                </div>

                <div style="margin-top:14px;">
                    <div style="font-size:.72rem;color:#888;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Raw GRU score</div>
                    <div style="font-size:.88rem;color:#ccc;font-family:monospace;">{raw_score:.6f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — VALIDATION METRICS
# ════════════════════════════════════════════
with tab2:
    st.markdown("#### Model Evaluation Analytics")
    st.caption("Validation metrics recorded over **9,917 serialised review tensors**.")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""<div class="metric-card"><div class="metric-label">Test Accuracy</div><div class="metric-value">88%</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card"><div class="metric-label">F1 Scores (Neg / Pos)</div><div class="metric-value">0.89 / 0.88</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card"><div class="metric-label">Test Set Rows</div><div class="metric-value">9,917</div></div>""", unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(get_confusion_fig(), use_container_width=True)
    with c2:
        st.plotly_chart(get_bar_fig(), use_container_width=True)

    with st.expander("📋 Raw Classification Report"):
        st.code("""
Confusion Matrix (GRU):
[[4439  500]
 [ 648 4330]]

Classification Report:
              precision  recall  f1-score  support
           0       0.87    0.90      0.89     4939
           1       0.90    0.87      0.88     4978
    accuracy                         0.88     9917
   macro avg       0.88    0.88      0.88     9917
weighted avg       0.88    0.88      0.88     9917
        """, language="text")


# ════════════════════════════════════════════
# TAB 3 — BATCH EVALUATION
# ════════════════════════════════════════════
with tab3:
    st.markdown("#### Batch Inference Pipeline")
    st.caption("Upload a `.csv` or `.xlsx` file with review strings to run high-throughput classification.")

    uploaded = st.file_uploader(
        "Drop target dataset here",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            df_batch = (
                pd.read_csv(uploaded)
                if uploaded.name.endswith(".csv")
                else pd.read_excel(uploaded)
            )
            st.success(f"✅ Loaded **{uploaded.name}** — {len(df_batch):,} rows")

            col_pick = st.selectbox(
                "Select the text column:",
                options=df_batch.columns.tolist(),
            )

            if st.button("▶  Run Batch Inference", type="primary"):
                if artifacts["status"] != "Success":
                    st.error("Model offline. Verify artifact files.")
                else:
                    model     = artifacts["model"]
                    tokenizer = artifacts["tokenizer"]
                    config    = artifacts["config"]

                    texts = df_batch[col_pick].astype(str).tolist()

                    with st.status("Processing batch…", expanded=True) as status:
                        st.write("Tokenising sequences…")
                        seqs   = tokenizer.texts_to_sequences(texts)
                        padded = pad_sequences(
                            seqs,
                            maxlen=config.get("max_length", 100),
                            padding="post",
                            truncating="post",
                        )
                        st.write("Running GRU inference…")
                        preds = model.predict(padded, verbose=0).flatten()
                        df_batch["Predicted_Score"]     = preds
                        df_batch["Predicted_Sentiment"] = np.where(preds >= 0.5, "Positive", "Negative")
                        status.update(label="✅ Batch inference complete!", state="complete")

                    st.divider()
                    st.markdown("##### Preview (first 10 rows)")
                    st.dataframe(df_batch.head(10), use_container_width=True)

                    fig_dist = px.histogram(
                        df_batch,
                        x="Predicted_Sentiment",
                        color="Predicted_Sentiment",
                        title="Batch Sentiment Distribution",
                        color_discrete_map={"Positive": "#2ecc71", "Negative": "#E24B4A"},
                    )
                    fig_dist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#ccc",
                        showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    fig_dist.update_yaxes(gridcolor="rgba(255,255,255,.06)")
                    st.plotly_chart(fig_dist, use_container_width=True)

                    st.download_button(
                        label="⬇  Download Predictions CSV",
                        data=df_batch.to_csv(index=False).encode("utf-8"),
                        file_name="imdb_gru_predictions.csv",
                        mime="text/csv",
                        use_container_width=False,
                    )

        except Exception as e:
            st.error(f"Failed to process file: {e}")