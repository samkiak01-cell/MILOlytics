import io
from pathlib import Path

import pandas as pd
import streamlit as st

from logic.agent import load_excel, ask_question   # build_agent removed


# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="Call Sensei",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# Dark MyBasePay Dashboard CSS
# =====================================================

st.markdown(
    """
<style>

:root {
    --mbp-navy: #020617;
    --mbp-cyan: #2BB3C0;
    --mbp-text: #F9FAFB;
    --mbp-subtext: #9CA3AF;
}

/* Overall page background */
.block-container {
    background: radial-gradient(circle at top left, #0b1120, #020617 55%, #000000 100%);
    color: var(--mbp-text);
    padding-top: 4rem !important;
}

/* Header title */
.mbp-header-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.mbp-header-sub {
    font-size: 1rem;
    color: var(--mbp-subtext);
}

/* Logo */
.mbp-logo-wrapper {
    background: white;
    padding: 12px 22px;
    border-radius: 40px;
    display: inline-block;
    box-shadow: 0px 5px 20px rgba(255,255,255,0.12);
}

.answer-box {
    background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(15,23,42,0.9));
    padding: 20px 22px;
    border-radius: 14px;
    border-left: 4px solid var(--mbp-cyan);
    box-shadow: 0px 10px 30px rgba(0,0,0,0.45);
    margin-top: 16px;
    color: var(--mbp-text);
}

.stats-box {
    background: radial-gradient(circle at top left, rgba(15,23,42,0.9), rgba(15,23,42,0.95));
    padding: 14px 18px;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.25);
    margin-bottom: 12px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.35);
}

.stButton>button {
    background: linear-gradient(135deg, var(--mbp-cyan), #06b6d4);
    color: #0f172a;
    border-radius: 999px;
    padding: 0.55rem 1.5rem;
    font-size: 15px;
    font-weight: 600;
    border: none;
    box-shadow: 0px 4px 14px rgba(8,145,178,0.5);
}
.stButton>button:hover {
    background: linear-gradient(135deg, #06b6d4, var(--mbp-cyan));
    color: #020617;
}

[data-testid="stSidebar"] {
    background: radial-gradient(circle at top, #020617, #030712);
    color: var(--mbp-text);
}

.stTextInput>div>div>input {
    background-color: #020617;
    color: var(--mbp-text);
    border-radius: 8px;
    border: 1px solid #1f2937;
}

[data-testid="stFileUploader"] section {
    background-color: #020617;
    border-radius: 10px;
    border: 1px dashed #475569;
}

</style>
""",
    unsafe_allow_html=True,
)


# =====================================================
# Header (Title + Logo)
# =====================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        """
        <div class="mbp-header-title">Call Sensei - myBasePay Call Center AI Agent</div>
        <div class="mbp-header-sub">
            AI-powered insights for call center performance, SLA tracking, and trend detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    logo_path = Path("mybasepay_logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=120)
    else:
        st.write("")

st.markdown("---")


# =====================================================
# Load Dataset (NOT from upload — from repository)
# =====================================================

DATA_PATH = Path("data/Demo Data.xlsx")   # your real demo data

df_data, system_text, sample_questions = load_excel(DATA_PATH)
data_loaded = True


# =====================================================
# Helper Functions
# =====================================================

def human_time(seconds):
    if pd.isna(seconds):
        return "N/A"
    seconds = float(seconds)
    hours = seconds / 3600
    days = seconds / 86400
    return f"{int(seconds):,} sec ({hours:.1f} hrs / {days:.1f} days)"


def compute_stats(df: pd.DataFrame):
    stats = {}
    stats["total"] = len(df)

    if "Resolution_Time_Seconds" in df.columns:
        rt = df["Resolution_Time_Seconds"].dropna()
        stats["avg"] = human_time(rt.mean()) if len(rt) else "N/A"
        stats["min"] = human_time(rt.min()) if len(rt) else "N/A"
        stats["max"] = human_time(rt.max()) if len(rt) else "N/A"

    if "Resolved_By" in df.columns and "Resolution_Time_Seconds" in df.columns:
        resolved_df = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"])
        if len(resolved_df) > 0:
            grouped = resolved_df.groupby("Resolved_By")["Resolution_Time_Seconds"].mean()
            stats["fastest_agent"] = f"{grouped.idxmin()} ({human_time(grouped.min())})"
            stats["slowest_agent"] = f"{grouped.idxmax()} ({human_time(grouped.max())})"
        else:
            stats["fastest_agent"] = "N/A"
            stats["slowest_agent"] = "N/A"

    if "SLA_Met" in df.columns:
        stats["sla_rate"] = df["SLA_Met"].mean() * 100
        stats["outside_sla"] = (df["SLA_Met"] == False).sum()

    return stats


# =====================================================
# Main UI
# =====================================================

left_col, right_col = st.columns([2.7, 1.3])

with left_col:
    st.subheader("Ask the Sensei a Question")

    user_q = st.text_input(
        "Your Question",
        placeholder="Example: Which ticket took the longest to resolve?",
    )

    if st.button("Submit Question"):
        if not user_q.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Analyzing dataset..."):
                answer = ask_question(df_data, user_q, system_text)

            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)


with right_col:
    st.subheader("📊 Quick Stats")

    stats = compute_stats(df_data)

    st.markdown(f"<div class='stats-box'><b>Total Tickets:</b> {stats['total']}</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
<div class='stats-box'>
<b>Avg Resolution:</b> {stats['avg']}<br>
<b>Fastest Ticket:</b> {stats['min']}<br>
<b>Slowest Ticket:</b> {stats['max']}
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class='stats-box'>
<b>Fastest Agent:</b> {stats['fastest_agent']}<br>
<b>Slowest Agent:</b> {stats['slowest_agent']}
</div>
""",
        unsafe_allow_html=True,
    )

    if stats.get("sla_rate") is not None:
        st.markdown(
            f"""
<div class='stats-box'>
<b>SLA Compliance:</b> {stats['sla_rate']:.1f}%<br>
<b>Outside SLA:</b> {stats['outside_sla']}
</div>
""",
            unsafe_allow_html=True,
        )
