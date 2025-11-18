import io
from pathlib import Path

import pandas as pd
import streamlit as st

from logic.agent import load_excel, build_agent, ask_question


# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="MILOlytics – myBasePay Analytics",
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

/* White oval logo container */
.mbp-logo-wrapper {
    background: white;
    padding: 12px 22px;
    border-radius: 40px;
    display: inline-block;
    box-shadow: 0px 5px 20px rgba(255, 255, 255, 0.12);
}
.mbp-logo-container {
    display: flex;
    justify-content: flex-end;
    align-items: center;
}

/* Answer card */
.answer-box {
    background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(15,23,42,0.9));
    padding: 20px 22px;
    border-radius: 14px;
    border-left: 4px solid var(--mbp-cyan);
    box-shadow: 0px 10px 30px rgba(0,0,0,0.45);
    margin-top: 16px;
}

/* Stats cards */
.stats-box {
    background: radial-gradient(circle at top left, rgba(15,23,42,0.9), rgba(15,23,42,0.95));
    padding: 14px 18px;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.25);
    margin-bottom: 12px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.35);
}

/* Buttons */
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

/* Sidebar */
[data-testid="stSidebar"] {
    background: radial-gradient(circle at top, #020617, #030712);
    color: var(--mbp-text);
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: var(--mbp-text);
}

/* Text Input */
.stTextInput>div>div>input {
    background-color: #020617;
    color: var(--mbp-text);
    border-radius: 8px;
    border: 1px solid #1f2937;
}

/* File Uploader */
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
# Header Row (Title + Logo)
# =====================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        """
        <div class="mbp-header-title">MILOlytics – myBasePay Ticket Assistant</div>
        <div class="mbp-header-sub">
            Real-time performance insights, SLA monitoring, and call center analytics.
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    logo_path = Path("mybasepay_logo.png")
    if logo_path.exists():
        st.markdown("<div class='mbp-logo-wrapper'>", unsafe_allow_html=True)
        st.image(str(logo_path), width=115)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("")

st.markdown("---")


# =====================================================
# Sidebar — Upload Dataset
# =====================================================

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Demo Data (.xlsx)",
    type=["xlsx"],
    help="Upload your Demo Data file or rely on the default dataset.",
)

# NEW default dataset path
default_path = Path("data/Demo Data.xlsx")


@st.cache_data(show_spinner=True)
def load_source(file_bytes: bytes | None):
    if file_bytes:
        return load_excel(io.BytesIO(file_bytes))
    return load_excel(default_path)


df_data = None
system_text = ""
sample_questions = []
data_loaded = False


# Load dataset logic
if uploaded_file:
    try:
        df_data, system_text, sample_questions = load_source(uploaded_file.read())
        data_loaded = True
        st.sidebar.success("Dataset loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

elif default_path.exists():
    try:
        df_data, system_text, sample_questions = load_source(None)
        data_loaded = True
        st.sidebar.info("Using default Demo Data.xlsx")
    except Exception as e:
        st.sidebar.error(f"Error loading default dataset: {e}")


# =====================================================
# Helper Functions (Updated for DEMO DATA)
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

    # SLA (Demo Data uses Resolved_Date + Due_Date)
    if "Due_Date" in df.columns and "Resolved_Date" in df.columns:
        valid = df.dropna(subset=["Due_Date", "Resolved_Date"])
        if len(valid) > 0:
            late = (valid["Resolved_Date"] > valid["Due_Date"]).sum()
            stats["sla_rate"] = 100 - (late / len(valid) * 100)
            stats["outside_sla"] = late
        else:
            stats["sla_rate"] = None
            stats["outside_sla"] = None

    # Resolution Times
    if "Resolution_Time_Seconds" in df.columns:
        rt = df["Resolution_Time_Seconds"].dropna()
        if len(rt) > 0:
            stats["avg"] = human_time(rt.mean())
            stats["min"] = human_time(rt.min())
            stats["max"] = human_time(rt.max())
        else:
            stats["avg"] = "N/A"
            stats["min"] = "N/A"
            stats["max"] = "N/A"

    # Agent Performance
    if "Resolved_By" in df.columns:
        df2 = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"])
        if len(df2) > 0:
            grouped = df2.groupby("Resolved_By")["Resolution_Time_Seconds"].mean()
            if len(grouped) > 0:
                fast = grouped.sort_values().head(1)
                slow = grouped.sort_values().tail(1)
                stats["fastest_agent"] = f"{fast.index[0]} ({human_time(fast.values[0])})"
                stats["slowest_agent"] = f"{slow.index[0]} ({human_time(slow.values[0])})"
            else:
                stats["fastest_agent"] = "N/A"
                stats["slowest_agent"] = "N/A"

    return stats


# =====================================================
# Main Layout
# =====================================================

if data_loaded and df_data is not None:

    left_col, right_col = st.columns([2.7, 1.3])

    # LEFT SIDE — Ask MILOlytics
    with left_col:

        st.subheader("Ask MILOlytics a Question")

        user_q = st.text_input(
            "Your Question",
            placeholder="Example: Which ticket took the longest to resolve?",
        )

        if st.button("Submit Question"):
            if not user_q.strip():
                st.error("Please enter a question.")
            else:
                with st.spinner("Analyzing dataset..."):
                    agent = build_agent(df_data, system_text)
                    answer = ask_question(agent, user_q, system_text, df_data)

                st.markdown(f"""
<div class='answer-box'>
{answer}
</div>
""", unsafe_allow_html=True)

    # RIGHT SIDE — Quick Stats
    with right_col:

        st.subheader("📊 Quick Stats")

        stats = compute_stats(df_data)

        st.markdown(f"""
<div class='stats-box'>
<b>Total Tickets:</b> {stats['total']}
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class='stats-box'>
<b>Avg Resolution:</b> {stats['avg']}<br>
<b>Fastest Ticket:</b> {stats['min']}<br>
<b>Slowest Ticket:</b> {stats['max']}
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class='stats-box'>
<b>Fastest Agent:</b> {stats['fastest_agent']}<br>
<b>Slowest Agent:</b> {stats['slowest_agent']}
</div>
""", unsafe_allow_html=True)

        if stats.get("sla_rate") is not None:
            st.markdown(f"""
<div class='stats-box'>
<b>SLA Compliance:</b> {stats['sla_rate']:.1f}%<br>
<b>Outside SLA:</b> {stats['outside_sla']}
</div>
""", unsafe_allow_html=True)

else:
    st.info("Upload a dataset using the sidebar to get started.")
