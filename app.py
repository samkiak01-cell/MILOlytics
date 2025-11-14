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
    --mbp-navy-softer: #030712;
    --mbp-navy-soft: #020617;
    --mbp-cyan: #2BB3C0;
    --mbp-cyan-soft: rgba(43, 179, 192, 0.35);
    --mbp-text: #F9FAFB;
    --mbp-subtext: #9CA3AF;
}

/* Overall page background & text */
.block-container {
    background: radial-gradient(circle at top left, #0b1120, #020617 55%, #000000 100%);
    color: var(--mbp-text);
    padding-top: 1.5rem;
}

/* Header text tweaks */
h1, h2, h3, h4 {
    color: var(--mbp-text);
}

/* Header badge */
.mbp-header-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--mbp-text);
}
.mbp-header-sub {
    font-size: 0.95rem;
    color: var(--mbp-subtext);
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

/* Strong labels inside stats */
.stats-box strong {
    color: var(--mbp-text);
}

/* Button styling */
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

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: radial-gradient(circle at top, #020617, #030712);
    color: var(--mbp-text);
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: var(--mbp-text);
}

/* Text input */
.stTextInput>div>div>input {
    background-color: #020617;
    color: var(--mbp-text);
    border-radius: 8px;
    border: 1px solid #1f2937;
}

/* File uploader */
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
        <div class="mbp-header-title">MILOlytics – myBasePay Ticket Analytics</div>
        <div class="mbp-header-sub">
            A dark-mode analytics workspace for call center performance, SLA tracking, and outlier detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    logo_path = Path("mybasepay_logo.png")
    if logo_path.exists():
        st.image(str(logo_path), use_column_width=False)
    else:
        st.write("")  # silent if not present


st.markdown("---")


# =====================================================
# Sidebar — Upload Dataset
# =====================================================

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose a BlockData-style Excel file (.xlsx)",
    type=["xlsx"],
    help="File should follow the BlockData structure with 'Data' sheet and expected columns.",
)

default_path = Path("data/BlockData.xlsx")


@st.cache_data(show_spinner=True)
def load_source(file_bytes: bytes | None):
    if file_bytes:
        return load_excel(io.BytesIO(file_bytes))
    return load_excel(default_path)


df_data = None
system_text = ""
sample_questions = []
data_loaded = False

if uploaded_file:
    try:
        df_data, system_text, sample_questions = load_source(uploaded_file.read())
        data_loaded = True
        st.sidebar.success("Dataset loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")
elif default_path.exists():
    try:
        df_data, system_text, sample_questions = load_source(None)
        data_loaded = True
        st.sidebar.info("Using default dataset: data/BlockData.xlsx")
    except Exception as e:
        st.sidebar.error(f"Error loading default dataset: {e}")
else:
    st.sidebar.warning("Upload a dataset to begin.")


# =====================================================
# Helper Functions for Stats
# =====================================================

def human_time(seconds):
    """Convert seconds to a human-friendly string."""
    if pd.isna(seconds):
        return "N/A"
    seconds = float(seconds)
    hours = seconds / 3600
    days = seconds / 86400
    return f"{int(seconds):,} sec ({hours:.1f} hrs / {days:.1f} days)"


def compute_stats(df: pd.DataFrame):
    stats = {}

    # Total tickets
    stats["total"] = len(df)

    # SLA Stats
    if "Due_Date_Time" in df.columns and "Resolution_Date_Time" in df.columns:
        valid = df.dropna(subset=["Due_Date_Time", "Resolution_Date_Time"])
        if len(valid) > 0:
            outside = (valid["Resolution_Date_Time"] > valid["Due_Date_Time"]).sum()
            total = len(valid)
            stats["sla_rate"] = 100 - (outside / total * 100)
            stats["outside_sla"] = outside
        else:
            stats["sla_rate"] = None
            stats["outside_sla"] = None
    else:
        stats["sla_rate"] = None
        stats["outside_sla"] = None

    # Resolution Time Stats
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
    else:
        stats["avg"] = stats["min"] = stats["max"] = "N/A"

    # Fastest / slowest agent (Resolved_By)
    if "Resolved_By" in df.columns and "Resolution_Time_Seconds" in df.columns:
        df_res = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"])
        if len(df_res) > 0:
            grouped = df_res.groupby("Resolved_By")["Resolution_Time_Seconds"].mean()
            if len(grouped) > 0:
                fastest = grouped.sort_values().head(1)
                slowest = grouped.sort_values().tail(1)
                stats["fastest_agent"] = (
                    f"{fastest.index[0]} ({human_time(fastest.values[0])})"
                )
                stats["slowest_agent"] = (
                    f"{slowest.index[0]} ({human_time(slowest.values[0])})"
                )
            else:
                stats["fastest_agent"] = "N/A"
                stats["slowest_agent"] = "N/A"
        else:
            stats["fastest_agent"] = "N/A"
            stats["slowest_agent"] = "N/A"
    else:
        stats["fastest_agent"] = "N/A"
        stats["slowest_agent"] = "N/A"

    return stats


# =====================================================
# Main Layout
# =====================================================

if data_loaded and df_data is not None:

    left_col, right_col = st.columns([2.7, 1.3])

    # ------------------------------
    # LEFT — Question + Answer
    # ------------------------------
    with left_col:
        st.subheader("Ask MILOlytics a Question")

        user_question = st.text_input(
            "Your Question",
            placeholder="Example: Which ticket took the longest to resolve?",
        )

        if st.button("Submit Question"):
            if not user_question.strip():
                st.error("Please enter a question.")
            else:
                with st.spinner("Analyzing dataset..."):
                    try:
                        agent = build_agent(df_data, system_text)
                        answer = ask_question(agent, user_question, system_text, df_data)

                        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                        st.markdown(answer)
                        st.markdown("</div>", unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    # ------------------------------
    # RIGHT — Stats Panel
    # ------------------------------
    with right_col:
        st.subheader("📊 Quick Statistics")

        stats = compute_stats(df_data)

        # Total tickets
        st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Total Tickets:** {stats['total']}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Resolution times
        st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Average Resolution Time:** {stats['avg']}")
        st.markdown(f"**Fastest Ticket:** {stats['min']}")
        st.markdown(f"**Slowest Ticket:** {stats['max']}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Agents
        st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Fastest Agent:** {stats['fastest_agent']}")
        st.markdown(f"**Slowest Agent:** {stats['slowest_agent']}")
        st.markdown("</div>", unsafe_allow_html=True)

        # SLA
        if stats["sla_rate"] is not None:
            st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
            st.markdown(f"**SLA Compliance:** {stats['sla_rate']:.1f}%")
            st.markdown(f"**Outside SLA:** {stats['outside_sla']}")
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Upload a dataset using the sidebar to get started.")
