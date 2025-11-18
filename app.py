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
    initial_sidebar_state="expanded",
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

/* Logo placement */
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

/* Dark text input */
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
        <div class="mbp-header-title">MILOlytics – myBasePay Ticket Assistant</div>
        <div class="mbp-header-sub">
            Our internal analytics agent for ticket resolution trends, SLA tracking, and call center insights.
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
# Sidebar — Upload Dataset
# =====================================================

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your Demo Data Excel (.xlsx)",
    type=["xlsx"],
)

# 🔹 Default is now Demo Data.xlsx only
default_path = Path("data/Demo Data.xlsx")


def load_source(file_bytes: bytes | None):
    """
    No caching on purpose to avoid stale BlockData.
    Always reads either the uploaded file or Demo Data.xlsx fresh.
    """
    if file_bytes:
        return load_excel(io.BytesIO(file_bytes))
    return load_excel(default_path)


df_data = None
system_text = ""
sample_questions = []
data_loaded = False
active_source_label = ""

if uploaded_file:
    try:
        df_data, system_text, sample_questions = load_source(uploaded_file.read())
        data_loaded = True
        active_source_label = f"Uploaded file: {uploaded_file.name}"
        st.sidebar.success("Dataset loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

elif default_path.exists():
    try:
        df_data, system_text, sample_questions = load_source(None)
        data_loaded = True
        active_source_label = "Default: Demo Data.xlsx"
        st.sidebar.info("Using default dataset: Demo Data.xlsx")
    except Exception as e:
        st.sidebar.error(f"Error loading default dataset: {e}")


if data_loaded:
    st.sidebar.caption(f"📂 Active dataset: {active_source_label}")


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
    """
    Quick Stats for the Demo Data schema:
      - Ticket_Number, Subject, Description, Caller, Status
      - Created_Date, Due_Date, Resolved_Date, Resolved_By
      - Resolution_Time_Seconds, SLA_Met
    """
    stats = {
        "total": len(df),
        "avg": "N/A",
        "min": "N/A",
        "max": "N/A",
        "fastest_agent": "N/A",
        "slowest_agent": "N/A",
        "sla_rate": None,
        "outside_sla": None,
    }

    # --- Resolution time stats ---
    if "Resolution_Time_Seconds" in df.columns:
        rt = pd.to_numeric(df["Resolution_Time_Seconds"], errors="coerce").dropna()
        if len(rt) > 0:
            stats["avg"] = human_time(rt.mean())
            stats["min"] = human_time(rt.min())
            stats["max"] = human_time(rt.max())

    # --- Agent performance (fastest/slowest avg resolution) ---
    if "Resolved_By" in df.columns and "Resolution_Time_Seconds" in df.columns:
        df_agents = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"]).copy()
        df_agents["Resolution_Time_Seconds"] = pd.to_numeric(
            df_agents["Resolution_Time_Seconds"], errors="coerce"
        )
        df_agents = df_agents.dropna(subset=["Resolution_Time_Seconds"])

        if not df_agents.empty:
            grouped = df_agents.groupby("Resolved_By")["Resolution_Time_Seconds"].mean()

            if len(grouped) > 0:
                # fastest
                fast_val = grouped.min()
                fast_names = list(grouped[grouped == fast_val].index)
                # slowest
                slow_val = grouped.max()
                slow_names = list(grouped[grouped == slow_val].index)

                def format_agents(names, seconds_value):
                    label_time = human_time(seconds_value)
                    if not names:
                        return "N/A"
                    if len(names) == 1:
                        return f"{names[0]} ({label_time})"
                    if len(names) == 2:
                        return f"{names[0]} & {names[1]} ({label_time})"
                    return f"{', '.join(names[:2])} + {len(names)-2} others ({label_time})"

                stats["fastest_agent"] = format_agents(fast_names, fast_val)
                stats["slowest_agent"] = format_agents(slow_names, slow_val)

    # --- SLA using SLA_Met boolean/yes-no ---
    if "SLA_Met" in df.columns:
        sla_series = df["SLA_Met"].dropna()
        total_sla = len(sla_series)
        if total_sla > 0:
            # handle True/False or strings like "Yes"/"No"
            normalized = sla_series.astype(str).str.lower()
            met_mask = normalized.isin(["true", "yes", "y", "1"])
            met_count = int(met_mask.sum())
            not_met = total_sla - met_count
            stats["sla_rate"] = met_count / total_sla * 100
            stats["outside_sla"] = not_met

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

                st.markdown(
                    f"""
<div class='answer-box'>
{answer}
</div>
""",
                    unsafe_allow_html=True,
                )

    # RIGHT SIDE — Quick Stats
    with right_col:

        st.subheader("📊 Quick Stats")

        stats = compute_stats(df_data)

        st.markdown(
            f"""
<div class='stats-box'>
<b>Total Tickets:</b> {stats['total']}
</div>
""",
            unsafe_allow_html=True,
        )

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

else:
    st.info("Upload a dataset using the sidebar to get started.")
