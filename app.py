import io
from pathlib import Path
import streamlit as st
import pandas as pd
from logic.agent import load_excel, build_agent, ask_question


# =====================================================================================
# PAGE CONFIG
# =====================================================================================

st.set_page_config(
    page_title="MILOlytics – myBasePay Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================================================
# SAFE CSS — DOES NOT INTERFERE WITH STREAMLIT CORE
# =====================================================================================

st.markdown("""
<style>

    /* === MYBASEPAY COLOR PALETTE === */
    :root {
        --mbp-navy: #0A1F44;
        --mbp-cyan: #2BB3C0;
        --mbp-light: #F7F9FC;
        --mbp-shadow: rgba(0,0,0,0.07);
    }

    /* BODY BACKGROUND (safe) */
    .block-container {
        background-color: var(--mbp-light);
        padding-top: 1rem;
    }

    /* LOGO container (isolated, safe) */
    #mbp-logo {
        position: absolute;
        top: 15px;
        right: 30px;
        z-index: 9999;
    }
    #mbp-logo img {
        width: 170px;
        height: auto;
    }

    /* CARD-LIKE COMPONENT (safe) */
    .mbp-card {
        background-color: white;
        padding: 20px 24px;
        border-radius: 12px;
        box-shadow: 0 3px 10px var(--mbp-shadow);
        border-left: 5px solid var(--mbp-cyan);
        margin-top: 15px;
    }

    /* STATS BOX (safe) */
    .mbp-stats-box {
        background-color: white;
        padding: 18px 20px;
        border-radius: 10px;
        box-shadow: 0 3px 10px var(--mbp-shadow);
        margin-bottom: 15px;
    }

    /* BUTTONS (safe) */
    .stButton>button {
        background-color: var(--mbp-navy);
        color: white;
        border-radius: 6px;
        padding: 0.55rem 1.2rem;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: var(--mbp-cyan);
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# =====================================================================================
# LOGO
# =====================================================================================

logo_path = Path("mybasepay_logo.png")
if logo_path.exists():
    st.markdown(
        f"""
        <div id="mbp-logo">
            <img src="{logo_path.as_posix()}">
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================================================
# HEADER
# =====================================================================================

st.title("MILOlytics – myBasePay Ticket Analytics Assistant")
st.write("Analyze call center tickets, trends, SLA performance, and outliers with ease.")


# =====================================================================================
# SIDEBAR — UPLOAD FILE
# =====================================================================================

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=["xlsx"]
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
        st.success("Dataset loaded successfully.")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

elif default_path.exists():
    try:
        df_data, system_text, sample_questions = load_source(None)
        data_loaded = True
        st.info("Using default dataset (BlockData.xlsx).")
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")


# =====================================================================================
# HELPER FUNCTIONS FOR STATS
# =====================================================================================

def human_time(seconds):
    if pd.isna(seconds):
        return "N/A"
    seconds = float(seconds)
    return f"{int(seconds):,} sec ({seconds/3600:.1f} hrs)"


def compute_stats(df):
    stats = {}

    stats["total"] = len(df)

    # SLA
    if "Due_Date_Time" in df.columns and "Resolution_Date_Time" in df.columns:
        valid = df.dropna(subset=["Due_Date_Time", "Resolution_Date_Time"])
        if len(valid) > 0:
            late = (valid["Resolution_Date_Time"] > valid["Due_Date_Time"]).sum()
            stats["sla_rate"] = 100 - (late / len(valid) * 100)
            stats["outside_sla"] = late
        else:
            stats["sla_rate"] = stats["outside_sla"] = None

    # Resolution time stats
    if "Resolution_Time_Seconds" in df.columns:
        rt = df["Resolution_Time_Seconds"].dropna()
        if len(rt) > 0:
            stats["avg"] = human_time(rt.mean())
            stats["min"] = human_time(rt.min())
            stats["max"] = human_time(rt.max())
        else:
            stats["avg"] = stats["min"] = stats["max"] = "N/A"

    # Fastest / slowest agent
    if "Resolved_By" in df.columns and "Resolution_Time_Seconds" in df.columns:
        df2 = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"])
        if len(df2) > 0:
            grouped = df2.groupby("Resolved_By")["Resolution_Time_Seconds"].mean()
            fastest = grouped.sort_values().head(1)
            slowest = grouped.sort_values().tail(1)

            stats["fastest_agent"] = f"{fastest.index[0]} ({human_time(fastest.values[0])})"
            stats["slowest_agent"] = f"{slowest.index[0]} ({human_time(slowest.values[0])})"
        else:
            stats["fastest_agent"] = stats["slowest_agent"] = "N/A"

    return stats


# =====================================================================================
# MAIN LAYOUT
# =====================================================================================

if data_loaded and df_data is not None:

    left, right = st.columns([2.5, 1])

    # ---------------------------------------
    # LEFT SIDE – Question Input
    # ---------------------------------------
    with left:
        st.subheader("Ask a Question")

        user_q = st.text_input(
            "Your Question:",
            placeholder="Example: Which ticket took the longest to resolve?"
        )

        if st.button("Submit Question"):
            if not user_q.strip():
                st.error("Please enter a question first.")
            else:
                with st.spinner("Analyzing dataset..."):
                    agent = build_agent(df_data, system_text)
                    response = ask_question(agent, user_q, system_text, df_data)

                st.markdown("<div class='mbp-card'>", unsafe_allow_html=True)
                st.markdown(response)
                st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------
    # RIGHT SIDE – Stats Panel
    # ---------------------------------------
    with right:

        st.subheader("📊 Quick Stats")

        stats = compute_stats(df_data)

        st.markdown("<div class='mbp-stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Total Tickets:** {stats['total']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='mbp-stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Avg Resolution Time:** {stats['avg']}")
        st.markdown(f"**Fastest Ticket:** {stats['min']}")
        st.markdown(f"**Slowest Ticket:** {stats['max']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='mbp-stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Fastest Agent:** {stats['fastest_agent']}")
        st.markdown(f"**Slowest Agent:** {stats['slowest_agent']}")
        st.markdown("</div>", unsafe_allow_html=True)

        if "sla_rate" in stats:
            st.markdown("<div class='mbp-stats-box'>", unsafe_allow_html=True)
            st.markdown(f"**SLA Compliance:** {stats['sla_rate']:.1f}%")
            st.markdown(f"**Outside SLA:** {stats['outside_sla']}")
            st.markdown("</div>", unsafe_allow_html=True)
