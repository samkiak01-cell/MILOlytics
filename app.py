import io
from pathlib import Path
import streamlit as st
import pandas as pd

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
# Minimal, Safe CSS
# =====================================================

st.markdown("""
<style>

.answer-box {
    background-color: #FFFFFF;
    padding: 22px;
    border-radius: 12px;
    border-left: 5px solid #2BB3C0;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.06);
    margin-top: 15px;
}

.stats-box {
    background-color: #FFFFFF;
    padding: 18px 20px;
    border-radius: 10px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

/* Button styling */
.stButton>button {
    background-color: #0A1F44;
    color: #FFFFFF;
    border-radius: 6px;
    padding: 0.55rem 1.2rem;
    font-size: 16px;
    border: none;
}
.stButton>button:hover {
    background-color: #2BB3C0;
    color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# Header Row (Title + Logo)
# =====================================================

header_col_left, header_col_right = st.columns([3, 1])

with header_col_left:
    st.title("MILOlytics – myBasePay Ticket Analytics Assistant")
    st.write(
        "Upload your call center ticket dataset. "
        "MILOlytics will help you analyze trends, SLA performance, outliers, and more."
    )

with header_col_right:
    logo_path = Path("mybasepay_logo.png")
    if logo_path.exists():
        st.image(str(logo_path), use_column_width=False)
    else:
        st.write("")  # stay silent if logo missing


# =====================================================
# Sidebar — Upload File
# =====================================================

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=["xlsx"],
    help="File should follow the BlockData structure."
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
        st.info("Using default dataset: BlockData.xlsx")
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
else:
    st.warning("Upload a dataset to continue.")


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
                stats["fastest_agent"] = f"{fastest.index[0]} ({human_time(fastest.values[0])})"
                stats["slowest_agent"] = f"{slowest.index[0]} ({human_time(slowest.values[0])})"
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

    left_col, right_col = st.columns([2.5, 1])

    # ------------------------------
    # LEFT — Question + Answer
    # ------------------------------
    with left_col:
        st.subheader("Ask a Question")

        user_question = st.text_input(
            "Your Question:",
            placeholder="Example: Which ticket took the longest to resolve?"
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

        st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Total Tickets:** {stats['total']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
        st.markmarkdown(f"**Average Resolution Time:** {stats['avg']}")
        st.markdown(f"**Fastest Ticket:** {stats['min']}")
        st.markdown(f"**Slowest Ticket:** {stats['max']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Fastest Agent:** {stats['fastest_agent']}")
        st.markdown(f"**Slowest Agent:** {stats['slowest_agent']}")
        st.markdown("</div>", unsafe_allow_html=True)

        if "sla_rate" in stats and stats["sla_rate"] is not None:
            st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
            st.markdown(f"**SLA Compliance:** {stats['sla_rate']:.1f}%")
            st.markdown(f"**Outside SLA:** {stats['outside_sla']}")
            st.markdown("</div>", unsafe_allow_html=True)
