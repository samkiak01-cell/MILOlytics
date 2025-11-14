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
# Custom CSS (MyBasePay Styling)
# =====================================================

st.markdown("""
<style>

body {
    background-color: #F7F9FC;
    font-family: 'Inter', sans-serif;
}

/* Top-right logo */
.logo-container {
    position: absolute;
    top: 15px;
    right: 25px;
    z-index: 1000;
}
.logo-container img {
    width: 160px;
    height: auto;
}

/* Answer box */
.answer-box {
    background-color: white;
    padding: 22px;
    border-radius: 12px;
    border-left: 5px solid #2BB3C0;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.06);
    margin-top: 20px;
}

/* Stats box */
.stats-box {
    background-color: white;
    padding: 18px 22px;
    border-radius: 10px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

/* Button styling */
.stButton>button {
    background-color: #0A1F44;
    color: white;
    border-radius: 6px;
    padding: 0.6rem 1.2rem;
    font-size: 16px;
    border: none;
}
.stButton>button:hover {
    background-color: #2BB3C0;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# Header + Logo
# =====================================================

st.title("MILOlytics – myBasePay Ticket Analytics Assistant")
st.write("Upload your call center ticket dataset. MILOlytics will analyze trends, outliers, SLA performance, and more.")

# Display logo
logo_path = Path("mybasepay_logo.png")
if logo_path.exists():
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="{logo_path.as_posix()}"/>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# Sidebar — Upload File
# =====================================================

st.sidebar.header("Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your BlockData-style Excel file",
    type=["xlsx"],
    help="Must include similar column headers."
)

default_path = Path("data/BlockData.xlsx")


@st.cache_data(show_spinner=True)
def load_source(file_bytes: bytes | None):
    """Load Excel from file upload or default."""
    if file_bytes:
        return load_excel(io.BytesIO(file_bytes))
    return load_excel(default_path)


df_data = None
system_text = ""
sample_questions = []
data_loaded = False


# Try loading dataset
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
        st.info("Using default BlockData.xlsx dataset.")
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")

else:
    st.warning("Upload a dataset to continue.")


# =====================================================
# Computation Helpers for Stats Panel
# =====================================================

def human_time(seconds):
    """Convert seconds to hr/min/day format."""
    if pd.isna(seconds):
        return "N/A"
    seconds = float(seconds)
    hours = seconds / 3600
    days = seconds / 86400
    return f"{int(seconds):,} sec ({hours:.1f} hrs / {days:.1f} days)"


def compute_stats(df):
    stats = {}

    # Total tickets
    stats["total"] = len(df)

    # SLA
    if "Due_Date_Time" in df and "Resolution_Date_Time" in df:
        valid = df.dropna(subset=["Due_Date_Time", "Resolution_Date_Time"])
        outside = (valid["Resolution_Date_Time"] > valid["Due_Date_Time"]).sum()
        total = len(valid)
        stats["sla_rate"] = 100 - (outside / total * 100) if total > 0 else None
        stats["outside_sla"] = outside

    # Resolution Time Stats
    if "Resolution_Time_Seconds" in df:
        rt = df["Resolution_Time_Seconds"].dropna()
        if not rt.empty:
            stats["average_time"] = human_time(rt.mean())
            stats["min_time"] = human_time(rt.min())
            stats["max_time"] = human_time(rt.max())
        else:
            stats["average_time"] = stats["min_time"] = stats["max_time"] = "N/A"

    # Fastest / slowest agent (Resolved_By)
    if "Resolved_By" in df and "Resolution_Time_Seconds" in df:
        g = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"]) \
              .groupby("Resolved_By")["Resolution_Time_Seconds"]

        if not g.empty:
            fastest = g.mean().sort_values().head(1)
            slowest = g.mean().sort_values().tail(1)

            stats["fastest_agent"] = (
                f"{fastest.index[0]} ({human_time(fastest.values[0])})"
            )
            stats["slowest_agent"] = (
                f"{slowest.index[0]} ({human_time(slowest.values[0])})"
            )
        else:
            stats["fastest_agent"] = stats["slowest_agent"] = "N/A"

    return stats


# =====================================================
# Main Layout
# =====================================================

if data_loaded and df_data is not None:

    # Create two columns: left=ask question, right=stats
    left_col, right_col = st.columns([2.5, 1])

    # ----------------------------------------
    # LEFT SIDE — Question Interface
    # ----------------------------------------

    with left_col:

        st.subheader("Ask a Question")

        user_question = st.text_input(
            "Enter your question:",
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

    # ----------------------------------------
    # RIGHT SIDE — Stats Panel
    # ----------------------------------------

    with right_col:

        st.subheader("📊 Quick Statistics")

        stats = compute_stats(df_data)

        st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Total Tickets:** {stats['total']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Average Resolution Time:** {stats['average_time']}")
        st.markdown(f"**Fastest Ticket:** {stats['min_time']}")
        st.markdown(f"**Slowest Ticket:** {stats['max_time']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
        st.markdown(f"**Fastest Agent:** {stats['fastest_agent']}")
        st.markdown(f"**Slowest Agent:** {stats['slowest_agent']}")
        st.markdown("</div>", unsafe_allow_html=True)

        if "sla_rate" in stats:
            st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
            st.markdown(f"**SLA Compliance:** {stats['sla_rate']:.1f}%")
            st.markdown(f"**Outside SLA:** {stats['outside_sla']}")
            st.markdown("</div>", unsafe_allow_html=True)
