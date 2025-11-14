import io
from pathlib import Path
import streamlit as st

from logic.agent import load_excel, build_agent, ask_question


# ================================================
# Page Config
# ================================================

st.set_page_config(
    page_title="MILOlytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================
# Custom CSS (MyBasePay Branding)
# ================================================

st.markdown("""
<style>

    /* Global Font + Background */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #F7F9FC;
    }

    /* Title */
    .main > div {
        padding-top: 1rem;
    }

    /* Logo Positioning */
    .logo-container {
        display: flex;
        justify-content: flex-end;
        margin-top: -70px;
    }
    .logo-container img {
        width: 180px;
    }

    /* Answer Box */
    .answer-box {
        background-color: white;
        padding: 20px 25px;
        border-radius: 12px;
        border-left: 5px solid #2BB3C0;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.06);
    }

    /* Button Styling */
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
        border: none;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #0A1F44 !important;
    }

    .css-1d391kg .stHeading, .css-1d391kg label, .css-1d391kg h2 {
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)


# ================================================
# Header Section
# ================================================

st.title("MILOlytics – Ticket Analytics Assistant")

st.markdown("""
Analyzing call-center tickets has never been easier.  
Upload your dataset and MILOlytics will help you uncover insights, trends, and performance breakthroughs.
""")

# Add logo in top right corner
if Path("mybasepay_logo.png").exists():
    st.markdown(
        """
        <div class="logo-container">
            <img src="mybasepay_logo.png">
        </div>
        """,
        unsafe_allow_html=True
    )


# ================================================
# Sidebar (Upload ONLY)
# ================================================

st.sidebar.header("Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your BlockData-style Excel file",
    type=["xlsx"],
    help="File must match the BlockData structure."
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


# Load data from upload or default
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
    st.warning("Please upload a dataset to continue.")


# ================================================
# Main Section — Ask a Question
# ================================================

if data_loaded and df_data is not None:

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
