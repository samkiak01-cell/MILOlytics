import io
import os
from pathlib import Path

import streamlit as st

from logic.agent import load_excel, build_agent, ask_question


# ============================
# Streamlit Page Configuration
# ============================

st.set_page_config(
    page_title="myBasePay Ticket Analytics Assistant",
    layout="wide",
)


st.title("myBasePay Ticket Analytics Assistant")
st.write(
    "Ask questions about your call center tickets dataset. "
    "The assistant analyzes the Excel file and answers using real calculations."
)


# ============================
# 1. OPENAI API KEY
# ============================

st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Required for running the agent."
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.sidebar.warning("Enter your OpenAI API key to enable the assistant.")


# ============================
# 2. DATA SOURCE (UPLOAD OR DEFAULT)
# ============================

st.sidebar.subheader("Data Source")

default_excel_path = Path("data/BlockData.xlsx")

uploaded_file = st.sidebar.file_uploader(
    "Upload an Excel file (must match BlockData structure)",
    type=["xlsx"],
)


@st.cache_data(show_spinner=True)
def load_from_source(file_bytes: bytes | None, use_uploaded: bool):
    """
    Load Excel file either from upload (BytesIO) or from default path.
    Automatically cached by Streamlit.
    """
    if use_uploaded and file_bytes is not None:
        buffer = io.BytesIO(file_bytes)
        return load_excel(buffer)
    else:
        return load_excel(default_excel_path)


df_data = None
system_text = ""
sample_questions = []
data_loaded = False

# Load uploaded file
if uploaded_file is not None:
    try:
        df_data, system_text, sample_questions = load_from_source(
            uploaded_file.read(),
            use_uploaded=True
        )
        data_loaded = True
        st.success("Dataset loaded successfully from uploaded file.")
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")

# If no uploaded file, load default
elif default_excel_path.exists():
    try:
        df_data, system_text, sample_questions = load_from_source(
            None,
            use_uploaded=False
        )
        data_loaded = True
        st.info(f"Using default dataset: {default_excel_path}")
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")

else:
    st.warning("No dataset found. Upload an Excel file or place BlockData.xlsx in /data/.")


# ============================
# 3. MAIN INTERFACE
# ============================

if data_loaded and df_data is not None:

    st.subheader("Dataset Preview")
    st.dataframe(df_data.head(20), use_container_width=True)

    with st.expander("Column Names"):
        st.write(list(df_data.columns))

    # Sample questions
    st.sidebar.subheader("Sample Questions")
    selected_question = ""
    if sample_questions:
        selected_question = st.sidebar.selectbox(
            "Choose a sample question:",
            [""] + sample_questions
        )

    # Question input
    st.subheader("Ask a Question")
    user_question = st.text_input(
        "Enter your question:",
        value=selected_question,
        placeholder="Example: Which ticket took the longest to resolve?"
    )

    if st.button("Submit Question"):

        if not api_key:
            st.error("Please enter your OpenAI API key before submitting.")
        else:
            # Build the agent
            agent = build_agent(df_data, system_text)

            st.write("---")
            st.write("### Answer")

            with st.spinner("Analyzing the dataset..."):
                try:
                    answer = ask_question(agent, user_question, system_text)
                    st.success("Result:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred while processing the question: {e}")
