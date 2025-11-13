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
    "Ask analytical questions about your call center ticket data. "
    "Upload your Excel file and start querying."
)


# ============================
# 1. DATA UPLOAD (Sidebar Only)
# ============================

st.sidebar.header("Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload an Excel file (must follow BlockData structure)",
    type=["xlsx"],
)

default_path = Path("data/BlockData.xlsx")

@st.cache_data(show_spinner=True)
def load_source(file_bytes: bytes | None):
    if file_bytes:
        return load_excel(io.BytesIO(file_bytes))
    else:
        return load_excel(default_path)


df_data = None
system_text = ""
sample_questions = []
data_loaded = False


if uploaded_file:
    try:
        df_data, system_text, sample_questions = load_source(uploaded_file.read())
        data_loaded = True
        st.success("Dataset loaded from uploaded file.")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

elif default_path.exists():
    try:
        df_data, system_text, sample_questions = load_source(None)
        data_loaded = True
        st.info("Using default dataset from /data/BlockData.xlsx")
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")

else:
    st.warning("Please upload a dataset to continue.")


# ============================
# 2. QUESTION INPUT (MAIN UI)
# ============================

if data_loaded and df_data is not None:

    st.subheader("Ask a Question")

    user_question = st.text_input(
        "Enter your question:",
        placeholder="Example: Which ticket took the longest to resolve?"
    )

    if st.button("Submit"):
        with st.spinner("Analyzing..."):
            try:
                agent = build_agent(df_data, system_text)
                answer = ask_question(agent, user_question, system_text)
                st.write("### Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
