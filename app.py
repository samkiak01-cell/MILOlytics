import io
from pathlib import Path
import streamlit as st

from logic.agent import load_excel, build_agent, ask_question


# ================================================
# Streamlit Page Config
# ================================================

st.set_page_config(
    page_title="MILOlytics",
    layout="wide",
)

st.title("myBasePay Ticket Analytics Assistant")
st.write(
    "Upload call center tickets data. "
    "MILOlytics will help you with all your analytical needs."
)


# ================================================
# Sidebar — Upload File ONLY
# ================================================

st.sidebar.header("Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your BlockData-style Excel file",
    type=["xlsx"],
    help="Must include same column headers."
)

default_path = Path("data/BlockData.xlsx")


@st.cache_data(show_spinner=True)
def load_source(file_bytes: bytes | None):
    """Load Excel source from uploaded file or default dataset."""
    if file_bytes:
        return load_excel(io.BytesIO(file_bytes))
    return load_excel(default_path)


df_data = None
system_text = ""
sample_questions = []
data_loaded = False


# Load data
if uploaded_file:
    try:
        df_data, system_text, sample_questions = load_source(uploaded_file.read())
        data_loaded = True
        st.success("Dataset loaded successfully from uploaded file.")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

elif default_path.exists():
    try:
        df_data, system_text, sample_questions = load_source(None)
        data_loaded = True
        st.info("Using BlockData.xlsx")
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")

else:
    st.warning("Please upload a dataset to continue.")


# ================================================
# Main UI — Ask Question
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
                    # Agent is just the df in the new architecture
                    agent = build_agent(df_data, system_text)

                    # Ask question using new execution engine
                    answer = ask_question(
                        agent,
                        user_question,
                        system_text,
                        df_data  # pass DF directly
                    )

                    st.write("### Answer")
                    st.write(answer)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
