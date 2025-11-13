# logic/agent.py

from pathlib import Path
from typing import Tuple, List, Union, IO

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


def load_excel(
    source: Union[str, Path, IO[bytes]]
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Load the Excel file and return:
      - df_data: main ticket dataset (Data sheet)
      - system_text: full description from 'System Prompt' sheet
      - sample_questions: list of sample questions from 'Questions' sheet

    'source' may be:
        - file path
        - BytesIO object (uploaded file in Streamlit)
    """
    xls = pd.ExcelFile(source)
    sheet_names = xls.sheet_names

    # MAIN DATA SHEET
    if "Data" not in sheet_names:
        raise ValueError("Excel must contain a sheet named 'Data'.")

    df_data = pd.read_excel(xls, "Data")

    # SYSTEM PROMPT SHEET
    system_text = ""
    if "System Prompt" in sheet_names:
        df_system = pd.read_excel(xls, "System Prompt")
        first_col = df_system.columns[0]
        system_lines = (
            df_system[first_col]
            .dropna()
            .astype(str)
            .tolist()
        )
        system_text = "\n".join(system_lines)
    else:
        system_text = "No system prompt found. Dataset columns describe a ticketing system."

    # SAMPLE QUESTIONS SHEET
    sample_questions: List[str] = []
    if "Questions" in sheet_names:
        df_questions = pd.read_excel(xls, "Questions")
        first_col = df_questions.columns[0]
        sample_questions = (
            df_questions[first_col]
            .dropna()
            .astype(str)
            .tolist()
        )

    return df_data, system_text, sample_questions


def build_agent(
    df: pd.DataFrame,
    system_text: str,
    model_name: str = "gpt-4o-mini",
):
    """
    Build a LangChain Pandas DataFrame agent.
    This allows the LLM to run Pandas code over df to answer questions.
    """
    llm = ChatOpenAI(model=model_name, temperature=0)

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        handle_parsing_errors=True,
    )

    return agent


def ask_question(agent, question: str, system_text: str) -> str:
    """
    Wrapper for the agent with added system context.
    Ensures answers come directly from the DataFrame.
    """
    prompt = (
        "You are a myBasePay analytics assistant answering questions "
        "about a call center ticket dataset stored in a Pandas DataFrame named `df`.\n\n"
        "Column meanings and definitions:\n"
        f"{system_text}\n\n"
        "Rules:\n"
        "- Use ONLY the data in the DataFrame.\n"
        "- Perform real calculations using pandas.\n"
        "- Do NOT hallucinate or guess.\n"
        "- If the answer cannot be determined from the data, respond exactly: 'I don't know'.\n\n"
        f"User question: {question}"
    )

    result = agent.run(prompt)
    return result
