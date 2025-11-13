# logic/agent.py

from pathlib import Path
from typing import Tuple, List, Union, IO

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


# ==========================================
# LOAD EXCEL
# ==========================================

def load_excel(
    source: Union[str, Path, IO[bytes]]
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Loads the Excel dataset files used by the myBasePay ticket assistant.

    Expected sheets:
      - "Data" (main dataset)
      - "System Prompt" (text explanations of each column)
      - "Questions" (sample user questions)

    Returns:
      df_data: main DataFrame
      system_text: combined descriptive text
      sample_questions: list of sample questions
    """

    xls = pd.ExcelFile(source)
    sheet_names = xls.sheet_names

    # ---- Main dataset ----
    if "Data" not in sheet_names:
        raise ValueError("Excel must contain a sheet named 'Data'.")

    df_data = pd.read_excel(xls, "Data")

    # ---- System prompt ----
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
        system_text = (
            "System Prompt sheet not found. Columns describe a call center ticketing dataset."
        )

    # ---- Sample questions ----
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


# ==========================================
# BUILD AGENT
# ==========================================

def build_agent(
    df: pd.DataFrame,
    system_text: str,
    model_name: str = "gpt-4o-mini",
):
    """
    Build a safe LangChain Pandas DataFrame agent.

    Fixes:
      - Converts datetime columns to strings (prevents tokenization errors)
      - Enables allow_dangerous_code=True for real Pandas operations
    """

    # Convert datetime columns to string to avoid LangChain crashes
    df = df.copy()
    datetime_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime", "datetimetz"]
    ).columns

    for col in datetime_cols:
        df[col] = df[col].astype(str)

    # Initialize LLM
    llm = ChatOpenAI(model=model_name, temperature=0)

    # Create the Pandas agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        handle_parsing_errors=True,
        allow_dangerous_code=True,  # REQUIRED to avoid ValueError
    )

    return agent


# ==========================================
# ASK QUESTION
# ==========================================

def ask_question(agent, question: str, system_text: str) -> str:
    """
    Wrap question with system prompt + instructions.
    Ensures:
      - No hallucinations
      - Answers come from DataFrame
      - Fallback to "I don't know" if not found
    """

    prompt = (
        "You are a myBasePay analytics assistant. "
        "You analyze a call center ticketing dataset stored in a Pandas DataFrame named `df`.\n\n"
        "Column meanings:\n"
        f"{system_text}\n\n"
        "Rules:\n"
        "- Use ONLY the data in the DataFrame.\n"
        "- Perform real calculations using Pandas.\n"
        "- Absolutely DO NOT guess or hallucinate.\n"
        "- If the answer cannot be determined from the dataset, respond exactly: 'I don't know'.\n\n"
        f"User question: {question}"
    )

    result = agent.run(prompt)
    return result
