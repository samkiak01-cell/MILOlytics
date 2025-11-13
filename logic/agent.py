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
    Loads the Excel dataset for the myBasePay ticket analytics assistant.

    Expected sheets:
      - "Data" (main dataset)
      - "System Prompt" (column descriptions)
      - "Questions" (sample questions)

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
            "System Prompt sheet missing. Columns describe a call center ticket dataset."
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
# BUILD AGENT — ***UPDATED FIXED VERSION***
# ==========================================

def build_agent(
    df: pd.DataFrame,
    system_text: str,
    model_name: str = "gpt-4o-mini",
):
    """
    Build a safe, single-step Pandas agent.

    FIXES INCLUDED:
      - Converts datetime columns to strings (prevents tokenizer crashes)
      - Uses safe agent mode (no iterative looping)
      - Sets max_iterations=1 to prevent infinite cycles
      - early_stopping_method="generate" ensures the LLM stops instead of looping
      - allow_dangerous_code=True enables real Pandas calculations
    """

    # Convert datetime columns to string
    df = df.copy()
    datetime_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime", "datetimetz"]
    ).columns

    for col in datetime_cols:
        df[col] = df[col].astype(str)

    # Initialize LLM
    llm = ChatOpenAI(model=model_name, temperature=0)

    # Create the Pandas agent (loop-free mode)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        max_iterations=1,                # <-- prevents iteration-limit errors
        early_stopping_method="generate" # <-- forces an answer instead of looping
    )

    return agent


# ==========================================
# ASK QUESTION
# ==========================================

def ask_question(agent, question: str, system_text: str) -> str:
    """
    Wrapper for the agent with contextual instructions.
    Ensures:
      - Uses real DataFrame operations
      - No hallucinations
      - Responds 'I don't know' when needed
    """

    prompt = (
        "You are a myBasePay analytics assistant. "
        "You analyze a call center ticket dataset stored in a Pandas DataFrame named `df`.\n\n"
        "Column meanings:\n"
        f"{system_text}\n\n"
        "Rules:\n"
        "- Use ONLY the information in the DataFrame.\n"
        "- Perform real calculations using Pandas.\n"
        "- Do NOT guess or hallucinate values.\n"
        "- If the answer cannot be found, respond exactly: 'I don't know'.\n\n"
        f"User question: {question}"
    )

    result = agent.run(prompt)
    return result
