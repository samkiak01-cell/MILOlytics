# logic/agent.py

from pathlib import Path
from typing import Tuple, List, Union, IO
import pandas as pd
from langchain_openai import ChatOpenAI


# ==========================================
# LOAD EXCEL
# ==========================================

def load_excel(
    source: Union[str, Path, IO[bytes]]
) -> Tuple[pd.DataFrame, str, List[str]]:

    xls = pd.ExcelFile(source)
    sheet_names = xls.sheet_names

    if "Data" not in sheet_names:
        raise ValueError("Excel must contain a sheet called 'Data'.")

    df_data = pd.read_excel(xls, "Data")

    # Load system prompt explanation
    system_text = ""
    if "System Prompt" in sheet_names:
        df_system = pd.read_excel(xls, "System Prompt")
        col = df_system.columns[0]
        system_text = "\n".join(
            df_system[col].dropna().astype(str).tolist()
        )
    else:
        system_text = "Dataset describing myBasePay call center tickets."

    # Load sample questions
    sample_questions = []
    if "Questions" in sheet_names:
        df_questions = pd.read_excel(xls, "Questions")
        col = df_questions.columns[0]
        sample_questions = df_questions[col].dropna().astype(str).tolist()

    return df_data, system_text, sample_questions


# ==========================================
# ZERO-AGENT EXECUTION SYSTEM
# ==========================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


PYTHON_INSTRUCTIONS = """
You are a Python data analyst.

You will be given:
- A Pandas DataFrame called df
- A natural language question

Your job is to write **ONLY Python code**, no explanation.
The code MUST assign the final answer to a variable named `result`.

Rules:
- Use only df and Pandas.
- Do not print anything.
- Do not import anything.
- Do not loop infinitely.
- Always assign the final computed answer to `result`.
- If the answer cannot be determined, set: result = "I don't know"
"""


def generate_code(df: pd.DataFrame, question: str, system_text: str) -> str:
    """
    LLM generates Python code to answer the question using df.
    """

    prompt = f"""
{PYTHON_INSTRUCTIONS}

Column meanings:
{system_text}

DataFrame columns:
{list(df.columns)}

Question:
{question}

Write Python code that computes the answer and stores it in `result`.
"""

    code = llm.invoke(prompt).content.strip()

    # Remove code fencing if present
    code = code.replace("```python", "").replace("```", "").strip()

    return code


def execute_code(df: pd.DataFrame, code: str):
    """
    Executes model-generated Python safely with df in scope.
    """

    local_vars = {"df": df, "result": None}

    try:
        exec(code, {}, local_vars)
        return local_vars.get("result", "I don't know")
    except Exception as e:
        return f"Error executing code: {e}"


def ask_question(agent_unused, question: str, system_text: str, df=None) -> str:
    """
    Agent wrapper for Streamlit (agent param kept for compatibility).
    """

    if df is None:
        return "Dataset not loaded."

    generated = generate_code(df, question, system_text)
    answer = execute_code(df, generated)
    return answer


def build_agent(df, system_text):
    """
    No agent needed. Just return df since ask_question() now handles execution.
    """
    return df
