# logic/agent.py

from pathlib import Path
from typing import Tuple, List, Union, IO
import pandas as pd
from langchain_openai import ChatOpenAI


# ===================================================
# LOAD EXCEL
# ===================================================

def load_excel(
    source: Union[str, Path, IO[bytes]]
) -> Tuple[pd.DataFrame, str, List[str]]:

    xls = pd.ExcelFile(source)
    sheet_names = xls.sheet_names

    if "Data" not in sheet_names:
        raise ValueError("Excel must contain a sheet called 'Data'.")

    df_data = pd.read_excel(xls, "Data")

    # System prompt / column definitions
    system_text = ""
    if "System Prompt" in sheet_names:
        df_system = pd.read_excel(xls, "System Prompt")
        col = df_system.columns[0]
        system_text = "\n".join(
            df_system[col].dropna().astype(str).tolist()
        )
    else:
        system_text = "Dataset describing myBasePay call center tickets."

    # Sample Questions
    sample_questions = []
    if "Questions" in sheet_names:
        df_questions = pd.read_excel(xls, "Questions")
        col = df_questions.columns[0]
        sample_questions = df_questions[col].dropna().astype(str).tolist()

    return df_data, system_text, sample_questions


# ===================================================
# GLOBAL LLM
# ===================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===================================================
# BASE CODE GENERATION INSTRUCTIONS
# ===================================================

PYTHON_INSTRUCTIONS = """
You are a Python data analyst for myBasePay.

You will receive:
- A Pandas DataFrame called `df`
- A natural language question
- Column definitions and clarifications

You MUST generate **Python code only** that:
1. Computes the correct answer logically.
2. Assigns the final value to a variable named `result`.
3. Does NOT print anything.
4. Does NOT import anything.
5. Does NOT rely on external libraries.
6. NEVER writes explanations — ONLY Python code.

IMPORTANT COLUMN LOGIC:
- The person **submitting** the ticket is ALWAYS `Caller`.
- The person **handling** or **working on** the ticket is `Created_By` or `Updated_By`.
- Resolution time is `Resolution_Time_Seconds`.
- Use `df` exactly as provided.

After computing the result value in Python, assign:
    result = {your computed variable}

If the answer cannot be determined, set:
    result = "I don't know"
"""


# ===================================================
# GENERATE PYTHON CODE
# ===================================================

def generate_code(df: pd.DataFrame, question: str, system_text: str) -> str:
    prompt = f"""
{PYTHON_INSTRUCTIONS}

Column meanings:
{system_text}

DataFrame columns:
{list(df.columns)}

User question:
{question}

Write ONLY Python code that uses Pandas to calculate the answer.
Store the final answer in `result`. No explanation.
"""

    code = llm.invoke(prompt).content.strip()

    # Remove code fences if they appear
    code = code.replace("```python", "").replace("```", "").strip()

    return code


# ===================================================
# EXECUTE PYTHON CODE SAFELY
# ===================================================

def execute_code(df: pd.DataFrame, code: str):
    """
    Safe local execution of generated Python code with df in scope.
    """
    local_vars = {"df": df, "result": None}

    try:
        exec(code, {}, local_vars)
        return local_vars.get("result", "I don't know")
    except Exception as e:
        return f"Error executing code: {e}"


# ===================================================
# EXPLANATION LAYER (NEW!)
# ===================================================

def explain_answer(question: str, result, df: pd.DataFrame, system_text: str):
    """
    Takes the computed result and generates a concise explanation.
    """

    explanation_prompt = f"""
You are an analytics assistant.

You will receive:
- A user's question
- The computed answer (Python result)
- The dataset column meanings

Your task:
- Provide a **short, executive-style explanation**.
- Reference the specific columns used.
- Explain how the answer was computed.
- Include numeric details (averages, counts, seconds, etc.) if relevant.
- Be concise and clear.
- DO NOT hallucinate data that is not in the result.

User question:
{question}

Computed Python result:
{result}

Column meanings:
{system_text}

Write a concise explanation:
"""

    explanation = llm.invoke(explanation_prompt).content.strip()
    return explanation


# ===================================================
# PUBLIC ASK FUNCTION (USED BY STREAMLIT)
# ===================================================

def ask_question(agent_unused, question: str, system_text: str, df=None):
    if df is None:
        return "Dataset not loaded."

    generated_code = generate_code(df, question, system_text)
    result = execute_code(df, generated_code)
    explanation = explain_answer(question, result, df, system_text)

    final_output = f"**Answer:** {result}\n\n**Explanation:** {explanation}"
    return final_output


# ===================================================
# STUB FOR STREAMLIT — NO REAL AGENT NEEDED
# ===================================================

def build_agent(df, system_text):
    return df
