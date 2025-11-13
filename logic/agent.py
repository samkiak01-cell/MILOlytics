# logic/agent.py

from pathlib import Path
from typing import Tuple, List, Union, IO, Any
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
        raise ValueError("Excel must contain a sheet named 'Data'.")

    df_data = pd.read_excel(xls, "Data")

    # System Prompt sheet
    if "System Prompt" in sheet_names:
        df_system = pd.read_excel(xls, "System Prompt")
        col = df_system.columns[0]
        system_text = "\n".join(df_system[col].dropna().astype(str).tolist())
    else:
        system_text = ""

    # Sample questions
    if "Questions" in sheet_names:
        df_questions = pd.read_excel(xls, "Questions")
        col = df_questions.columns[0]
        sample_questions = df_questions[col].dropna().astype(str).tolist()
    else:
        sample_questions = []

    return df_data, system_text, sample_questions


# ===================================================
# LLM
# ===================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===================================================
# CODE GENERATION PROMPT
# ===================================================

CODE_PROMPT = """
You are a senior Python data analyst.

You MUST output **ONLY Python code** that uses the Pandas DataFrame `df`
to compute the answer to the user's question.

Rules:
- The DataFrame variable available is `df`.
- NEVER print anything.
- NEVER import anything.
- NEVER create visualizations.
- You MUST store the final computed answer in a variable named `result`.
- The result may be:
    - a number
    - a string
    - a list of items
    - a Pandas Series
    - a Pandas DataFrame
- DO NOT write comments or narrative text.
- NO markdown. ONLY executable Python code.

If the answer cannot be computed, set:
    result = "I don't know"
"""


def generate_code(question: str, df: pd.DataFrame, system_text: str) -> str:

    prompt = f"""
{CODE_PROMPT}

Column descriptions:
{system_text}

DataFrame columns:
{list(df.columns)}

User question:
"{question}"

Write Python code ONLY. No explanation. No backticks.
"""

    code = llm.invoke(prompt).content.strip()
    return code.replace("```", "").replace("python", "").strip()


# ===================================================
# EXECUTE THE GENERATED CODE
# ===================================================

def execute_code(df: pd.DataFrame, code: str) -> Any:
    scope = {"df": df.copy(), "result": None}
    try:
        exec(code, {}, scope)
        return scope.get("result", None)
    except Exception as e:
        return f"Error executing code: {e}"


# ===================================================
# NATURAL LANGUAGE EXPLANATION LAYER
# ===================================================

EXPLANATION_PROMPT = """
You are a data analyst. Your job is to explain the result of a Pandas computation
in clear, natural language that a manager or executive could understand.

Guidelines:
- DO NOT hallucinate values not present in the result.
- If the result is a Series or dict-like object, summarize key items.
- If result is a list of names, describe what they represent.
- If the question asks "who", "which", or "what", give a clear, direct answer first.
- ALWAYS give at least 2–3 sentences.
- ALWAYS include numeric counts, averages, or comparisons if present.
- The explanation MUST be human, polished, and easy to read.
- NO one-word explanations.
- NO robotic phrasing.

Write the explanation as if you are summarizing insights to a team leader.

User question:
{question}

Computed result:
{result}

Column meanings:
{system_text}

Provide a natural language explanation:
"""

def explain_result(question: str, result: Any, system_text: str) -> str:
    prompt = EXPLANATION_PROMPT.format(
        question=question,
        result=str(result),
        system_text=system_text
    )
    return llm.invoke(prompt).content.strip()


# ===================================================
# MAIN ENTRYPOINT
# ===================================================

def ask_question(agent_unused, question: str, system_text: str, df=None):

    if df is None:
        return "Dataset not loaded."

    # STEP 1 — Generate code
    code = generate_code(question, df, system_text)

    # STEP 2 — Execute code
    result = execute_code(df, code)

    # STEP 3 — Explain result naturally
    explanation = explain_result(question, result, system_text)

    # FINAL OUTPUT
    return f"**Answer:** {result}\n\n**Explanation:**\n{explanation}"


def build_agent(df, system_text):
    return df
