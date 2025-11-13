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
    sheets = xls.sheet_names

    if "Data" not in sheets:
        raise ValueError("Excel must contain sheet 'Data'.")

    df = pd.read_excel(xls, "Data")

    if "System Prompt" in sheets:
        df_sys = pd.read_excel(xls, "System Prompt")
        col = df_sys.columns[0]
        system_text = "\n".join(df_sys[col].dropna().astype(str).tolist())
    else:
        system_text = ""

    if "Questions" in sheets:
        df_q = pd.read_excel(xls, "Questions")
        col = df_q.columns[0]
        questions = df_q[col].dropna().astype(str).tolist()
    else:
        questions = []

    return df, system_text, questions


# ===================================================
# MODEL
# ===================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===================================================
# CODE GENERATION — UPDATED SPEC
# ===================================================

CODE_PROMPT = """
You are a senior Python data analyst.

You MUST output ONLY Python code that uses the Pandas DataFrame `df`
to compute the answer to the user's question.

VERY IMPORTANT:
You MUST return the answer as a structured Python object (dict or list of dicts)
that contains BOTH:
- the identifier(s)
- all relevant numeric values used in the comparison

Examples of valid outputs:
result = {"ticket": "INC001234", "duration_seconds": 14820}
result = [{"caller": "John", "count": 5}, {"caller": "Mary", "count": 5}]

DO NOT return a single string unless the question truly requires it.

Rules:
- The DataFrame variable available is df
- NEVER print anything
- NEVER import anything
- NEVER use visualization
- You MUST assign your final answer into a variable named result
- DO NOT write explanations
- DO NOT write markdown
- ONLY Python code

If the answer cannot be computed:
    result = {"error": "I don't know"}
"""


def generate_code(question: str, df: pd.DataFrame, system_text: str) -> str:

    prompt = f"""
{CODE_PROMPT}

Column descriptions:
{system_text}

DataFrame columns:
{list(df.columns)}

User question:
{question}

Write only Python code. No backticks.
"""

    code = llm.invoke(prompt).content.strip()
    return code.replace("```", "").replace("python", "").strip()


# ===================================================
# EXECUTE
# ===================================================

def execute_code(df: pd.DataFrame, code: str) -> Any:
    scope = {"df": df.copy(), "result": None}
    try:
        exec(code, {}, scope)
        return scope["result"]
    except Exception as e:
        return {"error": f"Execution error: {e}"}


# ===================================================
# EXPLANATION — SHORT + FACTUAL
# ===================================================

EXPLANATION_PROMPT = """
You are a data analyst.

You will be given:
- The user's question
- A structured Python result (dict or list of dicts)
- Column meanings

Your task:
Write a SHORT, clean explanation (1–2 sentences MAX).
Use ONLY the numbers and values found in the result.
Never guess. Never write fluff. Never add unrelated commentary.

Format should ALWAYS be:
"<identifier> had <value> <metric> ..."

User question:
{question}

Result:
{result}

Column meanings:
{system_text}

Write a concise factual explanation:
"""


def explain_result(question: str, result: Any, system_text: str) -> str:
    prompt = EXPLANATION_PROMPT.format(
        question=question,
        result=str(result),
        system_text=system_text,
    )
    return llm.invoke(prompt).content.strip()


# ===================================================
# MAIN API
# ===================================================

def ask_question(agent_unused, question: str, system_text: str, df=None):

    if df is None:
        return "Dataset not loaded."

    code = generate_code(question, df, system_text)
    result = execute_code(df, code)
    explanation = explain_result(question, result, system_text)

    return f"**Answer:** {result}\n\n**Explanation:** {explanation}"


def build_agent(df, system_text):
    return df
