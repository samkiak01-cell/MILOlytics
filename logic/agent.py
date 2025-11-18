# logic/agent.py

from __future__ import annotations
from typing import Any, List, Tuple, Union, IO
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI


# ============================================================
# LOAD EXCEL (Generalized)
# ============================================================

def load_excel(
    source: Union[str, Path, IO[bytes]]
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Load any Excel file with:
      - a sheet named "Data"   (required)
      - optional sheet "System Prompt"
      - optional sheet "Questions"

    Returns:
      df, system_text, sample_questions
    """
    xls = pd.ExcelFile(source)
    sheets = xls.sheet_names

    if "Data" not in sheets:
        raise ValueError("Excel must contain a sheet named 'Data'.")

    df = pd.read_excel(xls, "Data")

    # Load system prompt text if exists
    if "System Prompt" in sheets:
        df_sys = pd.read_excel(xls, "System Prompt")
        col = df_sys.columns[0]
        system_text = "\n".join(df_sys[col].dropna().astype(str).tolist())
    else:
        system_text = ""

    # Load sample questions if exists
    if "Questions" in sheets:
        df_q = pd.read_excel(xls, "Questions")
        col = df_q.columns[0]
        sample_questions = df_q[col].dropna().astype(str).tolist()
    else:
        sample_questions = []

    return df, system_text, sample_questions



# ============================================================
# GLOBAL LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)



# ============================================================
# COLUMN SEMANTICS (Generalized for Demo Data)
# ============================================================

COLUMN_SEMANTICS = """
This dataset represents call center tickets.

Columns and meanings:
- Ticket_Number: ID of the ticket
- Caller: the person who called in
- Subject: high-level issue category (use this as-is; do NOT invent new categories)
- Description: free-text explanation of the issue (used only for context, not grouping)
- Created_Date_Time: timestamp when ticket was created
- Resolved_Date_Time: timestamp when it was resolved (if resolved)
- Resolution_Time_Seconds: time difference between creation and resolution
- Resolved_By: call center agent who resolved the ticket
- SLA_Met: whether the ticket was resolved before the SLA deadline (True/False)

Rules:
- “Who submits the most tickets” → group by Caller
- “Who resolves fastest / slowest” → use Resolved_By + average Resolution_Time_Seconds
- “Which tickets were resolved outside SLA” → SLA_Met == False
- “Longest/shortest ticket” → use Resolution_Time_Seconds
- “Most common type of issue” → group by Subject (DO NOT create or rename subjects)
- Outliers → choose the top or bottom 1–3 items that clearly stand apart
"""



# ============================================================
# ANALYTICAL CODE GENERATION (Layer 1)
# ============================================================

CODE_PROMPT = """
You are a senior Python analyst.

You MUST output ONLY Python code using the Pandas DataFrame `df`
to compute the answer.

The code MUST produce a final variable named `result`, a Python dict:
    result = {
        "answer_core": <string>,
        "metrics": <JSON-safe dict>,
        "mode": <string tag>
    }

RULES:
- Use ONLY DataFrame operations (no prints, no imports, no plotting).
- Result MUST be JSON serializable: convert all numpy types to Python types.
- Detect ties automatically (return all top performers if tied).
- Outliers: return top/bottom 1–3 items.
- SLA logic: sla_breaches = df[df["SLA_Met"] == False].

If unable to compute the answer:
    result = {
       "answer_core": "I don't know",
       "metrics": {"reason": "..."},
       "mode": "error"
    }
"""


def generate_code(question: str, df: pd.DataFrame, system_text: str) -> str:
    cols = list(df.columns)

    prompt = f"""
{CODE_PROMPT}

Column descriptions:
{COLUMN_SEMANTICS}

Additional system text (if any):
{system_text}

DataFrame columns:
{cols}

User question:
{question}

Write ONLY Python code (no comments, no markdown).
"""

    code = llm.invoke(prompt).content.strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return code



# ============================================================
# CODE EXECUTION ENGINE (Layer 1)
# ============================================================

def execute_code(df: pd.DataFrame, code: str) -> dict:
    local_vars = {"df": df.copy(), "result": None}

    try:
        exec(code, {}, local_vars)
        result = local_vars.get("result", None)

        # Ensure dict format
        if not isinstance(result, dict):
            return {
                "answer_core": "I don't know",
                "metrics": {"reason": "Generated code did not return a dict"},
                "mode": "error"
            }

        return result

    except Exception as e:
        return {
            "answer_core": "I don't know",
            "metrics": {"reason": f"Execution error: {e}"},
            "mode": "error"
        }



# ============================================================
# EXPLANATION ENGINE (Layer 2)
# ============================================================

EXPLANATION_PROMPT = """
You are a data analyst explaining results to an executive.

You will be given:
- The user’s question
- A structured result dict with:
    answer_core: short answer
    metrics: JSON-safe dict of numeric information
    mode: type of analysis
- Column semantics

Write a FINAL ANSWER that contains two parts:

1) A **headline answer** (1 sentence)
2) A **short explanation** (2–4 sentences max) that:
   - references metrics correctly
   - converts seconds to hours/days when helpful
   - mentions comparisons (faster/slower, highest/lowest, etc.)
   - is business-friendly and clear

DO NOT invent data.
DO NOT hallucinate subject categories.
Use ONLY the metrics given.

Format exactly as:

Answer:
<one-sentence headline>

Explanation:
<2–4 sentence explanation>

"""

def build_final_answer(question: str, result: dict, system_text: str) -> str:

    prompt = EXPLANATION_PROMPT + f"""

User question:
{question}

Result dict:
{result}

Column semantics:
{COLUMN_SEMANTICS}

Your response:
"""

    explanation = llm.invoke(prompt).content.strip()
    return explanation



# ============================================================
# PUBLIC API FOR STREAMLIT
# ============================================================

def ask_question(agent_unused, question: str, system_text: str, df=None) -> str:
    if df is None:
        return "Dataset not loaded."

    # Layer 1 — generate & run analytic code
    code = generate_code(question, df, system_text)
    result = execute_code(df, code)

    # Layer 2 — build final human-readable answer
    final_answer = build_final_answer(question, result, system_text)
    return final_answer



def build_agent(df, system_text):
    """No heavy agent object required — df is the agent."""
    return df
