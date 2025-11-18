from pathlib import Path
from typing import Tuple, List, Union, IO, Any

import pandas as pd
from langchain_openai import ChatOpenAI


# ===================================================
# LLM INITIALISATION
# ===================================================

# Single shared LLM instance for this module.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===================================================
# EXCEL LOADING
# ===================================================

def load_excel(
    source: Union[str, Path, IO[bytes]]
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Load an Excel file and return:
      - df: main ticket dataset (first sheet or 'Data'/'Sheet1')
      - system_text: optional system prompt text (if 'System Prompt' sheet exists)
      - sample_questions: optional questions (if 'Questions' sheet exists)

    This is schema-aware but not tied to any specific workbook name.
    """
    xls = pd.ExcelFile(source)
    sheets = xls.sheet_names

    # 1) Pick data sheet: prefer "Data" or "Sheet1", else the first sheet.
    data_sheet = None
    for candidate in ["Data", "Sheet1"]:
        if candidate in sheets:
            data_sheet = candidate
            break
    if data_sheet is None:
        data_sheet = sheets[0]

    df = pd.read_excel(xls, data_sheet)

    # 2) Optional system prompt sheet
    system_text = ""
    if "System Prompt" in sheets:
        df_sys = pd.read_excel(xls, "System Prompt")
        col = df_sys.columns[0]
        system_text = "\n".join(df_sys[col].dropna().astype(str).tolist())

    # 3) Optional sample questions sheet
    sample_questions: List[str] = []
    if "Questions" in sheets:
        df_q = pd.read_excel(xls, "Questions")
        col = df_q.columns[0]
        sample_questions = df_q[col].dropna().astype(str).tolist()

    return df, system_text, sample_questions


# ===================================================
# SCHEMA & PROMPTS
# ===================================================

SCHEMA_HINTS = """
You are analysing call-centre support tickets stored in a tabular dataset.

Typical columns and their meanings (if they exist):

- Ticket_Number: unique ID of the ticket.
- Subject: short category or subject line for the issue.
- Description: longer free-text description of the issue.
- Caller: name of the end user who reported the problem.
- Status: current state of the ticket (e.g. Open, In Progress, Resolved).
- Created_Date: when the ticket was created.
- Due_Date: SLA target date/time for resolution.
- Resolved_Date: when the ticket was actually resolved.
- Resolved_By: name of the agent who resolved the ticket.
- Resolution_Time_Seconds: numeric resolution time between creation and resolution.
- SLA_Met: boolean or flag indicating whether the ticket met SLA.

You must infer semantics from column names. For example:
- Questions about “who called in the most” -> group by Caller.
- Questions about “which agent is fastest / slowest on average” -> group by Resolved_By
  using Resolution_Time_Seconds, considering only rows where that value is present.
- SLA questions -> use SLA_Met if present; otherwise compare Resolved_Date > Due_Date.
"""


CODE_PROMPT = """
You are a senior Python data analyst.

You are given a Pandas DataFrame named `df` that contains call-centre tickets.
You must write Python code that uses ONLY this DataFrame and standard Python / pandas
to answer the user’s question.

CRITICAL RULES:
- You MUST define a variable named `result` at the end of your code.
- `result` MUST be a dict with at least these keys:

  result = {
      "answer": <short natural-language answer string>,
      "details": <underlying structured data used for the answer>,
      "mode": <short tag such as "max", "min", "count", "share", "sla", "trend", "outlier", "overview", or "error">
  }

- `result["answer"]`:
    * MUST be a short, business-friendly natural-language answer (1–3 sentences).
    * It should directly answer the user’s question and include key numeric values
      (e.g. counts or times in seconds).
    * If there are ties for “max”/“min” style questions (e.g. multiple agents
      share the slowest average time), include all relevant names in the answer.

- `result["details"]`:
    * SHOULD contain the data you used to compute the answer:
        - lists of dicts for tickets, agents, subjects, etc.
        - numeric metrics like counts, averages, percentages.
    * Use only JSON-serialisable types: dict, list, str, int, float, bool, None.
      Do NOT include pandas objects (Series/DataFrame), timestamps, or numpy types
      in `result["details"]`. Convert them to plain Python types or strings.

- `result["mode"]`:
    * A short string label describing the type of analysis, such as:
      "max", "min", "count", "share", "sla", "trend", "outlier", "overview", or "error".

GENERAL BEHAVIOUR:
- Use the column names in `df.columns` together with the hints you are given
  about typical schema names.
- For questions about “most common subject” or “which subject takes longest”,
  group by a column named like "Subject" (case-sensitive check against df.columns).
- For questions about “who submits the most tickets”, group by a column named like "Caller".
- For questions about “who resolves tickets fastest/slowest”, group by a column
  like "Resolved_By" (or any similar name present in df.columns) and use
  Resolution_Time_Seconds for averages.
- For “outside SLA” or “SLA breaches” questions:
    * Prefer column SLA_Met if it exists.
      - Treat True/"yes"/1 as met, False/"no"/0 as not met.
    * If SLA_Met does not exist, and both Due_Date and Resolved_Date exist,
      treat rows where Resolved_Date > Due_Date as outside SLA.

CONSTRAINTS:
- Do NOT print anything.
- Do NOT import any modules.
- Do NOT create plots.
- Do NOT add comments.
- Do NOT touch or modify global variables.
- Your code must be fully executable as-is and end with a variable named `result`.
- If the question truly cannot be answered from the data, set:

  result = {
      "answer": "I don't know.",
      "details": {"reason": "explain briefly why this cannot be computed from the available columns."},
      "mode": "error"
  }
"""


def generate_code(question: str, df: pd.DataFrame, system_text: str) -> str:
    """
    Ask the LLM to write Python code that:
    - Uses df
    - Computes an answer
    - Populates result = {answer, details, mode}
    """

    cols = list(df.columns)

    prompt = f"""
{CODE_PROMPT}

Additional schema hints:
{SCHEMA_HINTS}

Dataset columns:
{cols}

System prompt text (if any):
{system_text}

User question:
{question}

Write ONLY executable Python code. No backticks, no markdown, no comments.
"""

    code = llm.invoke(prompt).content.strip()
    # Defensive cleanup in case the model adds fences
    code = code.replace("```python", "").replace("```", "").strip()
    return code


# ===================================================
# EXECUTE GENERATED CODE
# ===================================================

def execute_code(df: pd.DataFrame, code: str) -> Any:
    """
    Execute the generated Python code with df in scope.
    Expect a dict named `result` at the end.
    """
    local_scope: dict[str, Any] = {"df": df.copy(), "result": None}
    try:
        exec(code, {}, local_scope)
        result = local_scope.get("result", None)
        return result
    except Exception as e:
        return {
            "answer": "I don't know.",
            "details": {"reason": f"Error executing generated code: {e}"},
            "mode": "error",
        }


# ===================================================
# MAIN ENTRYPOINT USED BY STREAMLIT
# ===================================================

def ask_question(agent_unused, question: str, system_text: str, df=None) -> str:
    """
    Main function used by the Streamlit app.

    Returns a single natural-language answer string that can be rendered directly.
    """
    if df is None:
        return "Dataset not loaded."

    # 1) LLM generates analysis code
    code = generate_code(question, df, system_text)

    # 2) Execute the code on the actual df
    result = execute_code(df, code)

    # 3) Normalise result into a dict
    if not isinstance(result, dict):
        result = {
            "answer": str(result),
            "details": {"note": "Result was not a dict; converted to string."},
            "mode": "raw",
        }

    answer_text = str(result.get("answer", "")).strip()
    if not answer_text:
        answer_text = "I don't know."

    return answer_text


def build_agent(df, system_text):
    """
    Kept for compatibility with the existing app structure.
    We don't need a complex agent object; df itself is enough.
    """
    return df
