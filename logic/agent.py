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
    """
    Load the Excel file and return:
      - df: main ticket dataset from 'Data' sheet
      - system_text: text from 'System Prompt' (if present)
      - sample_questions: from 'Questions' sheet (if present)
    """

    xls = pd.ExcelFile(source)
    sheets = xls.sheet_names

    if "Data" not in sheets:
        raise ValueError("Excel must contain a sheet named 'Data'.")

    df = pd.read_excel(xls, "Data")

    # System Prompt sheet
    if "System Prompt" in sheets:
        df_sys = pd.read_excel(xls, "System Prompt")
        col = df_sys.columns[0]
        system_text = "\n".join(df_sys[col].dropna().astype(str).tolist())
    else:
        system_text = ""

    # Questions sheet
    if "Questions" in sheets:
        df_q = pd.read_excel(xls, "Questions")
        col = df_q.columns[0]
        sample_questions = df_q[col].dropna().astype(str).tolist()
    else:
        sample_questions = []

    return df, system_text, sample_questions


# ===================================================
# LLM
# ===================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===================================================
# COLUMN SEMANTICS
# ===================================================

COLUMN_SEMANTICS = """
We have a call center where agents log issues as Jira support tickets.

Columns and meanings:
- Ticket_Number: ID of the Jira ticket
- Description: summary of the issue
- Caller: person who called in and submitted the ticket
- Priority: urgency level of the issue
- Status: state of the ticket (resolved, in progress, newly submitted)
- Support_Team: group that is handling the ticket
- Created_Date_Time: timestamp when the ticket was created (issue received)
- Created_By: call center team member who logged the ticket
- Updated_Date_Time: timestamp when the ticket was last updated
- Updated_By: call center team member who last updated the ticket
- Due_Date_Time: SLA deadline when the ticket should be resolved
- Resolution_Date_Time: timestamp when the issue was resolved
- Resolved_By: call center team member who resolved the ticket
- Resolution_Time_Seconds: number of seconds between Created_Date_Time and Resolution_Date_Time

Important semantic rules:
- "Who submits the most tickets?" or "who called in the most?" -> group by Caller.
- "Which call center member handles/resolves tickets fastest/slowest?" -> group by Created_By or Resolved_By using Resolution_Time_Seconds.
- SLA questions -> compare Resolution_Date_Time (or Resolution_Time_Seconds) with Due_Date_Time.
- For questions about "outliers", "unusual", "anomalies", "stands out", treat these as items that are clearly at the extreme (top or bottom) compared to the rest.
"""


# ===================================================
# CODE GENERATION PROMPT
# ===================================================

CODE_PROMPT = """
You are a senior Python data analyst.

You MUST output ONLY Python code that uses the Pandas DataFrame `df`
to answer the user's question about the ticket dataset.

The DataFrame is named `df` and has the columns described below.
Use the semantic rules correctly (Caller submits, Created_By/Resolved_By handle, SLA is based on Due_Date_Time vs Resolution).

Your job:
1. Analyze the question and decide what needs to be computed.
2. Use Pandas operations on `df` to compute the answer.
3. Construct a Python dict named result with at least these keys:

   result = {
       "answer": <short human-readable answer string>,
       "details": <structured data that includes any key values you used>,
       "mode": <optional string describing the type of analysis, e.g. "max", "min", "outlier", "count", "trend">
   }

   - result["answer"]:
       * A concise string that directly answers the question.
       * For multiple answers (ties), combine names in a human way,
         e.g. "Abraham Lincoln and Olivia Johnson" or
         "INC0010109, INC0010110 and INC0010200".
   - result["details"]:
       * A dict or list of dicts with the underlying numeric values.
       * Always include numeric metrics you used (e.g. counts, seconds, percentages).
       * Example for longest ticket:
         {
           "tickets": [
               {"Ticket_Number": "INC0010109", "Resolution_Time_Seconds": 259200.0}
           ]
         }
       * Example for callers:
         {
           "callers": [
               {"Caller": "Abraham Lincoln", "count": 3},
               {"Caller": "Olivia Johnson", "count": 3}
           ],
           "total_tickets": 40
         }
   - result["mode"]:
       * A short tag like "max", "min", "outlier", "count", "share", "sla", etc.
       * This helps the explanation reason about the type of answer.

VERY IMPORTANT:
- ALWAYS detect ties for max/min style questions.
  If multiple tickets, callers, or agents share the same extreme value,
  include ALL of them in result["answer"] and in result["details"].
- For anomaly / unusual / outlier style questions:
  * Consider items that are clearly at the top or bottom relative to others.
  * Return the 1–3 most extreme items.
  * Let the data distribution guide you; if several items are very similar,
    it is fine to mention up to 3 of them.
- DO NOT print anything.
- DO NOT import anything.
- DO NOT create plots.
- DO NOT write comments.
- Do NOT write explanations or text outside of Python code.
- Only executable Python code that ends with a variable named result.

If the answer cannot be computed from the available columns, set:

result = {
    "answer": "I don't know",
    "details": {"reason": "explain briefly why this cannot be computed"},
    "mode": "error"
}
"""


def generate_code(question: str, df: pd.DataFrame, system_text: str) -> str:
    """
    Ask the LLM to write Python code that:
    - Uses df
    - Computes an answer
    - Builds result = {"answer": ..., "details": ..., "mode": ...}
    """

    cols = list(df.columns)

    prompt = f"""
{CODE_PROMPT}

Column descriptions from the file (if any):
{system_text}

Additional semantics:
{COLUMN_SEMANTICS}

DataFrame columns:
{cols}

User question:
{question}

Write ONLY Python code. No backticks, no markdown, no comments.
"""

    code = llm.invoke(prompt).content.strip()
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
        return local_scope.get("result", None)
    except Exception as e:
        return {
            "answer": "I don't know",
            "details": {"reason": f"Error executing generated code: {e}"},
            "mode": "error"
        }


# ===================================================
# EXPLANATION LAYER
# ===================================================

EXPLANATION_PROMPT = """
You are a data analyst explaining results to business users.

You will be given:
- The user's question
- A structured Python result dict with keys like "answer", "details", "mode"
- Column meanings

Your job:
- Write a SHORT, clean explanation in **1–3 sentences** max.
- Use ONLY the information found in the result dict.
- Include key numeric values in plain language:
  * counts ("3 tickets", "5 callers")
  * time expressed from seconds into simple units:
    - if you see fields like Resolution_Time_Seconds or any key containing "seconds",
      convert the values to a friendly description: e.g.
      259200 seconds -> "259,200 seconds (72 hours / 3 days)"
      7200 seconds -> "7,200 seconds (2 hours)"
- Avoid any statistical jargon:
  * Do NOT mention standard deviation, variance, distributions, etc.
  * Use phrases like "much longer than most tickets", "unusually slow compared to typical tickets",
    "higher than what we usually see", "more often than other callers", etc.
- For multiple items with the same extreme value (ties), mention that they are tied and give their numbers.
- For anomaly / outlier / unusual questions:
  * Focus on which items stand out and by how much in simple terms.
- Keep it business-friendly, direct, and easy to read.
- NO fluff. NO long paragraphs.

User question:
{question}

Result dict:
{result}

Column meanings:
{system_text}

Write a concise explanation:
"""


def explain_result(question: str, result: Any, system_text: str) -> str:
    prompt = EXPLANATION_PROMPT.format(
        question=question,
        result=str(result),
        system_text=system_text
    )
    explanation = llm.invoke(prompt).content.strip()
    return explanation


# ===================================================
# MAIN ENTRYPOINT FOR STREAMLIT
# ===================================================

def ask_question(agent_unused, question: str, system_text: str, df=None) -> str:
    """
    Main function used by the app.

    Returns a formatted string:

    Answer:
    <answer>

    Explanation:
    <short explanation>
    """
    if df is None:
        return "Dataset not loaded."

    # STEP 1 — LLM generates analysis code
    code = generate_code(question, df, system_text)

    # STEP 2 — Execute the code on the actual df
    result = execute_code(df, code)

    # STEP 3 — Normalize result
    if not isinstance(result, dict):
        result = {
            "answer": str(result),
            "details": {"note": "Result was not a dict; converted to string."},
            "mode": "raw"
        }

    answer_text = str(result.get("answer", "I don't know")).strip()
    details = result.get("details", {})
    mode = str(result.get("mode", "")).strip()

    # STEP 4 — Generate a short, business-friendly explanation
    explanation_text = explain_result(question, result, system_text).strip()
    if not explanation_text:
        explanation_text = "This answer was computed from the dataset, but no further explanation was provided."

    return f"Answer:\n{answer_text}\n\nExplanation:\n{explanation_text}"


def build_agent(df, system_text):
    """
    Kept for compatibility with the existing app structure.
    We don't need a complex agent object; df itself is enough.
    """
    return df
