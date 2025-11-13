# logic/agent.py

from pathlib import Path
from typing import Tuple, List, Union, IO, Any
import pandas as pd
from langchain_openai import ChatOpenAI


# ===================================================
# LOAD EXCEL (same contract as before)
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
# COLUMN SEMANTICS (from your description)
# ===================================================

COLUMN_SEMANTICS = """
We have a call center and each person logs issues they get from calls into Jira as support tickets.
The exported dataset has these columns:

- Ticket_Number: ID of the Jira ticket
- Description: summary of the issue
- Caller: person's name who called in with the issue (person submitting the ticket)
- Priority: urgency level of the issue
- Status: state of the ticket (resolved, in progress, newly submitted)
- Support_Team: name of the group answering the calls and handling the ticket
- Created_Date_Time: date & time the issue was received
- Created_By: name of the call center team member who received the call and entered the ticket
- Updated_Date_Time: date & time the ticket was last updated
- Updated_By: name of the call center team member who last updated the ticket
- Due_Date_Time: expected date & time the ticket should be resolved to meet the SLA
- Resolution_Date_Time: date & time the issue was resolved
- Resolved_By: name of the call center team member who resolved the issue
- Resolution_Time_Seconds: time in seconds between Created_Date_Time and Resolution_Date_Time

Very important interpretation rules:
- "Who submits the most tickets?" or "who called in the most?" -> use the Caller column.
- "Which call center member handles/resolves tickets fastest/slowest?" -> use Resolution_Time_Seconds grouped by Created_By or Resolved_By.
- SLA questions -> compare Resolution_Date_Time (or Resolution_Time_Seconds) with Due_Date_Time.
"""


# ===================================================
# CODE GENERATION PROMPT
# ===================================================

CODE_PROMPT = """
You are a senior Python data analyst.

You MUST output ONLY Python code that uses the Pandas DataFrame `df`
to answer the user's question.

The DataFrame is named `df` and has the columns described below.
You should use these semantics correctly (e.g., Caller submits, Created_By/Resolved_By handle).

Your job:
1. Analyze the question and decide what needs to be computed.
2. Use Pandas operations on `df` to compute the answer.
3. Construct a Python dict named `result` with EXACTLY TWO keys:
   - "answer": a short, human-readable answer string
   - "explanation": 1–2 concise sentences that briefly explain WHY that is the answer,
                   including key numbers (e.g., how many tickets, how many seconds, etc.)

Examples of valid `result` values:

result = {
    "answer": "INC0010109",
    "explanation": "Ticket INC0010109 took 259200 seconds (72 hours) to resolve, the longest in the dataset."
}

result = {
    "answer": "Abraham Lincoln and Olivia Johnson",
    "explanation": "Abraham Lincoln and Olivia Johnson each called in 3 times, the highest number of tickets submitted by any caller."
}

result = {
    "answer": "Overall SLA compliance is 87.5%",
    "explanation": "70 out of 80 tickets were resolved on or before their due date, giving 87.5% of tickets within SLA."
}

Formatting and style rules:
- Keep `answer` short and direct (names, ticket numbers, a percentage, etc.).
- Keep `explanation` to 1–2 sentences max. No long paragraphs.
- Include the key numeric values used in the calculation (counts, percentages, seconds/hours).
- Do NOT print anything.
- Do NOT import anything.
- Do NOT create plots.
- Do NOT write markdown.
- Do NOT write comments.
- Only executable Python code that ends with defining the `result` dict.

If the answer truly cannot be determined from the available columns, set:
result = {
    "answer": "I don't know",
    "explanation": "Explain briefly why the answer cannot be determined from this dataset."
}
"""


def generate_code(question: str, df: pd.DataFrame, system_text: str) -> str:
    """
    Ask the LLM to write Python code that:
    - Uses df
    - Computes an answer
    - Builds `result = {"answer": ..., "explanation": ...}`
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
    # Clean possible fences just in case
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
        # If execution fails, wrap the error as a structured result
        return {
            "answer": "I don't know",
            "explanation": f"Error executing generated code: {e}"
        }


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
    <explanation>
    """
    if df is None:
        return "Dataset not loaded."

    code = generate_code(question, df, system_text)
    result = execute_code(df, code)

    # Defensive fallback handling
    if not isinstance(result, dict):
        answer_text = str(result)
        explanation_text = "This answer was computed from the dataset, but no structured explanation was provided."
    else:
        answer_text = str(result.get("answer", "I don't know"))
        explanation_text = str(result.get("explanation", "")).strip()
        if not explanation_text:
            explanation_text = "This answer was computed from the dataset, but no detailed explanation was provided."

    return f"Answer:\n{answer_text}\n\nExplanation:\n{explanation_text}"


def build_agent(df, system_text):
    """
    Kept for compatibility with the existing app structure.
    We don't need a complex agent object; df itself is enough.
    """
    return df
