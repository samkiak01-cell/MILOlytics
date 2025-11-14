# logic/agent.py

from pathlib import Path
from typing import Tuple, List, Union, IO, Any
import pandas as pd
from langchain_openai import ChatOpenAI



# ===================================================
# GLOBAL LLM
# ===================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)



# ===================================================
# DATA CLEANING
# ===================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize datetime, numeric, and text fields."""

    # Convert dates
    datetime_cols = [
        "Created_Date_Time",
        "Updated_Date_Time",
        "Due_Date_Time",
        "Resolution_Date_Time"
    ]

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert numeric
    if "Resolution_Time_Seconds" in df.columns:
        df["Resolution_Time_Seconds"] = pd.to_numeric(
            df["Resolution_Time_Seconds"],
            errors="coerce"
        )

    # Clean agent / caller fields
    text_cols = ["Caller", "Created_By", "Updated_By", "Resolved_By", "Description"]

    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": None, "None": None, "": None})
            )

    return df



# ===================================================
# ISSUE CATEGORIZATION LAYER (NEW)
# ===================================================

def categorize_issues_with_llm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Issue_Category column using LLM interpretation of Description.
    """

    if "Description" not in df.columns:
        df["Issue_Category"] = "Unknown Issue"
        return df

    categories = []
    descriptions = df["Description"].astype(str).tolist()

    for desc in descriptions:

        prompt = f"""
        You are an issue classification assistant.

        Categorize this support ticket description into ONE short category (1–3 words):

        "{desc}"

        Examples of categories:
        - Login Issue
        - Password Reset
        - Access Problem
        - System Error
        - Performance Issue
        - Data Update
        - Account Setup
        - Notification Issue
        - Workflow Question
        - Configuration Issue

        Rules:
        - ALWAYS return only the category name.
        - NEVER return explanations.
        - If unclear, choose the closest reasonable category.
        """

        try:
            cat = llm.invoke(prompt).content.strip()
        except:
            cat = "Uncategorized"

        # Safety — sanitize weird LLM outputs
        if "\n" in cat:
            cat = cat.split("\n")[0].strip()

        categories.append(cat)

    df["Issue_Category"] = categories
    return df



# ===================================================
# LOAD EXCEL
# ===================================================

def load_excel(
    source: Union[str, Path, IO[bytes]]
) -> Tuple[pd.DataFrame, str, List[str]]:

    xls = pd.ExcelFile(source)
    sheets = xls.sheet_names

    if "Data" not in sheets:
        raise ValueError("Excel must contain a sheet named 'Data'.")

    df = pd.read_excel(xls, "Data")
    df = clean_dataframe(df)
    df = categorize_issues_with_llm(df)  # <— NEW AUTOMATIC CATEGORY COLUMN

    # System Prompt
    if "System Prompt" in sheets:
        df_sys = pd.read_excel(xls, "System Prompt")
        col = df_sys.columns[0]
        system_text = "\n".join(df_sys[col].dropna().astype(str).tolist())
    else:
        system_text = ""

    # Sample Questions
    if "Questions" in sheets:
        df_q = pd.read_excel(xls, "Questions")
        col = df_q.columns[0]
        sample_questions = df_q[col].dropna().astype(str).tolist()
    else:
        sample_questions = []

    return df, system_text, sample_questions



# ===================================================
# COLUMN SEMANTICS
# ===================================================

COLUMN_SEMANTICS = """
We have a call center where agents log issues as Jira support tickets.

Columns and meanings:
- Ticket_Number: ID of the Jira ticket
- Description: summary of the issue
- Caller: person who called in and submitted the ticket
- Priority: urgency level
- Status: ticket state
- Support_Team: group handling the ticket
- Created_Date_Time: when ticket was created
- Created_By: agent who logged the ticket
- Updated_Date_Time: last update time
- Updated_By: agent who last updated the ticket
- Due_Date_Time: SLA deadline
- Resolution_Date_Time: when issue was resolved
- Resolved_By: agent who resolved the ticket
- Resolution_Time_Seconds: time between creation & resolution
- Issue_Category: LLM-interpreted category of the description

Semantic rules:
- "Who submits the most tickets?" -> group by Caller.
- "Who handles fastest/slowest?" -> group by Resolved_By using Resolution_Time_Seconds.
- A ticket is resolved only if both Resolution_Date_Time and Resolution_Time_Seconds exist.
- Issue analysis -> group by Issue_Category.
"""



# ===================================================
# LLM CODE GENERATION PROMPT
# ===================================================

CODE_PROMPT = """
You are a senior Python data analyst.

You MUST output ONLY Python code that uses the Pandas DataFrame `df`
to answer the user's question.

The DataFrame is named `df`.

Rules:
1. You must ALWAYS produce a `result = {...}` dictionary with keys:
   - answer: short human-readable answer
   - details: dict or list with the data used
   - mode: "max", "min", "count", "outlier", "sla", etc.

2. Use correct column meanings from the semantic rules.

3. For fastest/slowest handling:
   df2 = df[df["Resolution_Time_Seconds"].notna() & df["Resolution_Date_Time"].notna()]
   group by Resolved_By.

4. For issue types:
   ALWAYS use Issue_Category (already created by the system).

5. ALWAYS detect ties.

6. DO NOT print anything.
7. DO NOT return explanations.
8. DO NOT write comments.
9. ONLY return Python code that sets `result`.

If the answer cannot be computed, return:

result = {
    "answer": "I don't know",
    "details": {"reason": "explain the missing data"},
    "mode": "error"
}
"""



# ===================================================
# GENERATE CODE
# ===================================================

def generate_code(question: str, df: pd.DataFrame, system_text: str) -> str:
    cols = list(df.columns)

    prompt = f"""
{CODE_PROMPT}

Column descriptions:
{system_text}

Additional semantics:
{COLUMN_SEMANTICS}

DataFrame columns:
{cols}

User question:
{question}

Write ONLY executable Python code.
"""

    code = llm.invoke(prompt).content.strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return code



# ===================================================
# EXECUTE GENERATED CODE
# ===================================================

def execute_code(df: pd.DataFrame, code: str) -> Any:
    local_scope = {"df": df.copy(), "result": None}

    try:
        exec(code, {}, local_scope)
        return local_scope.get("result", None)

    except Exception as e:
        return {
            "answer": "I don't know",
            "details": {"reason": f"Error executing code: {e}", "code": code},
            "mode": "error"
        }



# ===================================================
# LLM EXPLANATION LAYER
# ===================================================

EXPLANATION_PROMPT = """
You are a senior business analyst.

You will be given a result dict containing:
- answer
- details
- mode

Your job:
- Write a SHORT explanation (1–3 sentences max)
- Include numeric values (counts, seconds, hours)
- If there are ties, mention it
- If seconds appear, convert them into human units (hours, days)
- Keep it simple, business-friendly, and direct

User question:
{question}

Result:
{result}

Column meanings:
{system_text}

Write the explanation:
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

def ask_question(agent_unused, question: str, system_text: str, df=None) -> str:

    if df is None:
        return "Dataset not loaded."

    code = generate_code(question, df, system_text)
    result = execute_code(df, code)

    if not isinstance(result, dict):
        result = {
            "answer": str(result),
            "details": {"note": "Non-dict result"},
            "mode": "raw"
        }

    answer_text = str(result.get("answer", "I don't know"))
    explanation_text = explain_result(question, result, system_text)

    return f"Answer:\n{answer_text}\n\nExplanation:\n{explanation_text}"


def build_agent(df, system_text):
    return df
