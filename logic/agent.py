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
    sample_questions: List[str] = []
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
# FORMAT HELPERS
# ===================================================

def human_join(names: List[str]) -> str:
    """Turns ['A', 'B', 'C'] into 'A, B & C'."""
    names = [str(n) for n in names]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" & {names[-1]}"


def format_seconds(seconds: float) -> str:
    """Convert seconds into friendly wording."""
    try:
        seconds = float(seconds)
    except Exception:
        return str(seconds)

    if seconds < 60:
        return f"{seconds:.0f} seconds"

    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {sec} sec" if sec else f"{minutes} min"

    hours, mins = divmod(minutes, 60)
    components = []
    if hours:
        components.append(f"{hours} hr")
    if mins:
        components.append(f"{mins} min")
    if sec:
        components.append(f"{sec} sec")
    return " ".join(components)


# ===================================================
# SPECIAL CASE: WHO SUBMITS MOST TICKETS
# ===================================================

def answer_who_submits_most(df: pd.DataFrame) -> str:
    """
    Always uses Caller.
    Handles ties.
    Provides friendly explanation.
    """

    if "Caller" not in df.columns:
        return "**Answer:** I don't know.\n\n**Explanation:** This dataset has no `Caller` column."

    counts = df["Caller"].value_counts()

    if counts.empty:
        return "**Answer:** I don't know.\n\n**Explanation:** No callers found."

    max_count = counts.iloc[0]
    top_callers = counts[counts == max_count].index.tolist()

    name_str = human_join(top_callers)
    total_tickets = len(df)

    explanation = (
        f"{name_str} {'each ' if len(top_callers)>1 else ''}"
        f"submitted **{max_count}** ticket(s), "
        f"which is the highest in the dataset. "
        f"This was calculated by grouping all tickets by the `Caller` column "
        f"and counting how many times each person appeared."
    )

    return f"**Answer:** {name_str}\n\n**Explanation:** {explanation}"


# ===================================================
# SPECIAL CASE: HANDLER SPEED (SLOWEST / FASTEST)
# ===================================================

def answer_handler_speed(df: pd.DataFrame, mode: str = "slowest") -> str:

    time_col = "Resolution_Time_Seconds"
    if time_col not in df.columns:
        return "**Answer:** I don't know.\n\n**Explanation:** Missing column `Resolution_Time_Seconds`."

    handler_col = None
    if "Created_By" in df.columns:
        handler_col = "Created_By"
    elif "Updated_By" in df.columns:
        handler_col = "Updated_By"

    if handler_col is None:
        return "**Answer:** I don't know.\n\n**Explanation:** No handler column found."

    sub = df.dropna(subset=[handler_col, time_col]).copy()
    sub[time_col] = pd.to_numeric(sub[time_col], errors="coerce")
    sub = sub.dropna(subset=[time_col])

    if sub.empty:
        return "**Answer:** I don't know.\n\n**Explanation:** No valid resolution times."

    grouped = sub.groupby(handler_col)[time_col].mean()

    if mode == "slowest":
        best = grouped.idxmax()
        avg_time = grouped.max()
        label = "slowest"
    else:
        best = grouped.idxmin()
        avg_time = grouped.min()
        label = "fastest"

    pretty_time = format_seconds(avg_time)

    explanation = (
        f"{best} has the **{label} average resolution time**, "
        f"taking about **{pretty_time}** per ticket on average. "
        f"This was calculated using the `{time_col}` column grouped by `{handler_col}`."
    )

    return f"**Answer:** {best}\n\n**Explanation:** {explanation}"


# ===================================================
# LLM CODE GENERATION FOR GENERAL QUESTIONS
# ===================================================

PYTHON_INSTRUCTIONS = """
You are a Python data analyst for myBasePay.

You must generate ONLY Python code (no explanation) that:
- Uses the Pandas DataFrame `df`
- Computes an answer to the user's question
- Assigns the answer to a variable named `result`
- Does not import anything
- Does not print anything
- Does not write explanations
"""

def generate_code(df: pd.DataFrame, question: str, system_text: str) -> str:

    prompt = f"""
{PYTHON_INSTRUCTIONS}

Column meanings:
{system_text}

DataFrame columns:
{list(df.columns)}

User question:
{question}

Write only Python code that calculates the correct result.
Store the final answer in the variable `result`.
"""

    code = llm.invoke(prompt).content.strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return code


def execute_code(df: pd.DataFrame, code: str):
    local_vars = {"df": df, "result": None}

    try:
        exec(code, {}, local_vars)
        return local_vars.get("result", "I don't know")
    except Exception as e:
        return f"Error executing code: {e}"


def explain_answer(question: str, result, system_text: str):

    prompt = f"""
User question:
{question}

Computed result:
{result}

Explain the result in 1–3 sentences, concise and clear.
Do NOT hallucinate — rely solely on the provided result and column meanings.

Column meanings:
{system_text}
"""

    return llm.invoke(prompt).content.strip()


# ===================================================
# MASTER ASK FUNCTION
# ===================================================

def ask_question(agent_unused, question: str, system_text: str, df=None):
    if df is None:
        return "Dataset not loaded."

    q = question.lower().strip()

    # Special-case: who submits the most tickets / who called in most
    if ("submit" in q or "called in" in q or "call in" in q or "caller" in q) and "most" in q:
        return answer_who_submits_most(df)

    # Special-case: handler speeds
    if "handle" in q or "handles" in q or "handling" in q:
        if "slowest" in q or "longest" in q:
            return answer_handler_speed(df, "slowest")
        if "fastest" in q or "quickest" in q or "shortest" in q:
            return answer_handler_speed(df, "fastest")

    # Generic LLM flow
    code = generate_code(df, question, system_text)
    result = execute_code(df, code)
    explanation = explain_answer(question, result, system_text)

    return f"**Answer:** {result}\n\n**Explanation:** {explanation}"


def build_agent(df, system_text):
    return df
