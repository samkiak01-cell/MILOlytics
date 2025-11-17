# logic/agent.py

from pathlib import Path
from typing import Tuple, List, Union, IO, Any, Dict
import json

import pandas as pd
from langchain_openai import ChatOpenAI


# ===================================================
# GLOBAL LLM
# ===================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===================================================
# LOAD & PREPROCESS EXCEL
# ===================================================

def load_excel(
    source: Union[str, Path, IO[bytes]]
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Load the Excel file and return:
      - df: main ticket dataset from 'Data' sheet (preprocessed)
      - system_text: text from 'System Prompt' (if present)
      - sample_questions: from 'Questions' sheet (if present)
    """

    xls = pd.ExcelFile(source)
    sheets = xls.sheet_names

    if "Data" not in sheets:
        raise ValueError("Excel must contain a sheet named 'Data'.")

    df = pd.read_excel(xls, "Data")
    df = preprocess_df(df)

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


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning + add Issue_Category using LLM into fixed buckets.
    """

    # Date columns
    date_cols = [
        "Created_Date_Time",
        "Updated_Date_Time",
        "Due_Date_Time",
        "Resolution_Date_Time",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric columns
    if "Resolution_Time_Seconds" in df.columns:
        df["Resolution_Time_Seconds"] = pd.to_numeric(
            df["Resolution_Time_Seconds"], errors="coerce"
        )

    # Text normalization
    for col in ["Caller", "Created_By", "Updated_By", "Resolved_By", "Description"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": None, "None": None, "": None})
            )

    # Add issue category via LLM (Option A buckets)
    df = add_issue_category(df)

    return df


def add_issue_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use LLM once per row to assign each Description into one of the fixed buckets:

      - "Login / Access"
      - "Password Reset"
      - "System Error"
      - "Performance Issue"
      - "Data Update"
      - "Account Setup"
      - "Connectivity"
      - "Other"

    If Issue_Category already exists, we leave it as-is.
    """

    if "Issue_Category" in df.columns:
        return df

    if "Description" not in df.columns:
        df["Issue_Category"] = "Other"
        return df

    categories: List[str] = []

    for desc in df["Description"].astype(str).tolist():
        prompt = f"""
You are categorizing call center tickets into ONE of these buckets:

- Login / Access
- Password Reset
- System Error
- Performance Issue
- Data Update
- Account Setup
- Connectivity
- Other

Description:
\"\"\"{desc}\"\"\"


Rules:
- Choose exactly ONE bucket from the list.
- Return ONLY the bucket text, nothing else.
"""

        try:
            cat = llm.invoke(prompt).content.strip()
        except Exception:
            cat = "Other"

        # Safety: ensure it matches one of the allowed labels
        allowed = {
            "login / access",
            "password reset",
            "system error",
            "performance issue",
            "data update",
            "account setup",
            "connectivity",
            "other",
        }
        norm = cat.lower().strip()
        # Simple normalization
        mapped = None
        for a in allowed:
            if norm.startswith(a.split()[0]):  # loose match on first word
                mapped = a
                break
        if mapped is None:
            mapped = "other"

        # Use canonical label capitalization
        if mapped == "login / access":
            final = "Login / Access"
        elif mapped == "password reset":
            final = "Password Reset"
        elif mapped == "system error":
            final = "System Error"
        elif mapped == "performance issue":
            final = "Performance Issue"
        elif mapped == "data update":
            final = "Data Update"
        elif mapped == "account setup":
            final = "Account Setup"
        elif mapped == "connectivity":
            final = "Connectivity"
        else:
            final = "Other"

        categories.append(final)

    df["Issue_Category"] = categories
    return df


# ===================================================
# COLUMN SEMANTICS (Still needed, but cleaner)
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
- Issue_Category: category of the issue derived from Description using fixed buckets.

Important semantic rules:
- "Who submits the most tickets?" or "who called in the most?" -> group by Caller.
- "Which call center member handles/resolves tickets fastest/slowest?" rules:
    * ALWAYS use the column "Resolved_By" (never Created_By).
    * A resolved ticket is defined STRICTLY as: Resolution_Date_Time not null AND Resolution_Time_Seconds not null.
    * Use only df rows where both of these conditions are true.
    * Example filter to get resolved tickets:
          df2 = df[df["Resolution_Time_Seconds"].notna() & df["Resolution_Date_Time"].notna()]
    * Then compute average Resolution_Time_Seconds grouped by Resolved_By.
    * Detect ties (multiple agents with same avg).
- SLA questions -> compare Resolution_Date_Time (or Resolution_Time_Seconds) with Due_Date_Time.
- For questions about "outliers", "unusual", "anomalies", "stands out", treat these as items that are clearly at the extreme (top or bottom) compared to the rest.
- For "most common issue/subject/type", you should use the Issue_Category column (not raw Description).
"""


# ===================================================
# CODE GENERATION PROMPT (for analytical questions)
# ===================================================

CODE_PROMPT = """
You are a senior Python data analyst.

You MUST output ONLY Python code that uses the Pandas DataFrame `df`
to answer the user's question about the ticket dataset.

The DataFrame is named `df` and has the columns described below.
Use the semantic rules correctly (Caller submits, Resolved_By handles, SLA is based on Due_Date_Time vs Resolution, Issue_Category is the subject bucket).

Your job:
1. Analyze the question and decide what needs to be computed.
2. Use Pandas operations on `df` to compute the answer.
3. Construct a Python dict named result with at least these keys:

   result = {
       "answer": <short human-readable answer string>,
       "details": <structured data that includes any key values you used>,
       "mode": <short tag like "max", "min", "count", "outlier", "sla", "category", "agent_performance", "caller", "insight", etc.>
   }

   - result["answer"]:
       * A concise string that directly answers the question.
       * For multiple answers (ties), combine names in a human way,
         e.g. "Abraham Lincoln and Olivia Johnson" or
         "INC0010109, INC0010110 and INC0010200".
   - result["details"]:
       * A dict or list of dicts with the underlying numeric values.
       * Always include numeric metrics you used (e.g. counts, seconds, percentages).
   - result["mode"]:
       * A short tag describing the type of analysis.

VERY IMPORTANT:
- ALWAYS detect ties for max/min style questions.
- For anomaly / unusual / outlier style questions:
  * Return the 1–3 most extreme items.
- For "most common issue" style questions:
  * Use df["Issue_Category"] and group by that column.

Technical constraints:
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
# INSIGHT MODE DETECTION
# ===================================================

INSIGHT_KEYWORDS = [
    "insight",
    "insights",
    "overview",
    "summary",
    "summarize",
    "overall",
    "how are we doing",
    "health",
    "high level",
    "big picture",
    "performance",
    "report",
    "diagnose",
]


def is_insight_request(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in INSIGHT_KEYWORDS)


# ===================================================
# EXECUTIVE INSIGHT MODE
# ===================================================

def to_seconds_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_insight_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a structured set of metrics to feed into Executive Insight mode.
    Everything here is deterministic and concrete: names, counts, ticket IDs, times.
    """

    metrics: Dict[str, Any] = {}
    metrics["total_tickets"] = int(len(df))

    # Resolution time stats
    if "Resolution_Time_Seconds" in df.columns:
        rt = to_seconds_series(df["Resolution_Time_Seconds"])
        metrics["resolved_count"] = int(rt.notna().sum())

        if metrics["resolved_count"] > 0:
            metrics["avg_resolution_sec"] = float(rt.mean())
            metrics["min_resolution_sec"] = float(rt.min())
            metrics["max_resolution_sec"] = float(rt.max())

            if "Ticket_Number" in df.columns:
                idx_min = rt.idxmin()
                idx_max = rt.idxmax()
                metrics["fastest_ticket"] = {
                    "Ticket_Number": str(df.loc[idx_min, "Ticket_Number"]),
                    "Resolution_Time_Seconds": float(rt.loc[idx_min]),
                }
                metrics["slowest_ticket"] = {
                    "Ticket_Number": str(df.loc[idx_max, "Ticket_Number"]),
                    "Resolution_Time_Seconds": float(rt.loc[idx_max]),
                }

    # SLA metrics
    if "Due_Date_Time" in df.columns and "Resolution_Date_Time" in df.columns:
        due = pd.to_datetime(df["Due_Date_Time"], errors="coerce")
        res = pd.to_datetime(df["Resolution_Date_Time"], errors="coerce")
        valid = due.notna() & res.notna()
        metrics["sla_checked_count"] = int(valid.sum())

        if metrics["sla_checked_count"] > 0:
            late_mask = res[valid] > due[valid]
            late_count = int(late_mask.sum())
            sla_rate = 100.0 * (1 - late_count / metrics["sla_checked_count"])
            metrics["sla_late_count"] = late_count
            metrics["sla_rate"] = sla_rate

    # Fastest / slowest agents
    if "Resolved_By" in df.columns and "Resolution_Time_Seconds" in df.columns:
        df_agents = df[["Resolved_By", "Resolution_Time_Seconds"]].dropna()
        df_agents["Resolution_Time_Seconds"] = to_seconds_series(df_agents["Resolution_Time_Seconds"])
        df_agents = df_agents.dropna()

        if not df_agents.empty:
            grouped = df_agents.groupby("Resolved_By")["Resolution_Time_Seconds"].mean()
            if len(grouped) > 0:
                min_val = grouped.min()
                max_val = grouped.max()
                fastest = grouped[grouped == min_val]
                slowest = grouped[grouped == max_val]

                metrics["fastest_agents"] = [
                    {"name": idx, "avg_seconds": float(val)} for idx, val in fastest.items()
                ]
                metrics["slowest_agents"] = [
                    {"name": idx, "avg_seconds": float(val)} for idx, val in slowest.items()
                ]

    # Top callers
    if "Caller" in df.columns:
        vc = df["Caller"].dropna().value_counts()
        metrics["top_callers"] = [
            {"caller": idx, "count": int(cnt)} for idx, cnt in vc.head(3).items()
        ]

    # Issue categories
    if "Issue_Category" in df.columns:
        vc_cat = df["Issue_Category"].value_counts()
        metrics["issue_categories"] = [
            {"category": idx, "count": int(cnt)} for idx, cnt in vc_cat.items()
        ]
        if not vc_cat.empty:
            metrics["top_issue_category"] = {
                "category": vc_cat.index[0],
                "count": int(vc_cat.iloc[0]),
            }

    return metrics


INSIGHT_PROMPT = """
You are an analytics assistant for a call center operations leader.

The user is asking for a high-level overview of how the program is performing.

You will be given:
- The user's question
- A structured metrics dictionary with concrete values (agents, ticket IDs, counts, seconds, categories)

Your job:
Write a clear, professional, executive-style summary that is still easy to read.

Tone:
- Mix of corporate and executive: direct, concise, action-oriented.
- Avoid fluff. Focus on facts, names, numbers, and concrete recommendations.

Structure your answer as:

Executive Summary:
- 1–2 sentences summarizing overall performance, using real numbers.

What's Going Well:
- 2–4 bullet points.
- Mention specific agents, categories, or ticket types that perform well.

What Needs Attention:
- 2–5 bullet points.
- Call out slow agents, long tickets, SLA problems, repeated callers, or heavy issue types.
- Reference exact names, ticket IDs, and numbers.

Opportunities:
- 2–4 bullet points.
- Suggest concrete actions (e.g., reassign complex tickets to specific agents, target a specific issue category, improve SLA on a certain priority).

Rules:
- Use only the metrics provided. Do NOT invent agents, tickets, or numbers.
- If some metrics are missing, simply omit them.
"""


def build_executive_insights(question: str, system_text: str, df: pd.DataFrame) -> str:
    """
    Executive Insight Mode:
    - Computes deterministic metrics
    - Asks the LLM to write a structured, specific summary
    """
    metrics = compute_insight_metrics(df)

    prompt = INSIGHT_PROMPT + f"""

User Question:
{question}

Metrics (JSON):
{json.dumps(metrics, indent=2)}
"""

    summary = llm.invoke(prompt).content.strip()
    return summary


# ===================================================
# ANSWER GENERATION (FINAL RESPONSE LAYER)
# ===================================================

FINAL_ANSWER_PROMPT = """
You are a data assistant for myBasePay.

You will be given:
- The user's original question
- A structured `result` dict that includes at least:
  - answer: a short direct answer (string)
  - details: numeric and structured context used to compute it
  - mode: a short label like "max", "min", "count", "outlier", "sla", "category", etc.
- Column meanings text

Your job is to produce the FINAL response to the user.
Do NOT repeat the raw `result` dict. Instead, write a polished, human-friendly answer.

Tone:
- Professional and clear, with a mix of corporate and executive style.
- Be direct and practical. No fluff.

Formatting rules:
- Start by directly answering the question in 1–2 sentences, using concrete values:
  * Names (agents, callers)
  * Ticket numbers
  * Counts
  * Times (convert seconds into hours/days when helpful)
- If useful, follow with up to 3 short bullet points that briefly explain or add context.
- For simple questions (e.g. "How many tickets..."), a short one-paragraph answer is enough.
- For more complex analytics (e.g. fastest/slowest, outliers, SLA, anomalies), include an extra 1–3 bullets.

Seconds conversion:
- If you see fields like "Resolution_Time_Seconds" or keys that contain "seconds", convert example values:
    259200 -> "259,200 seconds (72 hours / 3 days)"
    7200 -> "7,200 seconds (2 hours)"

Do NOT mention:
- Standard deviation, variance, or heavy statistics.
- The internal structure of the result dict.

User question:
{question}

Result dict:
{result}

Column meanings:
{system_text}

Write the final answer now:
"""


def build_final_answer(question: str, result: Any, system_text: str) -> str:
    prompt = FINAL_ANSWER_PROMPT.format(
        question=question,
        result=str(result),
        system_text=system_text,
    )
    answer = llm.invoke(prompt).content.strip()
    return answer


# ===================================================
# MAIN ENTRYPOINT FOR STREAMLIT
# ===================================================

def ask_question(agent_unused, question: str, system_text: str, df=None) -> str:
    """
    Main function used by the app.

    - If the question looks like an "insight" / "overview" request,
      we switch into Executive Insight Mode and produce a structured summary.
    - Otherwise, we use the code-generation + structured-result pipeline,
      and then let the LLM craft the final natural-language response.

    Returns a single, polished answer string (no rigid "Answer / Explanation" wrapper).
    """
    if df is None:
        return "Dataset not loaded."

    # Executive Insight Mode
    if is_insight_request(question):
        return build_executive_insights(question, system_text, df)

    # Normal analytical question
    code = generate_code(question, df, system_text)
    result = execute_code(df, code)

    if not isinstance(result, dict):
        result = {
            "answer": str(result),
            "details": {"note": "Result was not a dict; converted to string."},
            "mode": "raw",
        }

    final_answer = build_final_answer(question, result, system_text)
    return final_answer


def build_agent(df, system_text):
    """
    Kept for compatibility with the existing app structure.
    """
    return df
