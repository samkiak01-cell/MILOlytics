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
# LOAD & PREPROCESS EXCEL (DEMO DATA SCHEMA)
# ===================================================

def load_excel(
    source: Union[str, Path, IO[bytes]]
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Load the Excel file and return:
      - df: main ticket dataset (preprocessed)
      - system_text: text from 'System Prompt' sheet (if present) or empty string
      - sample_questions: from 'Questions' sheet (if present) or empty list

    Expected schema for the main sheet (Demo Data):
      Ticket_Number
      Subject
      Description
      Caller
      Status
      Created_Date
      Due_Date
      Resolved_Date
      Resolved_By
      Resolution_Time_Seconds
      SLA_Met
    """

    xls = pd.ExcelFile(source)
    sheets = xls.sheet_names

    # Assume the first sheet is the data sheet if "Data" is not explicitly present
    if "Data" in sheets:
        data_sheet_name = "Data"
    else:
        data_sheet_name = sheets[0]

    df = pd.read_excel(xls, data_sheet_name)
    df = preprocess_df(df)

    # Optional: System Prompt sheet
    if "System Prompt" in sheets:
        df_sys = pd.read_excel(xls, "System Prompt")
        col = df_sys.columns[0]
        system_text = "\n".join(df_sys[col].dropna().astype(str).tolist())
    else:
        system_text = ""

    # Optional: Questions sheet
    if "Questions" in sheets:
        df_q = pd.read_excel(xls, "Questions")
        col = df_q.columns[0]
        sample_questions = df_q[col].dropna().astype(str).tolist()
    else:
        sample_questions = []

    return df, system_text, sample_questions


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning & normalization for Demo Data schema.
    - Parse dates
    - Cast numeric fields
    - Normalize key text columns
    - Add Subject_Normalized as merged subject category
    """

    # --- Normalize column names a bit, in case of minor variations ---
    rename_map = {
        "Ticket Number": "Ticket_Number",
        "Created Date": "Created_Date",
        "Due Date": "Due_Date",
        "Resolved Date": "Resolved_Date",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # --- Date columns ---
    for col in ["Created_Date", "Due_Date", "Resolved_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Numeric columns ---
    if "Resolution_Time_Seconds" in df.columns:
        df["Resolution_Time_Seconds"] = pd.to_numeric(
            df["Resolution_Time_Seconds"], errors="coerce"
        )

    # --- Normalize key text columns ---
    for col in ["Ticket_Number", "Subject", "Description", "Caller", "Status", "Resolved_By", "SLA_Met"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": None, "None": None, "": None})
            )

    # --- Add normalized subject buckets based on Subject ---
    if "Subject" in df.columns:
        df["Subject_Normalized"] = df["Subject"].apply(normalize_subject)
    else:
        df["Subject_Normalized"] = "General Inquiry / Other"

    return df


# ===================================================
# SUBJECT NORMALIZATION
# ===================================================

def normalize_subject(subject: Any) -> str:
    """
    Merge similar subjects into umbrella groups using simple keyword rules.

    Buckets:
      - Login & Authentication Issues
      - Password Reset Issues
      - Access / Permission Issues
      - Software / Application Errors
      - Hardware Issues
      - Data / Record Issues
      - General Inquiry / Other

    Uses the raw Subject text only (no made-up categories).
    """

    if not isinstance(subject, str):
        subject = "" if subject is None else str(subject)

    s = subject.lower()

    # 1) Login & Authentication Issues
    login_keywords = [
        "login", "log in", "sign in", "signin", "auth", "authentication",
        "credential", "credentials", "unable to login", "cannot login"
    ]
    if any(k in s for k in login_keywords):
        return "Login & Authentication Issues"

    # 2) Password Reset Issues
    pw_keywords = ["password", "passcode", "pwd", "reset password", "password reset"]
    if any(k in s for k in pw_keywords):
        return "Password Reset Issues"

    # 3) Access / Permission Issues
    access_keywords = [
        "access", "permission", "permissions", "role", "privilege",
        "entitlement", "unauthorized", "not authorized"
    ]
    if any(k in s for k in access_keywords):
        return "Access / Permission Issues"

    # 4) Hardware Issues
    hardware_keywords = [
        "printer", "laptop", "desktop", "pc", "mouse", "keyboard", "monitor",
        "headset", "phone", "device", "hardware"
    ]
    if any(k in s for k in hardware_keywords):
        return "Hardware Issues"

    # 5) Data / Record Issues
    data_keywords = [
        "data", "record", "field", "value", "incorrect", "wrong", "mismatch",
        "update", "edit", "change", "fix", "correction"
    ]
    if any(k in s for k in data_keywords):
        return "Data / Record Issues"

    # 6) Software / Application Errors
    software_keywords = [
        "error", "bug", "crash", "exception", "system", "application", "app",
        "not working", "failed", "failure", "issue"
    ]
    if any(k in s for k in software_keywords):
        return "Software / Application Errors"

    # 7) Default
    return "General Inquiry / Other"


# ===================================================
# COLUMN SEMANTICS (FOR THE LLM)
# ===================================================

COLUMN_SEMANTICS = """
This dataset represents call center support tickets exported from Jira.

Columns and meanings (Demo Data schema):
- Ticket_Number: ID of the ticket
- Subject: short title summarizing the issue (used as the primary category)
- Description: detailed text describing the issue
- Caller: person who called in and reported the ticket
- Status: state of the ticket (e.g., Open, In Progress, Resolved)
- Created_Date: date the ticket was created
- Due_Date: target date by which the ticket should be resolved (SLA target)
- Resolved_Date: date the ticket was actually resolved
- Resolved_By: call center member who resolved the ticket
- Resolution_Time_Seconds: time in seconds between Created_Date and Resolved_Date
- SLA_Met: indicator of whether the ticket met the SLA (e.g., "Yes"/"No")
- Subject_Normalized: umbrella subject category derived from Subject, used for grouping
    * Login & Authentication Issues
    * Password Reset Issues
    * Access / Permission Issues
    * Software / Application Errors
    * Hardware Issues
    * Data / Record Issues
    * General Inquiry / Other

Important semantic rules:
- "Who submits the most tickets?" or "who calls the most?" → group by Caller.
- "Which call center member handles tickets fastest/slowest?" →
    * Use Resolved_By and Resolution_Time_Seconds.
    * Only consider rows where Resolution_Time_Seconds is not null.
    * Compute average Resolution_Time_Seconds per Resolved_By.
    * Handle ties by including all agents with the same extreme value.
- "Which ticket took the longest?" → use Resolution_Time_Seconds, identify max per Ticket_Number.
- SLA questions:
    * Prefer SLA_Met if available.
    * If SLA_Met is missing, you may compare Resolved_Date vs Due_Date where both are present.
- "Most common issue type" or "subject" or "category" →
    * Use Subject_Normalized for high-level insight.
    * You may also mention the most frequent raw Subject values within that category.
- For "outliers", "anomalies", "unusual":
    * Consider tickets or agents at the extremes of Resolution_Time_Seconds or SLA_Met failures.
"""


# ===================================================
# CODE GENERATION PROMPT (ANALYTICAL QUESTIONS)
# ===================================================

CODE_PROMPT = """
You are a senior Python data analyst.

You MUST output ONLY Python code that uses the Pandas DataFrame `df`
to answer the user's question about the ticket dataset.

The DataFrame is named `df` and has the columns described below.
Use the semantic rules correctly (Caller, Resolved_By, SLA_Met, Resolution_Time_Seconds, Subject_Normalized, etc.).

Your job:
1. Analyze the question and decide what needs to be computed.
2. Use Pandas operations on `df` to compute the answer.
3. Construct a Python dict named result with at least these keys:

   result = {
       "answer": <short human-readable answer string>,
       "details": <structured data that includes any key values you used>,
       "mode": <short tag like "max", "min", "count", "outlier", "sla", "category", "agent_performance", "caller", etc.>
   }

   - result["answer"]:
       * A concise string that directly answers the question.
       * For multiple answers (ties), combine names in a natural way.
   - result["details"]:
       * A dict or list of dicts with the underlying numeric values.
       * Always include numeric metrics you used (e.g., counts, seconds, percentages).
   - result["mode"]:
       * A short tag describing the type of analysis.

VERY IMPORTANT:
- ALWAYS detect ties for max/min style questions (agents, tickets, subjects, callers).
- For "most common issue" style questions:
  * Use df["Subject_Normalized"] as the primary category field.
- For "which ticket took the longest/shortest" style questions:
  * Use Resolution_Time_Seconds and include Ticket_Number, Subject, Description, and Resolved_By in details.
- For "fastest/slowest agent" questions:
  * Use Resolved_By and average Resolution_Time_Seconds.
- For SLA questions:
  * Use SLA_Met where available (e.g., count of "Yes"/"No", SLA rate).
- For anomalies/outliers:
  * Return the 1–3 most extreme items and include their numeric metrics in details.

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
                    "Subject": str(df.loc[idx_min, "Subject"]),
                    "Resolution_Time_Seconds": float(rt.loc[idx_min]),
                }
                metrics["slowest_ticket"] = {
                    "Ticket_Number": str(df.loc[idx_max, "Ticket_Number"]),
                    "Subject": str(df.loc[idx_max, "Subject"]),
                    "Resolution_Time_Seconds": float(rt.loc[idx_max]),
                }

    # SLA metrics
    if "SLA_Met" in df.columns:
        sla_series = df["SLA_Met"].dropna().astype(str).str.lower()
        metrics["sla_tracked_count"] = int(len(sla_series))
        if metrics["sla_tracked_count"] > 0:
            yes_mask = sla_series.isin(["yes", "y", "true", "1"])
            sla_yes = int(yes_mask.sum())
            sla_rate = 100.0 * sla_yes / metrics["sla_tracked_count"]
            metrics["sla_met_count"] = sla_yes
            metrics["sla_not_met_count"] = metrics["sla_tracked_count"] - sla_yes
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

    # Subject categories (normalized)
    if "Subject_Normalized" in df.columns:
        vc_cat = df["Subject_Normalized"].value_counts()
        metrics["subject_categories"] = [
            {"category": idx, "count": int(cnt)} for idx, cnt in vc_cat.items()
        ]
        if not vc_cat.empty:
            metrics["top_subject_category"] = {
                "category": vc_cat.index[0],
                "count": int(vc_cat.iloc[0]),
            }

    return metrics


INSIGHT_PROMPT = """
You are an analytics assistant for a call center operations leader at myBasePay.

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
- 1–2 sentences summarizing overall performance, using real numbers where possible.

What's Going Well:
- 2–4 bullet points.
- Mention specific agents, categories, or ticket types that perform well.

What Needs Attention:
- 2–5 bullet points.
- Call out slow agents, long tickets, SLA problems, repeated callers, or heavy subject categories.
- Reference exact names, ticket IDs, and numbers from the metrics.

Opportunities:
- 2–4 bullet points.
- Suggest concrete actions (e.g., reassign complex tickets to specific agents, focus on a certain subject category, improve SLA for specific types of issues).

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
# FINAL ANSWER GENERATION (NORMAL QUESTIONS)
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
  * Subjects (and Subject_Normalized where relevant)
  * Counts
  * Times (convert seconds into hours/days when helpful)
- If useful, follow with up to 3 short bullet points that briefly explain or add context.
- For simple questions (e.g., "How many tickets...?"), a short one-paragraph answer is enough.
- For more complex analytics (e.g., fastest/slowest, outliers, SLA, anomalies), include an extra 1–3 bullets.

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

    Returns a single, polished answer string.
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
