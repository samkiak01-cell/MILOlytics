# logic/agent.py

from pathlib import Path
from typing import Tuple, List, Union, IO, Any, Dict
import json

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
"""


# ===================================================
# CODE GENERATION PROMPT (NORMAL MODE)
# ===================================================

CODE_PROMPT = """
You are a senior Python data analyst.

You MUST output ONLY Python code that uses the Pandas DataFrame `df`
to answer the user's question about the ticket dataset.

The DataFrame is named `df` and has the columns described below.
Use the semantic rules correctly (Caller submits, Resolved_By handles, SLA is based on Due_Date_Time vs Resolution).

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
   - result["mode"]:
       * A short tag like "max", "min", "outlier", "count", "share", "sla", etc.

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
# EXECUTE GENERATED CODE (NORMAL MODE)
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
# EXPLANATION LAYER (NORMAL QUESTIONS)
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
# EXECUTIVE INSIGHT MODE — HELPERS
# ===================================================

INSIGHT_KEYWORDS = [
    "insight",
    "insights",
    "overview",
    "summary",
    "summarize",
    "overall",
    "how are we doing",
    "how are we doing?",
    "health",
    "high level",
    "big picture",
    "performance",
    "report",
    "diagnose"
]


def is_insight_request(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in INSIGHT_KEYWORDS)


def to_seconds_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bucket_issue(description: str) -> str:
    """
    Simple rule-based categorization into Option A-style buckets:
    - Login / Access
    - Password Reset
    - System Error
    - Performance Issue
    - Data Update
    - Account Setup
    - Connectivity
    - Other
    """
    if not isinstance(description, str):
        description = str(description)

    d = description.lower()

    if any(k in d for k in ["login", "log in", "sign in", "access", "locked out", "credential", "auth"]):
        return "Login / Access"
    if "password" in d or "passcode" in d:
        return "Password Reset"
    if any(k in d for k in ["timeout", "slow", "lag", "performance"]):
        return "Performance Issue"
    if any(k in d for k in ["error", "exception", "crash", "bug", "fail", "not responding"]):
        return "System Error"
    if any(k in d for k in ["update", "change", "modify", "correct data", "adjust data"]):
        return "Data Update"
    if any(k in d for k in ["new account", "account setup", "onboard", "create user"]):
        return "Account Setup"
    if any(k in d for k in ["network", "vpn", "connect", "connection", "offline", "wifi"]):
        return "Connectivity"

    return "Other"


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

            # fastest/slowest ticket IDs if available
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

    # Issue buckets from Description
    if "Description" in df.columns:
        cats = df["Description"].astype(str).apply(bucket_issue)
        vc_cat = cats.value_counts()
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
Write a **clear, boss-friendly summary** with REAL specifics:
- Use actual agent names, ticket numbers, categories, and counts from the metrics.
- Do NOT invent agents, tickets, or numbers.
- If something is missing from metrics, simply skip it.

Use this structure:

Executive Summary:
- 1–2 sentences summarizing overall performance, using numbers where possible.

What's Going Well:
- 2–4 bullet points
- Mention specific agents, categories, or ticket types that perform well.

What Needs Attention:
- 2–5 bullet points
- Call out slow agents, long tickets, SLA problems, repeated callers, or heavy issue types.
- Reference exact names, ticket IDs, and numbers.

Opportunities:
- 2–4 bullet points
- Suggest concrete actions (e.g. "assign complex tickets to <agent>", "train <agent> on <category>", "focus on reducing <category> tickets").
- Again, use the actual names & categories from the metrics.

Be direct and practical. Avoid fluff. Keep it within 10–15 total bullet points + the short executive summary.
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
    return f"Insights:\n\n{summary}"


# ===================================================
# MAIN ENTRYPOINT FOR STREAMLIT
# ===================================================

def ask_question(agent_unused, question: str, system_text: str, df=None) -> str:
    """
    Main function used by the app.

    If question looks like an "insight" / "overview" request,
    we switch into Executive Insight Mode.

    Otherwise, we use the original code-generation + explanation path.

    Returns a formatted string.
    """
    if df is None:
        return "Dataset not loaded."

    # 🔹 Executive Insight Mode
    if is_insight_request(question):
        return build_executive_insights(question, system_text, df)

    # 🔹 Normal Question Mode

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
