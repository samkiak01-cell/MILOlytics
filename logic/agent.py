from __future__ import annotations
from typing import Any, List, Tuple, Union, IO
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI


# ============================================================
# LLM (shared)
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# LOAD EXCEL – expects a sheet named "Data"
# ============================================================

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

    # Optional system prompt sheet
    if "System Prompt" in sheets:
        df_sys = pd.read_excel(xls, "System Prompt")
        col = df_sys.columns[0]
        system_text = "\n".join(df_sys[col].dropna().astype(str).tolist())
    else:
        system_text = ""

    # Optional questions sheet
    if "Questions" in sheets:
        df_q = pd.read_excel(xls, "Questions")
        col = df_q.columns[0]
        sample_questions = df_q[col].dropna().astype(str).tolist()
    else:
        sample_questions = []

    return df, system_text, sample_questions


# ============================================================
# COLUMN SEMANTICS (for Demo Data-style schema)
# ============================================================

COLUMN_SEMANTICS = """
This dataset represents call center tickets.

Typical columns and meanings:
- Ticket_Number: ID of the ticket.
- Subject: high-level issue category for the ticket.
- Description: free-text explanation of the issue.
- Caller: person who called in and reported the problem.
- Priority: urgency of the ticket.
- Status: current state of the ticket (Open, In Progress, Resolved, etc.).
- Created_Date_Time: timestamp when the ticket was created.
- Due_Date_Time: SLA target date/time (if present).
- Resolved_Date_Time: timestamp when the ticket was resolved (if resolved).
- Resolution_Time_Seconds: numeric resolution time (difference between creation and resolution).
- Resolved_By: call center agent who resolved the ticket.
- SLA_Met: whether the ticket met SLA (True/False, Yes/No, 1/0).

Semantic rules:
- "Who submits the most tickets" or "who called in the most" -> group by Caller.
- "Which agent resolves tickets fastest/slowest" -> group by Resolved_By and use average Resolution_Time_Seconds,
  only on rows where Resolution_Time_Seconds is not null.
- "Which ticket took the longest/shortest" -> use max/min on Resolution_Time_Seconds per Ticket_Number.
- "Outside SLA" or "SLA breaches":
    * If SLA_Met exists, treat False/0/"no" as outside SLA and True/1/"yes" as within SLA.
    * If SLA_Met does not exist but Due_Date_Time and Resolved_Date_Time exist, treat
      Resolved_Date_Time > Due_Date_Time as outside SLA.
- "Most common subject" or "subject category with longest average resolution time" -> group by Subject.
- Outliers: choose the top or bottom 1–3 tickets or agents that clearly stand out on resolution time or SLA performance.
- Trend/forecast questions:
    * Use Created_Date_Time to group by date, week, or month.
    * For each period, compute:
        - ticket volume
        - average resolution time
        - SLA breach rate
    * Trend: compare early vs late periods.
    * Forecast: extrapolate naively based on recent changes (e.g., last 3 periods).
"""

# ============================================================
# INTENT CLASSIFICATION PROMPT (Layer 1)
# ============================================================

INTENT_PROMPT = """
You classify user questions about a call center ticket dataset.

Given the question, choose ONE of these intent labels:

- "overview"            -> general understanding ("overview", "what stands out")
- "count"               -> pure counts (how many tickets, how many breaches)
- "top_subjects"        -> which subjects appear most / main subjects
- "top_callers"         -> who calls in the most, high ticket volume callers
- "ticket_duration_extreme" -> longest or shortest ticket questions
- "agent_performance"   -> fastest/slowest agents, average resolution per agent
- "sla_summary"         -> how many outside SLA, SLA rate overall
- "sla_by_agent"        -> who is best/worst at meeting SLA
- "subject_resolution"  -> subjects with longest or shortest average resolution time
- "outlier_tickets"     -> unusual or risky tickets, outliers
- "trend"               -> patterns over time, trends in resolution times or SLA
- "forecast"            -> future projection of resolution time or SLA compliance
- "executive_summary"   -> high-level summary for executives
- "recommendations"     -> what to improve, suggestions, next steps
- "workflow_focus"      -> which part of the workflow needs attention
- "other"               -> anything else

Return ONLY the label, nothing else.
"""


def classify_intent(question: str) -> str:
    """Use the LLM to classify the question into a high-level intent."""
    msg = f"{INTENT_PROMPT}\n\nQuestion:\n{question}\n\nIntent label:"
    intent = llm.invoke(msg).content.strip().lower()
    # Basic safety
    allowed = {
        "overview", "count", "top_subjects", "top_callers",
        "ticket_duration_extreme", "agent_performance", "sla_summary",
        "sla_by_agent", "subject_resolution", "outlier_tickets",
        "trend", "forecast", "executive_summary", "recommendations",
        "workflow_focus", "other"
    }
    if intent not in allowed:
        return "other"
    return intent


# ============================================================
# ANALYTICAL CODE GENERATION (Layer 2)
# ============================================================

CODE_PROMPT = """
You are a senior Python analyst.

You are given a Pandas DataFrame named `df` that contains call center tickets.
Your job is to write Python code that computes the answer to the user's question
using ONLY pandas and standard Python (no imports, no printing).

You MUST define a variable named `result` at the end, as a dict:

result = {
    "intent": <string>,          # the high-level intent you handled
    "headline": <string>,        # short 1-sentence factual answer
    "data": <JSON-safe dict>,    # underlying metrics and lists
    "insights": <list of strings>  # optional bullet-like insights
}

Constraints:
- Do NOT import anything.
- Do NOT print anything.
- Do NOT use markdown.
- Use only JSON-serialisable types in result["data"] and result["insights"]:
  dict, list, str, int, float, bool, or None.
- Convert pandas / numpy / timestamps to strings, ints, or floats before storing them.

Intent handling guidance:
- For "overview":
    * Count total tickets.
    * If Subject exists, compute distribution (top 3 subjects + their counts).
    * If Resolution_Time_Seconds exists, compute average resolution time.
    * If SLA_Met exists, compute SLA compliance rate (percentage met).
- For "count":
    * Answer simple "how many" questions (tickets, breaches, etc.).
- For "top_subjects":
    * Group by Subject, count tickets per subject, sort descending, take top 3.
- For "top_callers":
    * Group by Caller, count tickets per caller, sort descending, highlight top N (e.g. 3).
- For "ticket_duration_extreme":
    * Use Resolution_Time_Seconds to find longest or shortest ticket(s).
    * Support ties: if multiple tickets share the same extreme value, include them all.
- For "agent_performance":
    * Group by Resolved_By, use average Resolution_Time_Seconds.
    * Identify fastest and slowest agents (again, handle ties).
- For "sla_summary":
    * If SLA_Met exists, compute:
        - total tickets with SLA_Met not null
        - number and percentage of breaches (SLA_Met == False)
    * Also list the ticket numbers of breached tickets (up to a reasonable limit, e.g. 20).
- For "sla_by_agent":
    * Group by Resolved_By and SLA_Met.
    * Compute per-agent SLA compliance rate.
    * Identify best and worst agents for SLA.
- For "subject_resolution":
    * Group by Subject, compute average Resolution_Time_Seconds by subject.
    * Identify which subject has the longest average resolution time.
- For "outlier_tickets":
    * Use Resolution_Time_Seconds:
        - Compute mean and standard deviation (or IQR style logic).
        - Identify 1–3 tickets with unusually long resolution times (much higher than typical).
- For "trend":
    * If Created_Date_Time exists:
        - Convert to date or period (e.g. df["Created_Date_Time"].dt.date).
        - Group by date (or week) to compute:
            total_tickets, avg_resolution_time, sla_breach_rate.
        - Store these per-period metrics in result["data"]["time_buckets"].
- For "forecast":
    * Build on "trend" logic:
        - Use the per-period average resolution time and/or SLA breach rate.
        - Compute a simple trend based on the last few periods (e.g. last 3).
        - Extrapolate naively one step forward and store that as
          predicted_avg_resolution and predicted_sla_breach_rate in result["data"].
- For "executive_summary":
    * Combine overview-style metrics:
        - total tickets, SLA rate, average resolution time, top subject(s),
          best/worst agent if possible.
- For "recommendations":
    * You can compute the same metrics as overview + SLA + subject_resolution
      and then include numeric clues that will help a language model write recommendations.
- For "workflow_focus":
    * Surface which dimension (agent, subject, SLA, time-of-day, etc.) looks most problematic,
      using basic comparisons such as highest breach count or slowest subjects.

If the question truly cannot be answered from the columns available, then:

result = {
    "intent": "error",
    "headline": "I don't know.",
    "data": {"reason": "brief explanation why this cannot be computed"},
    "insights": []
}
"""


def generate_code(question: str, intent: str, df: pd.DataFrame, system_text: str) -> str:
    """
    Ask the LLM to write Python code that:
    - Uses df
    - Computes an answer
    - Builds result = {"intent", "headline", "data", "insights"}
    """

    cols = list(df.columns)

    prompt = f"""
{CODE_PROMPT}

Column descriptions:
{COLUMN_SEMANTICS}

System prompt text (if any):
{system_text}

DataFrame columns:
{cols}

High-level intent for this question:
{intent}

User question:
{question}

Write ONLY executable Python code. No comments. No markdown. No print.
"""

    code = llm.invoke(prompt).content.strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return code


# ============================================================
# EXECUTE GENERATED CODE (Layer 2)
# ============================================================

def execute_code(df: pd.DataFrame, code: str) -> dict:
    """
    Execute the generated Python code with df in scope.
    Expect a dict named `result`.
    """
    local_scope: dict[str, Any] = {"df": df.copy(), "result": None}
    try:
        exec(code, {}, local_scope)
        result = local_scope.get("result", None)

        if not isinstance(result, dict):
            return {
                "intent": "error",
                "headline": "I don't know.",
                "data": {"reason": "Generated code did not return a dict."},
                "insights": []
            }

        # Basic normalisation
        if "intent" not in result:
            result["intent"] = "other"
        if "headline" not in result:
            result["headline"] = "I don't know."
        if "data" not in result:
            result["data"] = {}
        if "insights" not in result:
            result["insights"] = []

        return result

    except Exception as e:
        return {
            "intent": "error",
            "headline": "I don't know.",
            "data": {"reason": f"Error executing generated code: {e}"},
            "insights": []
        }


# ============================================================
# PRESENTATION LAYER (Layer 3)
#   Hybrid style: light emojis + structured bullets
# ============================================================

PRESENTATION_PROMPT = """
You are MILOlytics, a data analyst assistant for myBasePay.

You will receive:
- The user's question
- A structured result dict with:
    intent: high-level intent (overview, top_callers, sla_summary, trend, forecast, etc.)
    headline: short 1-sentence factual answer
    data: JSON-safe dict with metrics and lists
    insights: list of short bullet-like insights
- Column semantics for context

Your job:
- Produce a final answer in a clean, readable, HYBRID style:
  * Use short section headers (with or without emojis) like:
      "📊 Dataset Overview", "🏆 Fastest Agent", "⚠️ SLA Risk", "📈 Trend", "🧭 Recommendation".
  * Use bullet points for lists of callers, agents, subjects, tickets, etc.
  * Keep everything concise and business-friendly.
- DO NOT use the literal labels "Answer:" or "Explanation:".
- Instead, start with a short headline or section heading, then bullets.
- For time-like metrics in seconds, if values look large (e.g. > 3600),
  you may naturally reference them in hours/days in the text (e.g. "about 6 hours").
- For trend/forecast:
  * Emphasise whether things are getting better or worse over time.
  * Summarise any simple projection (e.g., "if the current trend continues, average resolution
    time could rise to around X hours and SLA compliance could fall to around Y%.").

Output MUST be plain text that can be rendered directly in Streamlit.
No markdown code fences.
"""


def build_final_answer(question: str, result: dict) -> str:
    prompt = f"""
{PRESENTATION_PROMPT}

User question:
{question}

Result dict:
{result}

Column semantics:
{COLUMN_SEMANTICS}

Final answer (hybrid style, no 'Answer:'/'Explanation:' labels):
"""

    answer = llm.invoke(prompt).content.strip()
    return answer


# ============================================================
# PUBLIC API FOR STREAMLIT
# ============================================================

def ask_question(agent_unused, question: str, system_text: str, df=None) -> str:
    """
    Main entrypoint used by the Streamlit app.

    Steps:
    1) Classify intent of the question.
    2) Generate Python analysis code appropriate to that intent.
    3) Execute the code to get a structured result.
    4) Let the LLM format a clean, readable final answer that fits the intent.
    """
    if df is None:
        return "Dataset not loaded."

    # Layer 1 — Intent classification
    intent = classify_intent(question)

    # Layer 2 — Code generation + execution
    code = generate_code(question, intent, df, system_text)
    result = execute_code(df, code)

    # Layer 3 — Presentation
    final_answer = build_final_answer(question, result)
    return final_answer


def build_agent(df, system_text):
    """
    Kept for compatibility with the existing app structure.
    We don't need a complex agent object; df itself is enough.
    """
    return df
