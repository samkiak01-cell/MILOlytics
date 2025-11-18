from __future__ import annotations
from typing import Any, List, Tuple, Union, IO, Dict
from pathlib import Path
import json

import pandas as pd
from langchain_openai import ChatOpenAI


# ============================================================
# GLOBAL LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# LOAD EXCEL – Demo schema, sheet "Data"
# ============================================================

def load_excel(
    source: Union[str, Path, IO[bytes]]
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Load the Excel file and return:
      - df: main ticket dataset from 'Data' sheet
      - system_text: text from 'System Prompt' (if present)
      - sample_questions: from 'Questions' sheet (if present)

    Expects the demo schema columns:
      Ticket_Number, Subject, Description, Caller, Status,
      Created_Date, Due_Date, Resolved_Date, Resolved_By,
      Resolution_Time_Seconds, SLA_Met
    """
    xls = pd.ExcelFile(source)
    sheets = xls.sheet_names

    if "Data" not in sheets:
        # Fallback: use first sheet if 'Data' not found
        main_sheet = sheets[0]
    else:
        main_sheet = "Data"

    df = pd.read_excel(xls, main_sheet)

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

    # Ensure expected columns exist so analytics don’t crash
    expected_cols = [
        "Ticket_Number",
        "Subject",
        "Description",
        "Caller",
        "Status",
        "Created_Date",
        "Due_Date",
        "Resolved_Date",
        "Resolved_By",
        "Resolution_Time_Seconds",
        "SLA_Met",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    # Normalize date columns to datetime where possible
    for col in ["Created_Date", "Due_Date", "Resolved_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df, system_text, sample_questions


# ============================================================
# COLUMN SEMANTICS (for LLM context)
# ============================================================

COLUMN_SEMANTICS = """
This dataset represents call center tickets.

Columns and meanings (Demo schema):
- Ticket_Number: ID of the ticket.
- Subject: high-level issue category for the ticket. Do NOT invent new categories.
- Description: free-text explanation of the issue.
- Caller: person who called in and reported the problem.
- Status: current state of the ticket (Open, In Progress, Resolved, etc.).
- Created_Date: timestamp when the ticket was created.
- Due_Date: SLA target date/time.
- Resolved_Date: timestamp when the ticket was resolved (if resolved).
- Resolved_By: call center agent who resolved the ticket.
- Resolution_Time_Seconds: numeric resolution time in seconds.
- SLA_Met: whether the ticket met SLA (True/False, Yes/No, 1/0).

Key rules:
- "Who submits the most tickets" or "who called in the most" -> group by Caller.
- "Which agent resolves tickets fastest/slowest" -> group by Resolved_By
  using average Resolution_Time_Seconds on resolved tickets.
- "Which ticket took the longest/shortest" -> use max/min Resolution_Time_Seconds.
- "Outside SLA" or "SLA breaches":
    * Prefer SLA_Met if present: False/0/"no" => outside SLA; True/1/"yes" => within SLA.
    * If SLA_Met missing but Due_Date and Resolved_Date present, treat
      Resolved_Date > Due_Date as outside SLA.
- "Most common subject" -> group by Subject exactly as given; do NOT merge categories.
- "Subjects with longest average resolution time" -> group by Subject + mean Resolution_Time_Seconds.
- Outliers: tickets with unusually long Resolution_Time_Seconds vs the rest.
- Trend/forecast:
    * Use Created_Date to group by date.
    * For each date: ticket volume, average resolution time, SLA breach rate.
    * Trends are based on early vs late periods.
    * Forecast is a naive extrapolation of recent changes.
"""


# ============================================================
# INTENT CLASSIFICATION (Layer 1)
# ============================================================

INTENT_PROMPT = """
You classify user questions about a call center ticket dataset.

Given the question, choose ONE of these intent labels:

- "overview"            -> general understanding ("overview", "what stands out")
- "count"               -> "how many" questions
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

ALLOWED_INTENTS = {
    "overview", "count", "top_subjects", "top_callers",
    "ticket_duration_extreme", "agent_performance", "sla_summary",
    "sla_by_agent", "subject_resolution", "outlier_tickets",
    "trend", "forecast", "executive_summary", "recommendations",
    "workflow_focus", "other"
}


def classify_intent(question: str) -> str:
    q = question.lower()

    # --- Hard rules FIRST for stability ---
    trend_keywords = ["pattern", "patterns", "trend", "trends"]
    if any(k in q for k in trend_keywords):
        return "trend"

    forecast_keywords = ["forecast", "predict", "projection", "future", "expected"]
    if any(k in q for k in forecast_keywords):
        return "forecast"

    # LLM fallback
    msg = f"{INTENT_PROMPT}\n\nQuestion:\n{question}\n\nIntent label:"
    intent = llm.invoke(msg).content.strip().lower()
    if intent not in ALLOWED_INTENTS:
        return "other"
    return intent


# ============================================================
# HELPER FUNCTIONS FOR ANALYTICS (Layer 2 base)
# ============================================================

def normalize_sla(series: pd.Series) -> pd.Series:
    """Convert SLA_Met column to boolean True/False, safely."""
    if series is None:
        return pd.Series(dtype=bool)

    s = series.astype(str).str.strip().str.lower()
    true_vals = {"true", "yes", "y", "1"}
    false_vals = {"false", "no", "n", "0"}

    def to_bool(x: str) -> Any:
        if x in true_vals:
            return True
        if x in false_vals:
            return False
        return None

    return s.map(to_bool)


def human_time(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    try:
        seconds = float(seconds)
    except Exception:
        return "N/A"
    hours = seconds / 3600.0
    days = seconds / 86400.0
    if days >= 1:
        return f"{int(seconds):,} sec (~{hours:.1f} hrs / {days:.1f} days)"
    elif hours >= 1:
        return f"{int(seconds):,} sec (~{hours:.1f} hrs)"
    else:
        return f"{int(seconds):,} sec (~{seconds/60.0:.1f} min)"


def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


# ============================================================
# ANALYTIC FUNCTIONS PER INTENT (Layer 2)
# ============================================================

def analyze_overview(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)

    rt = df["Resolution_Time_Seconds"]
    avg_rt = float(rt.dropna().mean()) if rt.notna().any() else None

    sla_bool = normalize_sla(df["SLA_Met"])
    if sla_bool.notna().any():
        valid = sla_bool.dropna()
        compliance = float((valid == True).mean() * 100.0)
    else:
        compliance = None

    subj_counts = (
        df["Subject"].dropna()
        .value_counts()
        .head(5)
        .reset_index()
        .rename(columns={"index": "Subject", "Subject": "Count"})
        .to_dict(orient="records")
    )

    return {
        "intent": "overview",
        "metrics": {
            "total_tickets": total,
            "avg_resolution_seconds": avg_rt,
            "sla_compliance_percent": compliance,
        },
        "top_subjects": subj_counts,
    }


def analyze_count(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    q = question.lower()
    sla_bool = normalize_sla(df["SLA_Met"])

    if "outside sla" in q or "breach" in q or "breaches" in q:
        valid = sla_bool.dropna()
        breaches = int((valid == False).sum())
        total = int(len(valid))
        return {
            "intent": "count",
            "type": "outside_sla",
            "total_with_sla": total,
            "breaches": breaches,
        }
    else:
        total = int(len(df))
        return {
            "intent": "count",
            "type": "tickets",
            "total_tickets": total,
        }


def analyze_top_subjects(df: pd.DataFrame) -> Dict[str, Any]:
    counts = (
        df["Subject"].dropna()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Subject", "Subject": "Count"})
    )
    records = counts.to_dict(orient="records")
    return {
        "intent": "top_subjects",
        "subjects": records,
    }


def analyze_top_callers(df: pd.DataFrame) -> Dict[str, Any]:
    counts = (
        df["Caller"].dropna()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Caller", "Caller": "Count"})
    )
    records = counts.to_dict(orient="records")
    return {
        "intent": "top_callers",
        "callers": records,
    }


def analyze_ticket_duration_extreme(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    q = question.lower()
    rt = df["Resolution_Time_Seconds"].dropna()
    if rt.empty:
        return {
            "intent": "ticket_duration_extreme",
            "tickets": [],
            "kind": "none",
        }

    if "shortest" in q or "fastest" in q:
        extreme_val = rt.min()
        kind = "shortest"
    else:
        extreme_val = rt.max()
        kind = "longest"

    subset = df[df["Resolution_Time_Seconds"] == extreme_val]

    tickets = []
    for _, row in subset.iterrows():
        tickets.append({
            "Ticket_Number": str(row.get("Ticket_Number")),
            "Subject": str(row.get("Subject")),
            "Description": str(row.get("Description")),
            "Caller": str(row.get("Caller")),
            "Resolved_By": str(row.get("Resolved_By")),
            "Resolution_Time_Seconds": float(row.get("Resolution_Time_Seconds")),
            "SLA_Met": str(row.get("SLA_Met")),
        })

    return {
        "intent": "ticket_duration_extreme",
        "kind": kind,
        "extreme_resolution_seconds": float(extreme_val),
        "tickets": tickets,
    }


def analyze_agent_performance(df: pd.DataFrame) -> Dict[str, Any]:
    df_res = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"])
    if df_res.empty:
        return {
            "intent": "agent_performance",
            "fastest": [],
            "slowest": [],
        }

    grouped = df_res.groupby("Resolved_By")["Resolution_Time_Seconds"].mean().sort_values()
    fastest_val = float(grouped.iloc[0])
    slowest_val = float(grouped.iloc[-1])
    fastest_agents = grouped[grouped == fastest_val].index.tolist()
    slowest_agents = grouped[grouped == slowest_val].index.tolist()

    gap = slowest_val - fastest_val

    return {
        "intent": "agent_performance",
        "fastest": [
            {"Resolved_By": a, "avg_resolution_seconds": fastest_val}
            for a in fastest_agents
        ],
        "slowest": [
            {"Resolved_By": a, "avg_resolution_seconds": slowest_val}
            for a in slowest_agents
        ],
        "gap_seconds": gap,
    }


def analyze_sla_summary(df: pd.DataFrame) -> Dict[str, Any]:
    sla_bool = normalize_sla(df["SLA_Met"])
    valid = sla_bool.dropna()
    if valid.empty:
        return {
            "intent": "sla_summary",
            "has_data": False,
        }

    total = int(len(valid))
    met = int((valid == True).sum())
    breaches = int((valid == False).sum())
    compliance = met / total * 100.0
    breach_pct = breaches / total * 100.0

    breached_tickets = df.loc[sla_bool == False, "Ticket_Number"].astype(str).tolist()

    return {
        "intent": "sla_summary",
        "has_data": True,
        "total_with_sla": total,
        "met_sla": met,
        "breached_sla": breaches,
        "sla_compliance_percent": compliance,
        "outside_sla_percent": breach_pct,
        "breached_ticket_numbers": breached_tickets,
    }


def analyze_sla_by_agent(df: pd.DataFrame) -> Dict[str, Any]:
    sla_bool = normalize_sla(df["SLA_Met"])
    df2 = df.copy()
    df2["SLA_Bool"] = sla_bool
    df2 = df2.dropna(subset=["Resolved_By", "SLA_Bool"])
    if df2.empty:
        return {
            "intent": "sla_by_agent",
            "best": [],
            "worst": [],
        }

    grouped = df2.groupby("Resolved_By")["SLA_Bool"].mean().sort_values(ascending=False)
    best_val = float(grouped.iloc[0]) * 100.0
    worst_val = float(grouped.iloc[-1]) * 100.0
    best_agents = grouped[grouped == grouped.iloc[0]].index.tolist()
    worst_agents = grouped[grouped == grouped.iloc[-1]].index.tolist()

    return {
        "intent": "sla_by_agent",
        "best": [{"Resolved_By": a, "sla_compliance_percent": best_val} for a in best_agents],
        "worst": [{"Resolved_By": a, "sla_compliance_percent": worst_val} for a in worst_agents],
    }


def analyze_subject_resolution(df: pd.DataFrame) -> Dict[str, Any]:
    df2 = df.dropna(subset=["Subject", "Resolution_Time_Seconds"])
    if df2.empty:
        return {
            "intent": "subject_resolution",
            "subjects_by_avg": [],
            "longest_subjects": [],
        }

    grouped = df2.groupby("Subject")["Resolution_Time_Seconds"].mean().sort_values(ascending=False)
    longest_val = float(grouped.iloc[0])
    longest_subjs = grouped[grouped == grouped.iloc[0]].index.tolist()

    return {
        "intent": "subject_resolution",
        "subjects_by_avg": [
            {"Subject": subj, "avg_resolution_seconds": float(val)}
            for subj, val in grouped.items()
        ],
        "longest_subjects": [
            {"Subject": subj, "avg_resolution_seconds": longest_val}
            for subj in longest_subjs
        ],
    }


def analyze_outlier_tickets(df: pd.DataFrame) -> Dict[str, Any]:
    rt_series = df["Resolution_Time_Seconds"].dropna()
    if len(rt_series) < 5:
        return {
            "intent": "outlier_tickets",
            "outlier_threshold_seconds": None,
            "outlier_tickets": [],
        }

    q1 = float(rt_series.quantile(0.25))
    q3 = float(rt_series.quantile(0.75))
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    outliers = df[df["Resolution_Time_Seconds"] >= threshold].copy()
    outliers = outliers.sort_values("Resolution_Time_Seconds", ascending=False).head(3)

    tickets = []
    for _, row in outliers.iterrows():
        tickets.append({
            "Ticket_Number": str(row.get("Ticket_Number")),
            "Subject": str(row.get("Subject")),
            "Description": str(row.get("Description")),
            "Caller": str(row.get("Caller")),
            "Resolved_By": str(row.get("Resolved_By")),
            "Resolution_Time_Seconds": float(row.get("Resolution_Time_Seconds")),
            "SLA_Met": str(row.get("SLA_Met")),
        })

    return {
        "intent": "outlier_tickets",
        "outlier_threshold_seconds": threshold,
        "outlier_tickets": tickets,
    }


def build_time_buckets(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group by Created_Date (date only) and compute per-day stats."""
    created = to_datetime_safe(df["Created_Date"])
    if created.isna().all():
        return []

    df2 = df.copy()
    df2["Created_Date_DT"] = created.dt.date

    sla_bool = normalize_sla(df2["SLA_Met"])

    buckets = []
    for d, group in df2.groupby("Created_Date_DT"):
        total = int(len(group))
        rt = group["Resolution_Time_Seconds"].dropna()
        avg_rt = float(rt.mean()) if not rt.empty else None

        sla_sub = sla_bool.loc[group.index].dropna()
        if not sla_sub.empty:
            breaches = int((sla_sub == False).sum())
            breach_rate = breaches / len(sla_sub) * 100.0
        else:
            breaches = None
            breach_rate = None

        buckets.append({
            "date": str(d),
            "total_tickets": total,
            "avg_resolution_seconds": avg_rt,
            "sla_breach_percent": breach_rate,
        })

    buckets.sort(key=lambda x: x["date"])
    return buckets


def analyze_trend(df: pd.DataFrame) -> Dict[str, Any]:
    buckets = build_time_buckets(df)
    if len(buckets) < 2:
        return {
            "intent": "trend",
            "time_buckets": buckets,
            "has_trend": False,
        }

    mid = len(buckets) // 2
    first = buckets[:mid]
    last = buckets[mid:]

    def avg_of_key(items, key):
        vals = [x[key] for x in items if x[key] is not None]
        return float(sum(vals) / len(vals)) if vals else None

    avg_first_rt = avg_of_key(first, "avg_resolution_seconds")
    avg_last_rt = avg_of_key(last, "avg_resolution_seconds")
    avg_first_breach = avg_of_key(first, "sla_breach_percent")
    avg_last_breach = avg_of_key(last, "sla_breach_percent")

    return {
        "intent": "trend",
        "time_buckets": buckets,
        "avg_resolution_first_half": avg_first_rt,
        "avg_resolution_second_half": avg_last_rt,
        "avg_breach_first_half": avg_first_breach,
        "avg_breach_second_half": avg_last_breach,
        "has_trend": True,
    }


def analyze_forecast(df: pd.DataFrame) -> Dict[str, Any]:
    buckets = build_time_buckets(df)
    if len(buckets) < 3:
        return {
            "intent": "forecast",
            "time_buckets": buckets,
            "has_forecast": False,
        }

    recent = buckets[-3:]

    def project_next(values: List[float]) -> float | None:
        if len(values) < 2:
            return None
        diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
        avg_diff = sum(diffs) / len(diffs)
        return values[-1] + avg_diff

    rt_vals = [b["avg_resolution_seconds"] for b in recent if b["avg_resolution_seconds"] is not None]
    breach_vals = [b["sla_breach_percent"] for b in recent if b["sla_breach_percent"] is not None]

    current_avg_rt = rt_vals[-1] if rt_vals else None
    current_breach = breach_vals[-1] if breach_vals else None
    next_rt = project_next(rt_vals) if len(rt_vals) >= 2 else None
    next_breach = project_next(breach_vals) if len(breach_vals) >= 2 else None

    return {
        "intent": "forecast",
        "time_buckets": buckets,
        "current_avg_resolution_seconds": current_avg_rt,
        "forecast_avg_resolution_seconds": next_rt,
        "current_breach_percent": current_breach,
        "forecast_breach_percent": next_breach,
        "has_forecast": True if (next_rt is not None or next_breach is not None) else False,
    }


def analyze_executive_summary(df: pd.DataFrame) -> Dict[str, Any]:
    overview = analyze_overview(df)
    agent_perf = analyze_agent_performance(df)
    sla = analyze_sla_summary(df)
    subject_res = analyze_subject_resolution(df)
    outliers = analyze_outlier_tickets(df)

    return {
        "intent": "executive_summary",
        "overview": overview,
        "agent_performance": agent_perf,
        "sla_summary": sla,
        "subject_resolution": subject_res,
        "outliers": outliers,
    }


def analyze_recommendations(df: pd.DataFrame) -> Dict[str, Any]:
    overview = analyze_overview(df)
    agent_perf = analyze_agent_performance(df)
    sla = analyze_sla_summary(df)
    subject_res = analyze_subject_resolution(df)
    outliers = analyze_outlier_tickets(df)
    trend = analyze_trend(df)

    return {
        "intent": "recommendations",
        "overview": overview,
        "agent_performance": agent_perf,
        "sla_summary": sla,
        "subject_resolution": subject_res,
        "outliers": outliers,
        "trend": trend,
    }


def analyze_workflow_focus(df: pd.DataFrame) -> Dict[str, Any]:
    subject_res = analyze_subject_resolution(df)
    sla_agents = analyze_sla_by_agent(df)
    outliers = analyze_outlier_tickets(df)

    return {
        "intent": "workflow_focus",
        "subject_resolution": subject_res,
        "sla_by_agent": sla_agents,
        "outliers": outliers,
    }


def analyze_other(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    base = analyze_overview(df)
    base["intent"] = "other"
    base["note"] = "Question did not match a specific pattern; overview-based answer."
    return base


# ============================================================
# PRESENTATION LAYER – Option A (fully autonomous formatting)
# ============================================================

PRESENTATION_PROMPT = """
You are MILOlytics, an executive-quality analytics assistant for myBasePay.

You will receive:
- A structured "result" dict containing:
  - intent (what kind of question this was)
  - computed metrics (numbers, lists, dicts)
- Column semantics describing what each field means.

Your job:
- Produce a final answer in a clean, readable, executive style.
- DO NOT restate the user's question.
- Start directly with insight, using a strong header.
- You may choose the best structure for each answer:
  * Headings with icons (e.g., 📊, ⏱️, 🧑‍💼, 🚨, 📈, 🔮, 🧭)
  * Bullet lists for key metrics
  * Short paragraphs for context
  * Clearly highlighted risks/opportunities
- Keep it concise but insightful (no huge walls of text).
- Use plain language, avoid jargon.
- Where helpful, explain time-like fields (seconds) in natural terms (hours/days).
- Adapt the style to the intent:
  * "overview" / "executive_summary" -> big picture
  * "agent_performance" -> compare agents clearly
  * "sla_*" -> focus on risk and reliability
  * "trend"/"forecast" -> speak about direction over time
  * "recommendations" -> action-oriented bullets

Important:
- Output plain text only (no markdown code fences).
- No need to mention the word "intent".
"""


def build_final_answer(result: dict) -> str:
    prompt = f"""
{PRESENTATION_PROMPT}

Here is the structured analysis result to format:

{json.dumps(result, indent=2)}

Column semantics:
{COLUMN_SEMANTICS}

Write the final formatted answer:
"""
    return llm.invoke(prompt).content.strip()


# ============================================================
# MAIN ENTRYPOINT USED BY STREAMLIT
# ============================================================

def run_analysis_for_intent(intent: str, question: str, df: pd.DataFrame) -> Dict[str, Any]:
    if intent == "overview":
        return analyze_overview(df)
    if intent == "count":
        return analyze_count(question, df)
    if intent == "top_subjects":
        return analyze_top_subjects(df)
    if intent == "top_callers":
        return analyze_top_callers(df)
    if intent == "ticket_duration_extreme":
        return analyze_ticket_duration_extreme(question, df)
    if intent == "agent_performance":
        return analyze_agent_performance(df)
    if intent == "sla_summary":
        return analyze_sla_summary(df)
    if intent == "sla_by_agent":
        return analyze_sla_by_agent(df)
    if intent == "subject_resolution":
        return analyze_subject_resolution(df)
    if intent == "outlier_tickets":
        return analyze_outlier_tickets(df)
    if intent == "trend":
        return analyze_trend(df)
    if intent == "forecast":
        return analyze_forecast(df)
    if intent == "executive_summary":
        return analyze_executive_summary(df)
    if intent == "recommendations":
        return analyze_recommendations(df)
    if intent == "workflow_focus":
        return analyze_workflow_focus(df)
    return analyze_other(df, question)


def ask_question(df, question: str, system_text: str = "") -> str:
    """
    Main function used by the app.

    Pipeline:
    1) Classify intent.
    2) Run deterministic pandas analysis for that intent.
    3) Use LLM to format a natural-language, executive-style answer.
    """
    if df is None:
        return "Dataset not loaded."

    intent = classify_intent(question)
    result = run_analysis_for_intent(intent, question, df)
    final_answer = build_final_answer(result)
    return final_answer


# Optional stub for legacy compatibility (not used by app.py but harmless)
def build_agent(df, system_text=""):
    return df
