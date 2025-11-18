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
    xls = pd.ExcelFile(source)
    sheets = xls.sheet_names

    if "Data" not in sheets:
        main_sheet = sheets[0]
    else:
        main_sheet = "Data"

    df = pd.read_excel(xls, main_sheet)

    if "System Prompt" in sheets:
        df_sys = pd.read_excel(xls, "System Prompt")
        col = df_sys.columns[0]
        system_text = "\n".join(df_sys[col].dropna().astype(str).tolist())
    else:
        system_text = ""

    if "Questions" in sheets:
        df_q = pd.read_excel(xls, "Questions")
        col = df_q.columns[0]
        sample_questions = df_q[col].dropna().astype(str).tolist()
    else:
        sample_questions = []

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

    for col in ["Created_Date", "Due_Date", "Resolved_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df, system_text, sample_questions


# ============================================================
# COLUMN SEMANTICS
# ============================================================

COLUMN_SEMANTICS = """
This dataset represents call center tickets.

Columns and meanings:
- Ticket_Number: ID of the ticket.
- Subject: high-level issue category. Do NOT invent new categories.
- Description: free-text explanation of the issue.
- Caller: person who reported the problem.
- Status: current state of the ticket.
- Created_Date: timestamp when the ticket was created.
- Due_Date: SLA target deadline.
- Resolved_Date: timestamp when the ticket was resolved.
- Resolved_By: agent responsible for resolving it.
- Resolution_Time_Seconds: numeric duration to resolve.
- SLA_Met: True/False indicator of SLA compliance.

Rules:
- Ticket volume -> group by Caller.
- Agent speed -> group by Resolved_By using mean Resolution_Time_Seconds.
- Longest/shortest -> max/min Resolution_Time_Seconds.
- SLA -> use SLA_Met when available.
- Do NOT merge or invent subject categories.
- Trend analysis uses Created_Date-based time buckets.
"""


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

INTENT_PROMPT = """
You classify user questions about a call center ticket dataset.

Choose EXACTLY one intent label:

overview
count
top_subjects
top_callers
ticket_duration_extreme
agent_performance
sla_summary
sla_by_agent
subject_resolution
outlier_tickets
trend
forecast
executive_summary
recommendations
workflow_focus
other

Return ONLY the label.
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

    if any(k in q for k in ["pattern", "patterns", "trend", "trends"]):
        return "trend"
    if any(k in q for k in ["forecast", "predict", "future"]):
        return "forecast"

    msg = f"{INTENT_PROMPT}\n\nQuestion:\n{question}\n\nIntent label:"
    intent = llm.invoke(msg).content.strip().lower()
    return intent if intent in ALLOWED_INTENTS else "other"


# ============================================================
# HELPERS
# ============================================================

def normalize_sla(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    true_vals = {"true", "yes", "1", "y"}
    false_vals = {"false", "no", "0", "n"}

    def to_bool(x):
        if x in true_vals: return True
        if x in false_vals: return False
        return None

    return s.map(to_bool)


def human_time(seconds):
    if seconds is None:
        return "N/A"
    try:
        seconds = float(seconds)
    except:
        return "N/A"
    hours = seconds / 3600
    days = seconds / 86400
    if days >= 1:
        return f"{int(seconds):,} sec (~{hours:.1f} hrs / {days:.1f} days)"
    if hours >= 1:
        return f"{int(seconds):,} sec (~{hours:.1f} hrs)"
    return f"{int(seconds):,} sec (~{seconds/60:.1f} min)"


def to_datetime_safe(series: pd.Series):
    return pd.to_datetime(series, errors="coerce")


# ============================================================
# ANALYTIC FUNCTIONS (DETERMINISTIC)
# ============================================================

def analyze_overview(df):
    total = len(df)
    rt = df["Resolution_Time_Seconds"]
    avg_rt = float(rt.dropna().mean()) if rt.notna().any() else None

    sla_bool = normalize_sla(df["SLA_Met"])
    compliance = float((sla_bool == True).mean() * 100) if sla_bool.notna().any() else None

    subj_counts = df["Subject"].dropna().value_counts().head(5).reset_index()
    subj_counts.columns = ["Subject", "Count"]

    return {
        "intent": "overview",
        "metrics": {
            "total_tickets": total,
            "avg_resolution_seconds": avg_rt,
            "sla_compliance_percent": compliance,
        },
        "top_subjects": subj_counts.to_dict(orient="records"),
    }


def analyze_count(question, df):
    q = question.lower()
    sla_bool = normalize_sla(df["SLA_Met"])

    if "outside sla" in q or "breach" in q:
        valid = sla_bool.dropna()
        breaches = int((valid == False).sum())
        return {
            "intent": "count",
            "type": "outside_sla",
            "total_with_sla": len(valid),
            "breaches": breaches,
        }

    return {
        "intent": "count",
        "type": "tickets",
        "total_tickets": len(df),
    }


def analyze_top_subjects(df):
    records = df["Subject"].dropna().value_counts().reset_index()
    records.columns = ["Subject", "Count"]
    return {
        "intent": "top_subjects",
        "subjects": records.to_dict(orient="records"),
    }


def analyze_top_callers(df):
    records = df["Caller"].dropna().value_counts().reset_index()
    records.columns = ["Caller", "Count"]
    return {
        "intent": "top_callers",
        "callers": records.to_dict(orient="records"),
    }


def analyze_ticket_duration_extreme(question, df):
    q = question.lower()
    rt = df["Resolution_Time_Seconds"].dropna()
    if rt.empty:
        return {"intent": "ticket_duration_extreme", "tickets": [], "kind": "none"}

    extreme_val = rt.min() if "shortest" in q else rt.max()
    kind = "shortest" if "shortest" in q else "longest"

    subset = df[df["Resolution_Time_Seconds"] == extreme_val]

    tickets = []
    for _, row in subset.iterrows():
        tickets.append({
            "Ticket_Number": str(row["Ticket_Number"]),
            "Subject": str(row["Subject"]),
            "Description": str(row["Description"]),
            "Caller": str(row["Caller"]),
            "Resolved_By": str(row["Resolved_By"]),
            "Resolution_Time_Seconds": float(row["Resolution_Time_Seconds"]),
            "SLA_Met": str(row["SLA_Met"]),
        })

    return {
        "intent": "ticket_duration_extreme",
        "kind": kind,
        "extreme_resolution_seconds": float(extreme_val),
        "tickets": tickets,
    }


def analyze_agent_performance(df):
    df_res = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"])
    if df_res.empty:
        return {"intent": "agent_performance", "fastest": [], "slowest": []}

    grouped = df_res.groupby("Resolved_By")["Resolution_Time_Seconds"].mean().sort_values()
    fastest_val = float(grouped.iloc[0])
    slowest_val = float(grouped.iloc[-1])

    return {
        "intent": "agent_performance",
        "fastest": [
            {"Resolved_By": a, "avg_resolution_seconds": fastest_val}
            for a in grouped[grouped == fastest_val].index.tolist()
        ],
        "slowest": [
            {"Resolved_By": a, "avg_resolution_seconds": slowest_val}
            for a in grouped[grouped == slowest_val].index.tolist()
        ],
        "gap_seconds": slowest_val - fastest_val,
    }


def analyze_sla_summary(df):
    sla_bool = normalize_sla(df["SLA_Met"])
    valid = sla_bool.dropna()
    if valid.empty:
        return {"intent": "sla_summary", "has_data": False}

    total = len(valid)
    met = int((valid == True).sum())
    breaches = int((valid == False).sum())

    breached_tickets = df.loc[sla_bool == False, "Ticket_Number"].astype(str).tolist()

    return {
        "intent": "sla_summary",
        "has_data": True,
        "total_with_sla": total,
        "met_sla": met,
        "breached_sla": breaches,
        "sla_compliance_percent": met / total * 100,
        "outside_sla_percent": breaches / total * 100,
        "breached_ticket_numbers": breached_tickets,
    }


def analyze_sla_by_agent(df):
    sla_bool = normalize_sla(df["SLA_Met"])
    df2 = df.copy()
    df2["SLA_Bool"] = sla_bool
    df2 = df2.dropna(subset=["Resolved_By", "SLA_Bool"])

    if df2.empty:
        return {"intent": "sla_by_agent", "best": [], "worst": []}

    grouped = df2.groupby("Resolved_By")["SLA_Bool"].mean().sort_values(ascending=False)

    return {
        "intent": "sla_by_agent",
        "best": [
            {"Resolved_By": a, "sla_compliance_percent": float(grouped.iloc[0]) * 100}
            for a in grouped[grouped == grouped.iloc[0]].index.tolist()
        ],
        "worst": [
            {"Resolved_By": a, "sla_compliance_percent": float(grouped.iloc[-1]) * 100}
            for a in grouped[grouped == grouped.iloc[-1]].index.tolist()
        ],
    }


def analyze_subject_resolution(df):
    df2 = df.dropna(subset=["Subject", "Resolution_Time_Seconds"])
    if df2.empty:
        return {"intent": "subject_resolution", "subjects_by_avg": [], "longest_subjects": []}

    grouped = df2.groupby("Subject")["Resolution_Time_Seconds"].mean().sort_values(ascending=False)
    longest_val = float(grouped.iloc[0])

    return {
        "intent": "subject_resolution",
        "subjects_by_avg": [
            {"Subject": subj, "avg_resolution_seconds": float(val)}
            for subj, val in grouped.items()
        ],
        "longest_subjects": [
            {"Subject": subj, "avg_resolution_seconds": longest_val}
            for subj in grouped[grouped == longest_val].index.tolist()
        ],
    }


def analyze_outlier_tickets(df):
    rt_series = df["Resolution_Time_Seconds"].dropna()
    if len(rt_series) < 5:
        return {"intent": "outlier_tickets", "outlier_threshold_seconds": None, "outlier_tickets": []}

    q1 = float(rt_series.quantile(0.25))
    q3 = float(rt_series.quantile(0.75))
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    outliers = df[df["Resolution_Time_Seconds"] >= threshold].copy()
    outliers = outliers.sort_values("Resolution_Time_Seconds", ascending=False).head(3)

    tickets = []
    for _, row in outliers.iterrows():
        tickets.append({
            "Ticket_Number": str(row["Ticket_Number"]),
            "Subject": str(row["Subject"]),
            "Description": str(row["Description"]),
            "Caller": str(row["Caller"]),
            "Resolved_By": str(row["Resolved_By"]),
            "Resolution_Time_Seconds": float(row["Resolution_Time_Seconds"]),
            "SLA_Met": str(row["SLA_Met"]),
        })

    return {
        "intent": "outlier_tickets",
        "outlier_threshold_seconds": threshold,
        "outlier_tickets": tickets,
    }


def build_time_buckets(df):
    created = to_datetime_safe(df["Created_Date"])
    if created.isna().all():
        return []

    df2 = df.copy()
    df2["Created_Date_DT"] = created.dt.date

    sla_bool = normalize_sla(df2["SLA_Met"])

    buckets = []
    for d, group in df2.groupby("Created_Date_DT"):
        rt = group["Resolution_Time_Seconds"].dropna()
        sla_sub = sla_bool.loc[group.index].dropna()

        buckets.append({
            "date": str(d),
            "total_tickets": len(group),
            "avg_resolution_seconds": float(rt.mean()) if not rt.empty else None,
            "sla_breach_percent": (sla_sub == False).mean() * 100 if not sla_sub.empty else None,
        })

    buckets.sort(key=lambda x: x["date"])
    return buckets


def analyze_trend(df):
    buckets = build_time_buckets(df)
    if len(buckets) < 2:
        return {"intent": "trend", "time_buckets": buckets, "has_trend": False}

    mid = len(buckets) // 2
    first = buckets[:mid]
    last = buckets[mid:]

    def avg(items, key):
        vals = [x[key] for x in items if x[key] is not None]
        return sum(vals)/len(vals) if vals else None

    return {
        "intent": "trend",
        "time_buckets": buckets,
        "avg_resolution_first_half": avg(first, "avg_resolution_seconds"),
        "avg_resolution_second_half": avg(last, "avg_resolution_seconds"),
        "avg_breach_first_half": avg(first, "sla_breach_percent"),
        "avg_breach_second_half": avg(last, "sla_breach_percent"),
        "has_trend": True,
    }


def analyze_forecast(df):
    buckets = build_time_buckets(df)
    if len(buckets) < 3:
        return {"intent": "forecast", "time_buckets": buckets, "has_forecast": False}

    recent = buckets[-3:]

    def project(values):
        if len(values) < 2:
            return None
        diffs = [values[i] - values[i-1] for i in range(1, len(values))]
        return values[-1] + sum(diffs)/len(diffs)

    rt_vals = [b["avg_resolution_seconds"] for b in recent if b["avg_resolution_seconds"] is not None]
    breach_vals = [b["sla_breach_percent"] for b in recent if b["sla_breach_percent"] is not None]

    return {
        "intent": "forecast",
        "time_buckets": buckets,
        "current_avg_resolution_seconds": rt_vals[-1] if rt_vals else None,
        "forecast_avg_resolution_seconds": project(rt_vals) if len(rt_vals) >= 2 else None,
        "current_breach_percent": breach_vals[-1] if breach_vals else None,
        "forecast_breach_percent": project(breach_vals) if len(breach_vals) >= 2 else None,
        "has_forecast": True,
    }


def analyze_executive_summary(df):
    return {
        "intent": "executive_summary",
        "overview": analyze_overview(df),
        "agent_performance": analyze_agent_performance(df),
        "sla_summary": analyze_sla_summary(df),
        "subject_resolution": analyze_subject_resolution(df),
        "outliers": analyze_outlier_tickets(df),
    }


def analyze_recommendations(df):
    return {
        "intent": "recommendations",
        "overview": analyze_overview(df),
        "agent_performance": analyze_agent_performance(df),
        "sla_summary": analyze_sla_summary(df),
        "subject_resolution": analyze_subject_resolution(df),
        "outliers": analyze_outlier_tickets(df),
        "trend": analyze_trend(df),
    }


def analyze_workflow_focus(df):
    return {
        "intent": "workflow_focus",
        "subject_resolution": analyze_subject_resolution(df),
        "sla_by_agent": analyze_sla_by_agent(df),
        "outliers": analyze_outlier_tickets(df),
    }


def analyze_other(df, question):
    base = analyze_overview(df)
    base["intent"] = "other"
    base["note"] = "Fallback pattern detection."
    return base


# ============================================================
# PRESENTATION LAYER (UPDATED — NO BOLD/ASTERISKS)
# ============================================================

PRESENTATION_PROMPT = """
You are MILOlytics, an executive-quality analytics assistant for myBasePay.

You will receive:
- A structured result dict.
- Column semantics.

Formatting rules:
- DO NOT use markdown formatting.
- DO NOT use *, **, _, or any markdown emphasis.
- Use plain text only.
- You may use emojis, bullet points, headings (plain text), spacing, and short paragraphs.
- Do NOT restate the user's question.

Style rules:
- Start with a clear heading using emojis (e.g., 📊 Overview, ⏱️ Resolution Insights).
- Use bullet lists for metrics.
- Use short paragraphs for interpretation.
- Convert seconds into readable time (hours/days).
- Be concise, clear, and executive.
- Adapt tone based on intent (overview, trend, forecast, SLA, etc.).

Output plain text only.
"""


def build_final_answer(result: dict) -> str:
    prompt = f"""
{PRESENTATION_PROMPT}

Here is the structured analysis result:

{json.dumps(result, indent=2)}

Column semantics:
{COLUMN_SEMANTICS}

Write the final formatted answer:
"""
    return llm.invoke(prompt).content.strip()


# ============================================================
# MAIN ENTRYPOINT
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
    if df is None:
        return "Dataset not loaded."

    intent = classify_intent(question)
    result = run_analysis_for_intent(intent, question, df)
    return build_final_answer(result)


def build_agent(df, system_text=""):
    return df
