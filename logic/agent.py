from __future__ import annotations
from typing import Any, List, Tuple, Union, IO, Dict
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI


# ============================================================
# GLOBAL LLM
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

    # Ensure key columns exist; if not, create empty
    expected_cols = [
        "Ticket_Number", "Subject", "Description", "Caller", "Status",
        "Created_Date", "Due_Date", "Resolved_Date", "Resolved_By",
        "Resolution_Time_Seconds", "SLA_Met",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    return df, system_text, sample_questions


# ============================================================
# COLUMN SEMANTICS FOR DEMO DATA
# ============================================================

COLUMN_SEMANTICS = """
This dataset represents call center tickets.

Columns and meanings (Demo schema):
- Ticket_Number: ID of the ticket.
- Subject: high-level issue category for the ticket (do NOT merge or invent new categories).
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
    msg = f"{INTENT_PROMPT}\n\nQuestion:\n{question}\n\nIntent label:"
    intent = llm.invoke(msg).content.strip().lower()
    if intent not in ALLOWED_INTENTS:
        return "other"
    return intent


# ============================================================
# HELPER FUNCTIONS FOR ANALYTICS
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

    headline_parts = [f"Total Tickets: {total}"]
    if avg_rt is not None:
        headline_parts.append(f"Average Resolution Time: {human_time(avg_rt)}")
    if compliance is not None:
        headline_parts.append(f"SLA Compliance Rate: {compliance:.1f}%")
    headline = " | ".join(headline_parts)

    insights = []
    if avg_rt is not None:
        insights.append(f"The average resolution time is {human_time(avg_rt)}, which may indicate potential bottlenecks.")
    if compliance is not None:
        insights.append(f"SLA compliance is {compliance:.1f}%, leaving room for improvement in service reliability.")
    if subj_counts:
        top_subj = subj_counts[0]["Subject"]
        insights.append(f"The most common reported subject is '{top_subj}', which likely deserves closer attention.")

    return {
        "intent": "overview",
        "headline": headline,
        "data": {
            "total_tickets": total,
            "avg_resolution_seconds": avg_rt,
            "sla_compliance_percent": compliance,
            "subject_distribution": subj_counts,
        },
        "insights": insights,
    }


def analyze_count(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    q = question.lower()

    sla_bool = normalize_sla(df["SLA_Met"])

    if "outside sla" in q or "breach" in q or "breaches" in q:
        valid = sla_bool.dropna()
        breaches = int((valid == False).sum())
        total = int(len(valid))
        headline = f"{breaches} ticket(s) were resolved outside SLA out of {total} with SLA information."
        data = {"outside_sla": breaches, "tickets_with_sla_info": total}
        insights = []
    else:
        total = int(len(df))
        headline = f"There are {total} tickets in this dataset."
        data = {"total_tickets": total}
        insights = []

    return {"intent": "count", "headline": headline, "data": data, "insights": insights}


def analyze_top_subjects(df: pd.DataFrame) -> Dict[str, Any]:
    counts = (
        df["Subject"].dropna()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Subject", "Subject": "Count"})
    )
    records = counts.to_dict(orient="records")

    if records:
        top = records[0]
        headline = f"The most common subject is '{top['Subject']}' with {top['Count']} tickets."
    else:
        headline = "No subjects are available in this dataset."

    return {
        "intent": "top_subjects",
        "headline": headline,
        "data": {"subjects": records},
        "insights": [],
    }


def analyze_top_callers(df: pd.DataFrame) -> Dict[str, Any]:
    counts = (
        df["Caller"].dropna()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Caller", "Caller": "Count"})
    )
    records = counts.head(5).to_dict(orient="records")

    if records:
        top = records[0]
        headline = f"The top caller is {top['Caller']} with {top['Count']} tickets."
    else:
        headline = "No caller data is available."

    return {
        "intent": "top_callers",
        "headline": headline,
        "data": {"callers": records},
        "insights": [],
    }


def analyze_ticket_duration_extreme(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    q = question.lower()
    rt = df["Resolution_Time_Seconds"].dropna()
    if rt.empty:
        return {
            "intent": "ticket_duration_extreme",
            "headline": "There are no resolved tickets with resolution times.",
            "data": {},
            "insights": [],
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
            "Caller": str(row.get("Caller")),
            "Resolved_By": str(row.get("Resolved_By")),
            "Resolution_Time_Seconds": float(row.get("Resolution_Time_Seconds")),
            "SLA_Met": str(row.get("SLA_Met")),
        })

    if kind == "longest":
        headline = f"The longest ticket took {human_time(extreme_val)} to resolve."
    else:
        headline = f"The fastest ticket was resolved in {human_time(extreme_val)}."

    return {
        "intent": "ticket_duration_extreme",
        "headline": headline,
        "data": {"kind": kind, "tickets": tickets, "extreme_resolution_seconds": float(extreme_val)},
        "insights": [],
    }


def analyze_agent_performance(df: pd.DataFrame) -> Dict[str, Any]:
    df_res = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"])
    if df_res.empty:
        return {
            "intent": "agent_performance",
            "headline": "There are no resolved tickets with agent information.",
            "data": {},
            "insights": [],
        }

    grouped = df_res.groupby("Resolved_By")["Resolution_Time_Seconds"].mean().sort_values()
    fastest_val = float(grouped.iloc[0])
    slowest_val = float(grouped.iloc[-1])
    fastest_agents = grouped[grouped == fastest_val].index.tolist()
    slowest_agents = grouped[grouped == slowest_val].index.tolist()

    gap = slowest_val - fastest_val

    data = {
        "fastest_agents": [
            {"Resolved_By": a, "avg_resolution_seconds": fastest_val} for a in fastest_agents
        ],
        "slowest_agents": [
            {"Resolved_By": a, "avg_resolution_seconds": slowest_val} for a in slowest_agents
        ],
        "gap_seconds": gap,
    }

    headline = (
        f"The fastest agent(s) resolve tickets in about {human_time(fastest_val)}, "
        f"while the slowest agent(s) average {human_time(slowest_val)}."
    )

    insights = [f"The performance gap between the fastest and slowest agents is about {human_time(gap)}."]

    return {
        "intent": "agent_performance",
        "headline": headline,
        "data": data,
        "insights": insights,
    }


def analyze_sla_summary(df: pd.DataFrame) -> Dict[str, Any]:
    sla_bool = normalize_sla(df["SLA_Met"])
    valid = sla_bool.dropna()
    if valid.empty:
        return {
            "intent": "sla_summary",
            "headline": "SLA information is not available for this dataset.",
            "data": {},
            "insights": [],
        }

    total = int(len(valid))
    met = int((valid == True).sum())
    breaches = int((valid == False).sum())
    compliance = met / total * 100.0
    breach_pct = breaches / total * 100.0

    breached_tickets = df.loc[sla_bool == False, "Ticket_Number"].astype(str).tolist()

    headline = f"{breaches} ticket(s) were resolved outside SLA ({breach_pct:.1f}% of {total} tickets with SLA data)."

    data = {
        "total_with_sla": total,
        "met_sla": met,
        "breached_sla": breaches,
        "sla_compliance_percent": compliance,
        "outside_sla_percent": breach_pct,
        "breached_ticket_numbers": breached_tickets,
    }

    insights = [
        f"Overall SLA compliance is {compliance:.1f}%, with {breach_pct:.1f}% of tickets breaching SLA.",
    ]

    return {
        "intent": "sla_summary",
        "headline": headline,
        "data": data,
        "insights": insights,
    }


def analyze_sla_by_agent(df: pd.DataFrame) -> Dict[str, Any]:
    sla_bool = normalize_sla(df["SLA_Met"])
    df2 = df.copy()
    df2["SLA_Bool"] = sla_bool
    df2 = df2.dropna(subset=["Resolved_By", "SLA_Bool"])
    if df2.empty:
        return {
            "intent": "sla_by_agent",
            "headline": "There is no SLA data per agent.",
            "data": {},
            "insights": [],
        }

    grouped = df2.groupby("Resolved_By")["SLA_Bool"].mean().sort_values(ascending=False)
    best_val = float(grouped.iloc[0]) * 100.0
    worst_val = float(grouped.iloc[-1]) * 100.0
    best_agents = grouped[grouped == grouped.iloc[0]].index.tolist()
    worst_agents = grouped[grouped == grouped.iloc[-1]].index.tolist()

    data = {
        "best_agents": [{"Resolved_By": a, "sla_compliance_percent": best_val} for a in best_agents],
        "worst_agents": [{"Resolved_By": a, "sla_compliance_percent": worst_val} for a in worst_agents],
    }

    headline = (
        f"The best SLA performance is around {best_val:.1f}% compliance, "
        f"while the weakest is around {worst_val:.1f}%."
    )

    insights = [
        f"Agents {', '.join(best_agents)} are leading in SLA performance, "
        f"while {', '.join(worst_agents)} appear to need support or process changes.",
    ]

    return {
        "intent": "sla_by_agent",
        "headline": headline,
        "data": data,
        "insights": insights,
    }


def analyze_subject_resolution(df: pd.DataFrame) -> Dict[str, Any]:
    df2 = df.dropna(subset=["Subject", "Resolution_Time_Seconds"])
    if df2.empty:
        return {
            "intent": "subject_resolution",
            "headline": "There is no subject-level resolution time data available.",
            "data": {},
            "insights": [],
        }

    grouped = df2.groupby("Subject")["Resolution_Time_Seconds"].mean().sort_values(ascending=False)
    longest_val = float(grouped.iloc[0])
    longest_subjs = grouped[grouped == grouped.iloc[0]].index.tolist()

    data = {
        "subjects_by_avg_resolution": [
            {"Subject": subj, "avg_resolution_seconds": float(val)}
            for subj, val in grouped.items()
        ],
        "longest_subjects": [
            {"Subject": subj, "avg_resolution_seconds": longest_val}
            for subj in longest_subjs
        ],
    }

    headline = (
        f"The subject(s) with the longest average resolution time "
        f"are {', '.join(longest_subjs)} at about {human_time(longest_val)}."
    )

    insights = [
        f"Issues under these subjects may be more complex or under-resourced and could benefit from targeted improvements."
    ]

    return {
        "intent": "subject_resolution",
        "headline": headline,
        "data": data,
        "insights": insights,
    }


def analyze_outlier_tickets(df: pd.DataFrame) -> Dict[str, Any]:
    rt_series = df["Resolution_Time_Seconds"].dropna()
    if len(rt_series) < 5:
        return {
            "intent": "outlier_tickets",
            "headline": "There is not enough data to reliably identify outlier tickets.",
            "data": {},
            "insights": [],
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
            "Caller": str(row.get("Caller")),
            "Resolved_By": str(row.get("Resolved_By")),
            "Resolution_Time_Seconds": float(row.get("Resolution_Time_Seconds")),
            "SLA_Met": str(row.get("SLA_Met")),
        })

    if tickets:
        headline = f"{len(tickets)} ticket(s) have unusually long resolution times compared with the rest."
    else:
        headline = "No strong outlier tickets were detected based on resolution time."

    data = {
        "outlier_threshold_seconds": threshold,
        "outlier_tickets": tickets,
    }

    insights = []
    if tickets:
        insights.append(
            "These outlier tickets likely represent either complex issues or process breakdowns and may merit root-cause analysis."
        )

    return {
        "intent": "outlier_tickets",
        "headline": headline,
        "data": data,
        "insights": insights,
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

    # sort by date
    buckets.sort(key=lambda x: x["date"])
    return buckets


def analyze_trend(df: pd.DataFrame) -> Dict[str, Any]:
    buckets = build_time_buckets(df)
    if len(buckets) < 2:
        return {
            "intent": "trend",
            "headline": "There is not enough time-based data to identify clear trends.",
            "data": {"time_buckets": buckets},
            "insights": [],
        }

    # Compare first half vs second half averages for resolution time and breach rate
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

    headline_parts = []
    if avg_first_rt is not None and avg_last_rt is not None:
        direction = "increasing" if avg_last_rt > avg_first_rt else "decreasing"
        headline_parts.append(
            f"Average resolution time appears to be {direction} over the observed period."
        )

    if avg_first_breach is not None and avg_last_breach is not None:
        direction = "increasing" if avg_last_breach > avg_first_breach else "decreasing"
        headline_parts.append(
            f"SLA breach rates also seem to be {direction} over time."
        )

    if not headline_parts:
        headline = "There are some time buckets available, but no clear trend emerges."
    else:
        headline = " ".join(headline_parts)

    data = {
        "time_buckets": buckets,
        "avg_resolution_first_half": avg_first_rt,
        "avg_resolution_second_half": avg_last_rt,
        "avg_breach_first_half": avg_first_breach,
        "avg_breach_second_half": avg_last_breach,
    }

    insights = []
    return {
        "intent": "trend",
        "headline": headline,
        "data": data,
        "insights": insights,
    }


def analyze_forecast(df: pd.DataFrame) -> Dict[str, Any]:
    buckets = build_time_buckets(df)
    if len(buckets) < 3:
        return {
            "intent": "forecast",
            "headline": "There is not enough historical data to make a meaningful forecast.",
            "data": {"time_buckets": buckets},
            "insights": [],
        }

    # Use last 3 periods for naive projections
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

    headline_parts = []
    if current_avg_rt is not None and next_rt is not None:
        direction = "rise" if next_rt > current_avg_rt else "fall"
        headline_parts.append(
            f"If recent patterns continue, average resolution time may {direction} from {human_time(current_avg_rt)} to about {human_time(next_rt)}."
        )

    if current_breach is not None and next_breach is not None:
        direction = "increase" if next_breach > current_breach else "decrease"
        headline_parts.append(
            f"SLA breach rates are projected to {direction} from roughly {current_breach:.1f}% to about {next_breach:.1f}%."
        )

    if not headline_parts:
        headline = "A forecast cannot be reliably estimated from the available data."
    else:
        headline = " ".join(headline_parts)

    data = {
        "time_buckets": buckets,
        "current_avg_resolution_seconds": current_avg_rt,
        "forecast_avg_resolution_seconds": next_rt,
        "current_breach_percent": current_breach,
        "forecast_breach_percent": next_breach,
    }

    insights = []

    return {
        "intent": "forecast",
        "headline": headline,
        "data": data,
        "insights": insights,
    }


def analyze_executive_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Combine key metrics into one result; LLM will narrate."""
    overview = analyze_overview(df)
    agent_perf = analyze_agent_performance(df)
    sla = analyze_sla_summary(df)
    subject_res = analyze_subject_resolution(df)

    data = {
        "overview": overview["data"],
        "agent_performance": agent_perf["data"],
        "sla_summary": sla["data"],
        "subject_resolution": subject_res["data"],
    }

    headline = "High-level summary of call center performance based on the current dataset."

    insights = []
    return {
        "intent": "executive_summary",
        "headline": headline,
        "data": data,
        "insights": insights,
    }


def analyze_recommendations(df: pd.DataFrame) -> Dict[str, Any]:
    """Reuse metrics; LLM will turn them into next-step recommendations."""
    overview = analyze_overview(df)
    agent_perf = analyze_agent_performance(df)
    sla = analyze_sla_summary(df)
    subject_res = analyze_subject_resolution(df)
    outliers = analyze_outlier_tickets(df)

    data = {
        "overview": overview["data"],
        "agent_performance": agent_perf["data"],
        "sla_summary": sla["data"],
        "subject_resolution": subject_res["data"],
        "outliers": outliers["data"],
    }

    headline = "Data-driven recommendations based on current performance, SLA results, and outliers."

    insights = []
    return {
        "intent": "recommendations",
        "headline": headline,
        "data": data,
        "insights": insights,
    }


def analyze_workflow_focus(df: pd.DataFrame) -> Dict[str, Any]:
    """Surface big problem areas: slowest subjects, worst SLA agents."""
    subject_res = analyze_subject_resolution(df)
    sla_agents = analyze_sla_by_agent(df)
    outliers = analyze_outlier_tickets(df)

    data = {
        "subject_resolution": subject_res["data"],
        "sla_by_agent": sla_agents["data"],
        "outliers": outliers["data"],
    }

    headline = "Key pressure points in the workflow based on slow subjects, SLA performance, and extreme tickets."

    insights = []
    return {
        "intent": "workflow_focus",
        "headline": headline,
        "data": data,
        "insights": insights,
    }


def analyze_other(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """Fallback: treat unknown questions as overview-style."""
    base = analyze_overview(df)
    base["intent"] = "other"
    base["headline"] = (
        base["headline"] + " (Answer derived from a general overview because the question did not match a specific pattern.)"
    )
    return base


# ============================================================
# PRESENTATION LAYER (Layer 3)
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
- Produce a final answer in a clean, readable HYBRID style (the one the user likes), e.g.:

  📊 Dataset Overview
  - Total Tickets: 95
  - Average Resolution Time: about 11.6 hours
  - SLA Compliance Rate: 47.4%

  🏷️ Top Subjects
  - Software Bug: 12 tickets
  - Password Reset: 12 tickets
  - Login Issue: 11 tickets

  🔍 Insights
  - Short, sharp observations and what they mean.

Guidelines:
- Use short headers like:
    "📊 Dataset Overview", "🏷️ Top Subjects", "👥 Top Callers",
    "🏆 Fastest Agent", "🐢 Slowest Agent", "⚠️ SLA Risk",
    "📈 Trend", "🔮 Forecast", "📌 Executive Summary", "🧭 Recommendations".
- Use bullet points for key metrics.
- Keep explanations concise and business-friendly.
- For numeric fields with "seconds" in the name, it's helpful to mention approximate hours/days,
  but do not overdo the math; natural phrasing is fine.
- DO NOT use literal labels "Answer:" or "Explanation:".
- Output plain text that can be shown directly in Streamlit (no code fences).

Now, format the final answer.
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

Final answer:
"""
    return llm.invoke(prompt).content.strip()


# ============================================================
# MAIN ENTRYPOINTS FOR STREAMLIT
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


def ask_question(agent_unused, question: str, system_text: str, df=None) -> str:
    """
    Main entrypoint used by the Streamlit app.

    1) Classify intent from the question.
    2) Run deterministic pandas analysis for that intent.
    3) Ask the LLM to format a hybrid-style, human-readable answer.
    """
    if df is None:
        return "Dataset not loaded."

    # Layer 1: Intent classification
    intent = classify_intent(question)

    # Layer 2: Deterministic analytics
    result = run_analysis_for_intent(intent, question, df)

    # Layer 3: Presentation
    final_answer = build_final_answer(question, result)
    return final_answer


def build_agent(df, system_text):
    """Kept for compatibility with existing app structure."""
    return df
