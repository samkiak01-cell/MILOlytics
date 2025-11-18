# logic/agent.py

import json
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Union, IO, Any
from langchain_openai import ChatOpenAI


# ===================================================
# LLM
# ===================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)



# ===================================================
# LOAD EXCEL
# ===================================================

def load_excel(source: Union[str, Path, IO[bytes]]) -> pd.DataFrame:
    """
    Load Excel with a sheet named ANYTHING.
    Return a single DataFrame.
    """
    xls = pd.ExcelFile(source)
    first_sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, first_sheet)
    return df



# ===================================================
# COLUMN SEMANTICS
# ===================================================

COLUMN_SEMANTICS = """
Dataset schema:

- Ticket_Number: unique ID
- Subject: category of issue
- Description: short issue summary
- Caller: who submitted the ticket
- Status: current state
- Created_Date: when ticket was created
- Due_Date: SLA deadline
- Resolved_Date: when ticket was resolved
- Resolved_By: agent who resolved
- Resolution_Time_Seconds: numeric duration between created/resolved
- SLA_Met: Yes/No — whether SLA was met

Rules:

- “Longest ticket” = max Resolution_Time_Seconds (ignoring nulls)
- “Fastest ticket” = min Resolution_Time_Seconds
- “Agents fastest/slowest” = group by Resolved_By, compute avg Resolution_Time_Seconds
- SLA questions use SLA_Met or Due_Date vs Resolved_Date
- Trend questions = analyze Created_Date, Resolved_Date, SLA over time
- Summary questions = general insights
"""



# ===================================================
# INTENT CLASSIFICATION
# ===================================================

INTENT_PROMPT = """
Classify the intent of the question into one category:

- overview
- basic_stats
- subject_analysis
- caller_volume
- longest
- fastest
- agent_speed
- agent_slow
- high_callers
- sla_overview
- sla_agents_best
- sla_agents_worst
- slowest_subjects
- outliers
- patterns
- recommendations
- exec_summary
- trend
- forecast
- other

Respond with ONLY the label. Nothing else.
"""

ALLOWED_INTENTS = {
    "overview", "basic_stats", "subject_analysis", "caller_volume",
    "longest", "fastest", "agent_speed", "agent_slow",
    "high_callers", "sla_overview", "sla_agents_best", "sla_agents_worst",
    "slowest_subjects", "outliers", "patterns", "recommendations",
    "exec_summary", "trend", "forecast", "other"
}

def classify_intent(question: str) -> str:
    q = question.lower()

    # HARD WIRES
    if any(k in q for k in ["pattern", "patterns", "trend", "trends"]):
        return "trend"

    if any(k in q for k in ["forecast", "predict", "projection", "future"]):
        return "forecast"

    # LLM fallback
    msg = f"{INTENT_PROMPT}\n\nQuestion:\n{question}\n\nIntent label:"
    intent = llm.invoke(msg).content.strip().lower()
    if intent not in ALLOWED_INTENTS:
        return "other"
    return intent



# ===================================================
# EXECUTION HELPER
# ===================================================

def safe_mean(series):
    series = series.dropna()
    return series.mean() if len(series) > 0 else None



# ===================================================
# ANALYSIS FUNCTIONS
# ===================================================

def analyze_overview(df):
    return {
        "total": len(df),
        "avg_resolution": safe_mean(df["Resolution_Time_Seconds"]),
        "sla_rate": (df["SLA_Met"].astype(str).str.lower() == "yes").mean()
    }

def analyze_subjects(df):
    return df["Subject"].value_counts().to_dict()

def analyze_callers(df):
    return df["Caller"].value_counts().to_dict()

def analyze_longest(df):
    df2 = df.dropna(subset=["Resolution_Time_Seconds"])
    if df2.empty: return None
    max_val = df2["Resolution_Time_Seconds"].max()
    rows = df2[df2["Resolution_Time_Seconds"] == max_val]
    return rows.to_dict(orient="records")

def analyze_fastest(df):
    df2 = df.dropna(subset=["Resolution_Time_Seconds"])
    if df2.empty: return None
    min_val = df2["Resolution_Time_Seconds"].min()
    rows = df2[df2["Resolution_Time_Seconds"] == min_val]
    return rows.to_dict(orient="records")

def analyze_agent_speed(df):
    df2 = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"])
    if df2.empty: return None
    grouped = df2.groupby("Resolved_By")["Resolution_Time_Seconds"].mean()
    fastest = grouped[grouped == grouped.min()]
    return fastest.to_dict()

def analyze_agent_slow(df):
    df2 = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"])
    if df2.empty: return None
    grouped = df2.groupby("Resolved_By")["Resolution_Time_Seconds"].mean()
    slowest = grouped[grouped == grouped.max()]
    return slowest.to_dict()

def analyze_high_callers(df):
    counts = df["Caller"].value_counts()
    threshold = counts.mean() + counts.std()
    return counts[counts > threshold].to_dict()

def analyze_sla(df):
    late = df[df["SLA_Met"].astype(str).str.lower() == "no"]
    return late.to_dict(orient="records")

def analyze_sla_best(df):
    df2 = df[df["SLA_Met"].astype(str).str.lower().isin(["yes","no"])]
    if df2.empty: return None
    grouped = df2.groupby("Resolved_By")["SLA_Met"].apply(lambda x: (x.str.lower()=="yes").mean())
    best = grouped[grouped == grouped.max()]
    return best.to_dict()

def analyze_sla_worst(df):
    df2 = df[df["SLA_Met"].astype(str).str.lower().isin(["yes","no"])]
    if df2.empty: return None
    grouped = df2.groupby("Resolved_By")["SLA_Met"].apply(lambda x: (x.str.lower()=="yes").mean())
    worst = grouped[grouped == grouped.min()]
    return worst.to_dict()

def analyze_slowest_subject(df):
    df2 = df.dropna(subset=["Resolution_Time_Seconds"])
    if df2.empty: return None
    g = df2.groupby("Subject")["Resolution_Time_Seconds"].mean()
    slow = g[g == g.max()]
    return slow.to_dict()

def analyze_outliers(df):
    df2 = df.dropna(subset=["Resolution_Time_Seconds"])
    if df2.empty: return None
    q1 = df2["Resolution_Time_Seconds"].quantile(0.25)
    q3 = df2["Resolution_Time_Seconds"].quantile(0.75)
    iqr = q3 - q1
    cutoff = q3 + 1.5 * iqr
    out = df2[df2["Resolution_Time_Seconds"] > cutoff]
    return out.to_dict(orient="records")

def analyze_patterns(df):
    return {
        "volume_over_time": df.groupby(df["Created_Date"].dt.date).size().to_dict(),
        "avg_resolution_over_time": df.groupby(df["Created_Date"].dt.date)["Resolution_Time_Seconds"].mean().to_dict(),
        "sla_over_time": df.groupby(df["Created_Date"].dt.date)["SLA_Met"].apply(lambda x: (x.astype(str).str.lower()=="yes").mean()).to_dict()
    }

def analyze_recommendations(df):
    slowdown_drivers = analyze_slowest_subject(df)
    high_callers = analyze_high_callers(df)
    return {"slow_subjects": slowdown_drivers, "high_callers": high_callers}

def analyze_exec(df):
    return {
        "summary": "High-level executive review.",
        "key_metrics": analyze_overview(df),
        "top_subjects": analyze_subjects(df),
        "bottlenecks": analyze_slowest_subject(df)
    }



# ===================================================
# PRESENTATION LAYER
# ===================================================

PRESENTATION_PROMPT = """
You are MILOlytics, an executive-quality analytics assistant.

Rules:
- DO NOT restate the user's question.
- Pick an appropriate header like:
  * 📊 Dataset Overview
  * ⏱️ Resolution Speed
  * 🧑‍💼 Agent Performance
  * 🚨 SLA Risks
  * 🧭 Executive Summary
- Keep answers visually clean.
- Use bullets, spacing, and icons.
- Convert seconds → hours → days automatically.
- Provide business-level clarity.
- Highlight anomalies or insights.
"""

def build_final_answer(result: dict) -> str:
    prompt = f"""
{PRESENTATION_PROMPT}

Here are the results to format:

{json.dumps(result, indent=2)}

Write the final formatted answer:
"""
    return llm.invoke(prompt).content.strip()



# ===================================================
# MAIN ENTRY
# ===================================================

def ask_question(df: pd.DataFrame, question: str, system_text: str = "") -> str:
    intent = classify_intent(question)

    if intent == "overview":
        result = analyze_overview(df)
    elif intent == "subject_analysis":
        result = analyze_subjects(df)
    elif intent == "caller_volume":
        result = analyze_callers(df)
    elif intent == "longest":
        result = analyze_longest(df)
    elif intent == "fastest":
        result = analyze_fastest(df)
    elif intent == "agent_speed":
        result = analyze_agent_speed(df)
    elif intent == "agent_slow":
        result = analyze_agent_slow(df)
    elif intent == "high_callers":
        result = analyze_high_callers(df)
    elif intent == "sla_overview":
        result = analyze_sla(df)
    elif intent == "sla_agents_best":
        result = analyze_sla_best(df)
    elif intent == "sla_agents_worst":
        result = analyze_sla_worst(df)
    elif intent == "slowest_subjects":
        result = analyze_slowest_subject(df)
    elif intent == "outliers":
        result = analyze_outliers(df)
    elif intent == "trend":
        result = analyze_patterns(df)
    elif intent == "recommendations":
        result = analyze_recommendations(df)
    elif intent == "exec_summary":
        result = analyze_exec(df)
    else:
        result = {"info": "No matching intent"}

    return build_final_answer(result)
