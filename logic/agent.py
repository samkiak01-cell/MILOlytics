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
    Load Demo Data Excel file.

    Rules:
    - ALWAYS load the FIRST SHEET as the main ticket dataset.
    - System Prompt sheet is optional.
    - Questions sheet is optional.
    """

    xls = pd.ExcelFile(source)
    sheets = xls.sheet_names

    # Always load the first sheet for DEMO DATA
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



# ===================================================
# PREPROCESS DEMO DATA
# ===================================================

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize Demo Data schema.

    Expected columns:
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

    # Normalize column names from variations
    rename_map = {
        "Ticket Number": "Ticket_Number",
        "Created Date": "Created_Date",
        "Due Date": "Due_Date",
        "Resolved Date": "Resolved_Date",
        "Resolved By": "Resolved_By",
        "Resolution Time Seconds": "Resolution_Time_Seconds",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Ensure required columns exist
    required_cols = [
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
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Dates
    for col in ["Created_Date", "Due_Date", "Resolved_Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric
    df["Resolution_Time_Seconds"] = pd.to_numeric(df["Resolution_Time_Seconds"], errors="coerce")

    # Clean text
    for col in ["Ticket_Number", "Subject", "Description", "Caller", "Status", "Resolved_By", "SLA_Met"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": None, "None": None, "": None})
        )

    # Add normalized subject buckets
    df["Subject_Normalized"] = df["Subject"].apply(normalize_subject)

    return df



# ===================================================
# SUBJECT NORMALIZATION (MERGED SUBJECT BUCKETS)
# ===================================================

def normalize_subject(subject: Any) -> str:
    """
    Collapse similar subjects into merged umbrella categories.
    """

    if not isinstance(subject, str):
        subject = "" if subject is None else str(subject)

    s = subject.lower()

    # 1. Login Issues
    if any(k in s for k in ["login", "log in", "signin", "auth", "authentication", "credential", "unable to login"]):
        return "Login & Authentication Issues"

    # 2. Password Reset
    if any(k in s for k in ["password", "pwd", "passcode", "reset password"]):
        return "Password Reset Issues"

    # 3. Access / Permissions
    if any(k in s for k in ["access", "permission", "role", "privilege", "unauthorized"]):
        return "Access / Permission Issues"

    # 4. Hardware
    if any(k in s for k in ["printer", "device", "laptop", "keyboard", "phone", "monitor", "hardware"]):
        return "Hardware Issues"

    # 5. Data / Records
    if any(k in s for k in ["data", "record", "field", "value", "incorrect", "mismatch", "update"]):
        return "Data / Record Issues"

    # 6. Software / Application errors
    if any(k in s for k in ["error", "bug", "crash", "system", "app", "not working", "failure"]):
        return "Software / Application Errors"

    # Default
    return "General Inquiry / Other"



# ===================================================
# COLUMN SEMANTICS (FOR LLM REASONING)
# ===================================================

COLUMN_SEMANTICS = """
This dataset represents call center support tickets.

Columns:
- Ticket_Number: ID of the ticket.
- Subject: short title summarizing the issue.
- Description: detailed description.
- Caller: person who reported the ticket.
- Status: Open / In Progress / Resolved.
- Created_Date: creation date.
- Due_Date: SLA target date.
- Resolved_Date: actual resolution date.
- Resolved_By: person who resolved the ticket.
- Resolution_Time_Seconds: time between Created and Resolved.
- SLA_Met: "Yes" or "No".
- Subject_Normalized: merged category created from Subject.

Semantic rules:
- "Who calls the most?" → Caller.
- "Fastest/slowest agent" → group by Resolved_By.
- Longest/shortest ticket → Resolution_Time_Seconds.
- SLA → SLA_Met or compare Resolved_Date vs Due_Date.
- Most common issue type → Subject_Normalized.
"""



# ===================================================
# CODE GENERATION PROMPT
# ===================================================

CODE_PROMPT = """
You are a senior Python analyst generating Pandas code.

Rules:
- You MUST output ONLY Python code.
- The DataFrame is named `df`.
- Compute the answer using correct columns based on semantics.
- Must end with:
    result = { "answer": ..., "details": ..., "mode": ... }

Ties:
- If multiple agents, callers, or tickets share the same extreme value,
  include ALL of them in both answer and details.

Use:
- Caller → ticket volume
- Resolved_By → performance (avg resolution time)
- Resolution_Time_Seconds → longest/shortest tickets
- Subject_Normalized → issue categories

If you cannot compute the answer:
    result = { "answer": "I don't know", "details": {"reason": "…"}, "mode": "error" }
"""



# ===================================================
# EXECUTIVE INSIGHT MODE (HIGH-LEVEL STORY)
# ===================================================

INSIGHT_KEYWORDS = [
    "insight", "overview", "summary", "summarize",
    "how are we doing", "overall", "big picture", "performance",
    "report", "diagnose", "story"
]


def is_insight_request(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in INSIGHT_KEYWORDS)



def compute_insight_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    metrics = {}

    metrics["total_tickets"] = int(len(df))

    # Resolution stats
    rt = df["Resolution_Time_Seconds"].dropna()
    if len(rt) > 0:
        metrics["avg_resolution"] = float(rt.mean())
        metrics["min_resolution"] = float(rt.min())
        metrics["max_resolution"] = float(rt.max())

        # Fastest and slowest tickets
        idx_fast = rt.idxmin()
        idx_slow = rt.idxmax()

        metrics["fastest_ticket"] = {
            "Ticket_Number": df.loc[idx_fast, "Ticket_Number"],
            "Subject": df.loc[idx_fast, "Subject"],
            "seconds": float(rt.loc[idx_fast])
        }
        metrics["slowest_ticket"] = {
            "Ticket_Number": df.loc[idx_slow, "Ticket_Number"],
            "Subject": df.loc[idx_slow, "Subject"],
            "seconds": float(rt.loc[idx_slow])
        }

    # SLA
    if "SLA_Met" in df.columns:
        sla = df["SLA_Met"].dropna().str.lower()
        metrics["sla_met"] = int((sla == "yes").sum())
        metrics["sla_not_met"] = int((sla == "no").sum())
        if len(sla) > 0:
            metrics["sla_rate"] = float(metrics["sla_met"] / len(sla) * 100)

    # Agents
    if "Resolved_By" in df.columns:
        df_agents = df.dropna(subset=["Resolved_By", "Resolution_Time_Seconds"])
        if len(df_agents) > 0:
            grouped = df_agents.groupby("Resolved_By")["Resolution_Time_Seconds"].mean()
            min_val = grouped.min()
            max_val = grouped.max()

            metrics["fastest_agents"] = [{"name": i, "avg_seconds": float(v)} for i, v in grouped.items() if v == min_val]
            metrics["slowest_agents"] = [{"name": i, "avg_seconds": float(v)} for i, v in grouped.items() if v == max_val]

    # Caller volume
    vc = df["Caller"].dropna().value_counts()
    metrics["top_callers"] = [{"caller": i, "count": int(c)} for i, c in vc.head(3).items()]

    # Category volume
    vc2 = df["Subject_Normalized"].value_counts()
    metrics["top_categories"] = [{"category": i, "count": int(c)} for i, c in vc2.items()]

    return metrics



INSIGHT_PROMPT = """
You are an executive analytics assistant for myBasePay.

Write a structured, executive-level summary of the program’s performance using ONLY the metrics provided.
Be specific, mention names, categories, agents, ticket numbers, and real counts.

Format:

Executive Summary:
- 1–2 sentences summarizing performance.

What’s Going Well:
- 2–4 bullets.

What Needs Attention:
- 2–4 bullets.

Opportunities:
- 2–4 bullets.

No filler. Be concrete. Use the numbers directly.
"""



def build_executive_insights(question: str, df: pd.DataFrame) -> str:
    metrics = compute_insight_metrics(df)
    prompt = INSIGHT_PROMPT + f"\n\nMetrics:\n{json.dumps(metrics, indent=2)}"
    return llm.invoke(prompt).content.strip()



# ===================================================
# CODE GENERATION + EXECUTION (NORMAL QUESTIONS)
# ===================================================

def generate_code(question: str, df: pd.DataFrame) -> str:
    cols = list(df.columns)

    prompt = f"""
{CODE_PROMPT}

Column Semantics:
{COLUMN_SEMANTICS}

Columns: {cols}

User question:
{question}

Write ONLY executable Python code. No markdown.
"""
    code = llm.invoke(prompt).content.strip()
    code = code.replace("```python", "").replace("```", "")
    return code



def execute_code(df: pd.DataFrame, code: str) -> Any:
    local_scope = {"df": df.copy(), "result": None}
    try:
        exec(code, {}, local_scope)
        return local_scope["result"]
    except Exception as e:
        return {
            "answer": "I don't know",
            "details": {"reason": str(e)},
            "mode": "error"
        }



# ===================================================
# FINAL ANSWER GENERATION (NATURAL LANGUAGE)
# ===================================================

FINAL_ANSWER_PROMPT = """
You are a data assistant for myBasePay.

Given:
- User question
- A structured `result` dict with fields: answer, details, mode
- Column meanings

Write a polished, human-readable final answer.
Start with the answer directly.
Include 1–3 short bullets if useful.
Convert seconds to hours/days when helpful.
Do NOT mention the raw Python result.
Tone: professional, concise, business-friendly.
"""



def build_final_answer(question: str, result: Dict, system_text: str) -> str:
    prompt = FINAL_ANSWER_PROMPT + f"""

User Question:
{question}

Result:
{json.dumps(result, indent=2)}

Column Info:
{system_text}
"""
    return llm.invoke(prompt).content.strip()



# ===================================================
# MASTER ENTRYPOINT USED BY STREAMLIT
# ===================================================

def ask_question(agent_unused, question: str, system_text: str, df=None) -> str:
    if df is None:
        return "Dataset not loaded."

    # Executive summary requests
    if is_insight_request(question):
        return build_executive_insights(question, df)

    # Normal analytical question
    code = generate_code(question, df)
    result = execute_code(df, code)

    if not isinstance(result, dict):
        result = {
            "answer": str(result),
            "details": {"note": "Result was not a dict"},
            "mode": "raw"
        }

    return build_final_answer(question, result, system_text)



def build_agent(df, system_text):
    return df
