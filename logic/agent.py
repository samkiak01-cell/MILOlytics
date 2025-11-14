import pandas as pd
from langchain_openai import ChatOpenAI
import json
import numpy as np
import re



# ======================================================
# Data Cleaning + Standardization
# ======================================================

def clean_dataframe(df):
    """Ensures datetime, numeric, and name fields are standardized."""
    datetime_cols = [
        "Created_Date_Time",
        "Updated_Date_Time",
        "Due_Date_Time",
        "Resolution_Date_Time",
    ]

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Resolution_Time_Seconds" in df.columns:
        df["Resolution_Time_Seconds"] = pd.to_numeric(
            df["Resolution_Time_Seconds"], errors="coerce"
        )

    name_cols = ["Caller", "Created_By", "Updated_By", "Resolved_By"]

    for col in name_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                  .astype(str)
                  .str.strip()
                  .replace({"nan": None, "None": None, "": None})
            )

    if "Description" in df.columns:
        df["Description"] = df["Description"].astype(str).fillna("")

    return df



# ======================================================
# Load Excel
# ======================================================

def load_excel(source):
    """Load dataset + system text + sample questions."""
    df_data = pd.read_excel(source, sheet_name=0)
    system_text = str(pd.read_excel(source, sheet_name=1).iloc[0, 0])
    df_questions = pd.read_excel(source, sheet_name=2)

    df_data = clean_dataframe(df_data)
    sample_questions = df_questions.iloc[:, 0].dropna().tolist()

    return df_data, system_text, sample_questions



# ======================================================
# Chat model
# ======================================================

def build_agent(df, system_prompt):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return {"llm": llm, "df": df, "system": system_prompt}



# ======================================================
# JSON extraction helpers (VERY IMPORTANT FIX)
# ======================================================

def extract_json(text):
    """
    Extracts the FIRST valid JSON object from an LLM response,
    even if extra text is around it.
    """

    # find the first {...} block
    match = re.search(r"{.*}", text, flags=re.DOTALL)
    if match:
        json_str = match.group(0)

        try:
            return json.loads(json_str)
        except:
            pass  # fallthrough to next fix attempt

    # Try to repair JSON with GPT
    return None



def repair_json_with_llm(llm, broken_text):
    """Ask GPT to fix malformed JSON."""
    fix_prompt = f"""
Your job is to FIX the malformed JSON below and return VALID JSON only.

Malformed JSON:
{broken_text}

Return ONLY valid JSON. Nothing else.
"""

    fixed = llm.invoke(fix_prompt).content

    try:
        match = re.search(r"{.*}", fixed, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        return None

    return None



# ======================================================
# MAIN ASK FUNCTION
# ======================================================

def ask_question(agent, question, system_prompt, df):
    df = clean_dataframe(df)
    llm = agent["llm"]

    # --------------------------------------------
    # 1. LLM interpret question → JSON instructions
    # --------------------------------------------

    interpretation_prompt = f"""
You are an analytics interpreter.

Given the question:
"{question}"

And these columns:
{list(df.columns)}

Return ONLY a JSON dictionary with keys:
- operation: one of ["max","min","average","count","groupby","top","bottom"]
- column: column name to use
- groupby: column name or null
- filters: dict of filters or null
- return_multiple: true/false
- natural_language_goal: short description of desired final answer

Respond with JSON ONLY.
"""

    raw_response = llm.invoke(interpretation_prompt).content

    # Try extracting JSON
    parsed_json = extract_json(raw_response)

    if parsed_json is None:
        parsed_json = repair_json_with_llm(llm, raw_response)

    if parsed_json is None:
        return "I couldn't interpret your request due to invalid JSON."

    # Extract fields safely
    operation = parsed_json.get("operation")
    col = parsed_json.get("column")
    group = parsed_json.get("groupby")
    filters = parsed_json.get("filters")
    nl_goal = parsed_json.get("natural_language_goal", "")
    multi = parsed_json.get("return_multiple", False)

    # --------------------------------------------
    # 2. Apply filters
    # --------------------------------------------

    working = df.copy()

    if isinstance(filters, dict):
        for k, v in filters.items():
            if k in working.columns:
                working = working[working[k] == v]

    if working.empty:
        return "No records matched that question."

    # --------------------------------------------
    # 3. Perform computation
    # --------------------------------------------

    result = None

    try:
        if group and group in working.columns and col in working.columns:
            grouped = working.groupby(group)[col]

            # numeric enforcement
            try:
                grouped_vals = grouped.mean()
            except:
                grouped_vals = grouped.apply(lambda x: x)

            if operation == "max":
                max_val = grouped_vals.max()
                result = grouped_vals[grouped_vals == max_val]

            elif operation == "min":
                min_val = grouped_vals.min()
                result = grouped_vals[grouped_vals == min_val]

            else:
                result = grouped_vals

        elif col in working.columns:
            series = working[col]

            try:
                series = pd.to_numeric(series, errors="coerce")
            except:
                pass

            if operation == "max":
                val = series.max()
                result = working[working[col] == val]

            elif operation == "min":
                val = series.min()
                result = working[working[col] == val]

            elif operation == "average":
                result = series.mean()

            elif operation == "count":
                result = len(series)

            else:
                result = series

    except Exception as e:
        return f"An error occurred performing the calculation: {e}"

    # --------------------------------------------
    # 4. Summarize final answer with LLM
    # --------------------------------------------

    summary_prompt = f"""
You are a data explanation AI.

User question: "{question}"

Python result:
{str(result)}

Write a clear answer and a SHORT explanation.
If there are ties, list them all.
Convert seconds into hours/days when helpful.

Format:
Answer: <short answer>
Explanation: <why this is the answer>
"""

    final = llm.invoke(summary_prompt).content
    return final
