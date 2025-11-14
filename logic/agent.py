import pandas as pd
from langchain_openai import ChatOpenAI
import json
import numpy as np


# ======================================================
# Data Cleaning + Standardization
# ======================================================

def clean_dataframe(df):
    """
    Ensures all date, numeric, and text columns are standardized.
    Prevents LLM from misreading blank strings as null values.
    """

    # Convert datetimes
    datetime_cols = [
        "Created_Date_Time",
        "Updated_Date_Time",
        "Due_Date_Time",
        "Resolution_Date_Time",
    ]

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert numeric columns
    if "Resolution_Time_Seconds" in df.columns:
        df["Resolution_Time_Seconds"] = pd.to_numeric(
            df["Resolution_Time_Seconds"], errors="coerce"
        )

    # Normalize agent columns
    name_cols = ["Caller", "Created_By", "Updated_By", "Resolved_By"]

    for col in name_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": None, "None": None, "": None})
            )

    # Ensure Description is string
    if "Description" in df.columns:
        df["Description"] = df["Description"].astype(str).fillna("")

    return df


# ======================================================
# Load Excel
# ======================================================

def load_excel(source):
    """
    Expects Excel with:
      Sheet 1 → Data
      Sheet 2 → System Prompt
      Sheet 3 → Sample Questions
    """
    df_data = pd.read_excel(source, sheet_name=0)
    system_text = str(pd.read_excel(source, sheet_name=1).iloc[0, 0])
    df_questions = pd.read_excel(source, sheet_name=2)

    # Clean data before anything else
    df_data = clean_dataframe(df_data)

    sample_questions = df_questions.iloc[:, 0].dropna().tolist()
    return df_data, system_text, sample_questions


# ======================================================
# Build LLM Agent
# ======================================================

def build_agent(df, system_prompt):
    """
    Only builds the LLM model.
    Data handling is separate so it stays clean & consistent.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return {"llm": llm, "df": df, "system": system_prompt}


# ======================================================
# LLM-Driven Analytics Engine
# ======================================================

def ask_question(agent, question, system_prompt, df):
    """
    Clean, universal answering function.
    - LLM interprets the question
    - LLM decides what fields to use
    - Python computes the actual numbers
    - LLM writes a clean final answer + explanation
    """

    df = clean_dataframe(df)  # always use cleaned data

    llm = agent["llm"]

    # First stage: let the LLM parse the question
    interpretation_prompt = f"""
You are an analytics interpreter. 
You must read the user's question and output JSON telling Python what to compute.

Available columns:
{list(df.columns)}

For the question: "{question}"

Return a JSON dictionary with:
- "operation": one of ["max", "min", "average", "count", "groupby", "match", "filter", "top", "bottom"]
- "column": which column(s) the operation should apply to
- "groupby": which column to group by (or null)
- "filters": conditions needed (or null)
- "return_multiple": true if more than one answer is possible
- "natural_language_goal": short text describing what final answer should look like

Return ONLY JSON.
"""

    try:
        parsed = llm.invoke(interpretation_prompt).content
        parsed_json = json.loads(parsed)
    except:
        return "I couldn't interpret your request."

    operation = parsed_json.get("operation")
    col = parsed_json.get("column")
    group = parsed_json.get("groupby")
    filters = parsed_json.get("filters")
    multi = parsed_json.get("return_multiple", False)
    nl_goal = parsed_json.get("natural_language_goal", "")

    # ====================================================
    # APPLY FILTERS
    # ====================================================
    working_df = df.copy()

    if isinstance(filters, dict):
        for k, v in filters.items():
            if k in working_df.columns:
                working_df = working_df[working_df[k] == v]

    if len(working_df) == 0:
        return "No matching records were found to answer your question."

    # ====================================================
    # CORE COMPUTATION
    # ====================================================

    result = None

    # -------- Grouped operations --------
    if group and group in working_df.columns and col in working_df.columns:
        grouped = working_df.groupby(group)[col]

        # Safety: convert to numeric if possible
        try:
            grouped_numeric = grouped.mean()
        except:
            grouped_numeric = grouped.apply(lambda x: x)

        if operation == "max":
            max_val = grouped_numeric.max()
            result = grouped_numeric[grouped_numeric == max_val]

        elif operation == "min":
            min_val = grouped_numeric.min()
            result = grouped_numeric[grouped_numeric == min_val]

        else:
            result = grouped_numeric

    # -------- Simple operations --------
    elif col in working_df.columns:

        series = working_df[col]

        # convert numbers if needed
        try:
            series = pd.to_numeric(series, errors="coerce")
        except:
            pass

        if operation == "max":
            max_val = series.max()
            result = working_df[working_df[col] == max_val]

        elif operation == "min":
            min_val = series.min()
            result = working_df[working_df[col] == min_val]

        elif operation == "average":
            result = series.mean()

        elif operation == "count":
            result = len(series)

        else:
            result = series

    # ====================================================
    # FORMAT FOR LLM OUTPUT
    # ====================================================

    summary_prompt = f"""
You are a data explanation AI for myBasePay.

The user asked: "{question}"

Here is the computed raw result (Python output):

{str(result)}

Write a clear answer followed by a very short explanation.
Keep the tone professional but simple.
If more than one person/ticket is tied, list all of them.
If numeric values are included, convert seconds to hours/days when appropriate.

Answer format:
Answer: <short answer>
Explanation: <why this is the answer>
"""

    final_answer = llm.invoke(summary_prompt).content
    return final_answer
