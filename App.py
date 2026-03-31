import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# Streamlit App Config
# ============================================================
st.set_page_config(
    page_title="SPC Dashboard - Single Upload Workflow",
    layout="wide"
)

st.title("SPC Dashboard")
with st.expander("About this app & how to use it", expanded=False):
    st.markdown(
        """
This app lets you upload **one dataset** (`CSV`, `XLSX`, or `XLS`) and then manually map the columns needed for the appropriate **Statistical Process Control (SPC)** chart.

### Supported SPC chart families

#### Continuous charts
- **I-MR**: continuous data with subgroup size = 1
- **X̄-R**: continuous data with subgroup sizes 2–10
- **X̄-S**: continuous data with subgroup sizes 11–25

#### Attribute charts
- **np**: count of defectives with **constant** sample size
- **p**: proportion defective with **varying or constant** sample size
- **c**: count of defects with **constant** inspection area / opportunity
- **u**: defects per unit with **varying or constant** inspection area / opportunity

### Workflow
1. Upload one **CSV** or **Excel** file.
2. If Excel, select the **sheet** to use.
3. The app checks for invalid columns (blank headers, duplicates, empty columns).
4. Choose the dataset structure:
   - **Continuous**
   - **Defectives**
   - **Defects**
5. Choose how to handle empty rows / missing entries:
   - **Replace empty entries with zeros**
   - **Disregard rows with empty required fields**
6. Map the relevant columns (time, value, subgroup, denominators as applicable).
7. The app automatically enables only **valid SPC charts** for your selected data.
8. Select the **time period** for the main SPC chart.
9. Choose whether the selected period should be broken down into **Yearly** or **Quarterly** charts.
10. The remaining SPC logic (limits, rules, visuals) is applied unchanged.
"""
    )

# ============================================================
# Constants for Control Charts
# ============================================================
A2 = {
    2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577,
    6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308
}

D3 = {
    2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000,
    6: 0.000, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223
}

D4 = {
    2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114,
    6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777
}

A3 = {
    2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287,
    7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975, 11: 0.927,
    12: 0.886, 13: 0.850, 14: 0.817, 15: 0.789, 16: 0.763,
    17: 0.739, 18: 0.718, 19: 0.698, 20: 0.680, 21: 0.663,
    22: 0.647, 23: 0.633, 24: 0.619, 25: 0.606
}

B3 = {
    2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.030,
    7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284, 11: 0.321,
    12: 0.354, 13: 0.382, 14: 0.406, 15: 0.428, 16: 0.448,
    17: 0.466, 18: 0.482, 19: 0.497, 20: 0.510, 21: 0.523,
    22: 0.534, 23: 0.545, 24: 0.555, 25: 0.565
}

B4 = {
    2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970,
    7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716, 11: 1.679,
    12: 1.646, 13: 1.618, 14: 1.594, 15: 1.572, 16: 1.552,
    17: 1.534, 18: 1.518, 19: 1.503, 20: 1.490, 21: 1.477,
    22: 1.466, 23: 1.455, 24: 1.445, 25: 1.435
}

# ============================================================
# Rule Styling
# ============================================================
RULE_STYLE_MAP = {
    "Rule 1": {"color": "#2E8B57", "label": "Rule 1"},
    "Rule 2": {"color": "#C71585", "label": "Rule 2"},
    "Rule 3": {"color": "#7D3C98", "label": "Rule 3"},
    "Rule 4": {"color": "#A0522D", "label": "Rule 4"},
    "Rule 5": {"color": "#D35400", "label": "Rule 5"},
    "Rule 6": {"color": "#B8860B", "label": "Rule 6"},
    "Rule 7": {"color": "#008B8B", "label": "Rule 7"},
    "Rule 8": {"color": "#8B0000", "label": "Rule 8"},
    "Secondary chart: point beyond control limit": {
        "color": "#4B4B4B",
        "label": "Secondary limit breach"
    },
    "Multiple rules": {"color": "#FF0000", "label": "Multiple rules"}
}

DEFAULT_RULE_STYLE = {"color": "#333333", "label": "Special cause"}

RULE_DISPLAY_TEXT = {
    "Rule 1": "One point is more than 3 standard deviations from the mean.",
    "Rule 2": "Nine (or more) points in a row are on the same side of the mean.",
    "Rule 3": "Six (or more) points in a row are continually increasing (or decreasing).",
    "Rule 4": "Fourteen (or more) points in a row alternate in direction, increasing then decreasing.",
    "Rule 5": "Two (or three) out of three points in a row are more than 2 standard deviations from the mean in the same direction.",
    "Rule 6": "Four (or five) out of five points in a row are more than 1 standard deviation from the mean in the same direction.",
    "Rule 7": "Fifteen points in a row are all within 1 standard deviation of the mean on either side of the mean.",
    "Rule 8": "Eight points in a row exist, but none within 1 standard deviation of the mean, and the points are in both directions from the mean.",
    "Secondary chart: point beyond control limit": "A point on the secondary chart is beyond the control limit."
}

RULE_SORT_ORDER = {
    "Rule 1": 1,
    "Rule 2": 2,
    "Rule 3": 3,
    "Rule 4": 4,
    "Rule 5": 5,
    "Rule 6": 6,
    "Rule 7": 7,
    "Rule 8": 8,
    "Secondary chart: point beyond control limit": 9
}

# ============================================================
# General Helpers
# ============================================================
def is_constant_series(s: pd.Series, tol: float = 1e-12) -> bool:
    vals = pd.to_numeric(s, errors="coerce").dropna().astype(float).values
    if len(vals) <= 1:
        return True
    return np.nanmax(vals) - np.nanmin(vals) <= tol


def coerce_numeric_nonnegative_integer(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.isna().any():
        return False
    if (vals < 0).any():
        return False
    return np.allclose(vals, np.round(vals))


def ensure_required_columns(df: pd.DataFrame, required_cols: list):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def metric_value_or_range(line_arr):
    arr = np.asarray(line_arr, dtype=float)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return "N/A"
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if np.isclose(mn, mx):
        return f"{mn:.5f}"
    return f"{mn:.5f} to {mx:.5f}"


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() if c is not None else "" for c in df.columns]
    return df


def validate_uploaded_columns(df: pd.DataFrame):
    report = {
        "fatal_errors": [],
        "warnings": [],
        "all_null_columns": []
    }

    cols = list(df.columns)
    if len(cols) == 0:
        report["fatal_errors"].append("The uploaded file has no columns.")
        return report

    blank_cols = [c for c in cols if str(c).strip() == ""]
    if blank_cols:
        report["fatal_errors"].append(
            "The file contains blank column names. Please fix the headers and upload again."
        )

    normalized = [str(c).strip() for c in cols]
    duplicated_norm = pd.Series(normalized)
    duplicate_names = duplicated_norm[duplicated_norm.duplicated()].unique().tolist()
    if duplicate_names:
        report["fatal_errors"].append(
            f"Duplicate column names detected after trimming spaces: {duplicate_names}"
        )

    all_null_cols = [c for c in cols if df[c].isna().all()]
    if all_null_cols:
        report["warnings"].append(
            f"These columns are completely empty and should not be used: {all_null_cols}"
        )
        report["all_null_columns"] = all_null_cols

    return report


def get_excel_engine(file_name: str):
    ext = file_name.lower().split(".")[-1]
    if ext == "xlsx":
        return "openpyxl"
    elif ext == "xls":
        return "xlrd"
    return None


def load_uploaded_table(uploaded_file, selected_sheet=None):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        file_bytes = uploaded_file.getvalue()
        engine = get_excel_engine(uploaded_file.name)
        excel_buffer = io.BytesIO(file_bytes)

        if selected_sheet is None:
            xls = pd.ExcelFile(excel_buffer, engine=engine)
            return xls.sheet_names

        excel_buffer = io.BytesIO(file_bytes)
        df = pd.read_excel(excel_buffer, sheet_name=selected_sheet, engine=engine)
        return df

    raise ValueError("Unsupported file type. Please upload CSV, XLSX, or XLS.")


def get_candidate_columns(df: pd.DataFrame):
    all_cols = df.columns.tolist()
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
    non_empty_cols = [c for c in all_cols if not df[c].isna().all()]
    return all_cols, numeric_cols, non_empty_cols


def validate_distinct_column_selection(mapping: dict):
    used = []
    for _, v in mapping.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip().lower() == "-- none --":
            continue
        used.append(v)

    duplicates = pd.Series(used)
    duplicated_vals = duplicates[duplicates.duplicated()].unique().tolist()
    if duplicated_vals:
        raise ValueError(
            f"The same column was selected for multiple roles: {duplicated_vals}. Please choose distinct columns."
        )


def replace_blank_strings_with_nan(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].replace(r"^\s*$", np.nan, regex=True)
    return df


def apply_missing_value_strategy(source_df: pd.DataFrame, mode: str, mapping: dict, strategy: str):
    """
    strategy:
      - "zeros"
      - "drop"

    Returns:
      cleaned_df, notes_list
    """
    df = source_df.copy()

    time_col = mapping.get("time_col")
    selected_cols = [time_col]

    numeric_fill_cols = []
    text_fill_cols = []

    if mode == "Continuous":
        value_col = mapping.get("value_col")
        subgroup_col = mapping.get("subgroup_col")

        selected_cols.append(value_col)
        numeric_fill_cols.append(value_col)

        if subgroup_col and subgroup_col != "-- None --":
            selected_cols.append(subgroup_col)
            text_fill_cols.append(subgroup_col)

    elif mode == "Defectives":
        defectives_col = mapping.get("defectives_col")
        sample_size_col = mapping.get("sample_size_col")

        selected_cols.extend([defectives_col, sample_size_col])
        numeric_fill_cols.extend([defectives_col, sample_size_col])

    elif mode == "Defects":
        defects_col = mapping.get("defects_col")
        inspection_units_col = mapping.get("inspection_units_col")

        selected_cols.append(defects_col)
        numeric_fill_cols.append(defects_col)

        if inspection_units_col and inspection_units_col != "-- None --":
            selected_cols.append(inspection_units_col)
            numeric_fill_cols.append(inspection_units_col)

    selected_cols = [c for c in selected_cols if c is not None and c in df.columns]
    numeric_fill_cols = [c for c in numeric_fill_cols if c is not None and c in df.columns]
    text_fill_cols = [c for c in text_fill_cols if c is not None and c in df.columns]

    df = replace_blank_strings_with_nan(df, selected_cols)

    notes = []
    initial_rows = len(df)

    if strategy == "zeros":
        numeric_filled = 0
        text_filled = 0

        for c in numeric_fill_cols:
            missing_count = int(df[c].isna().sum())
            if missing_count > 0:
                df[c] = df[c].fillna(0)
                numeric_filled += missing_count

        for c in text_fill_cols:
            missing_count = int(df[c].isna().sum())
            if missing_count > 0:
                df[c] = df[c].fillna("0")
                text_filled += missing_count

        rows_missing_time = int(df[time_col].isna().sum()) if time_col in df.columns else 0
        if rows_missing_time > 0:
            df = df[df[time_col].notna()].copy()

        if numeric_filled > 0:
            notes.append(f"Missing numeric entries in the selected columns were replaced with zeros ({numeric_filled} cell(s)).")
        if text_filled > 0:
            notes.append(f"Missing subgroup entries were replaced with '0' ({text_filled} cell(s)).")
        if rows_missing_time > 0:
            notes.append(f"Rows with empty time values could not be zero-filled and were disregarded ({rows_missing_time} row(s)).")

        # Denominator cleanup for charts that require positive denominators
        if mode == "Defectives":
            sample_size_col = mapping.get("sample_size_col")
            if sample_size_col in df.columns:
                ss = pd.to_numeric(df[sample_size_col], errors="coerce")
                invalid_rows = int((ss <= 0).sum())
                if invalid_rows > 0:
                    df = df[ss > 0].copy()
                    notes.append(f"Rows with sample size less than or equal to zero were disregarded after zero-filling ({invalid_rows} row(s)).")

        if mode == "Defects":
            inspection_units_col = mapping.get("inspection_units_col")
            if inspection_units_col and inspection_units_col != "-- None --" and inspection_units_col in df.columns:
                uu = pd.to_numeric(df[inspection_units_col], errors="coerce")
                invalid_rows = int((uu <= 0).sum())
                if invalid_rows > 0:
                    df = df[uu > 0].copy()
                    notes.append(f"Rows with inspection units/opportunities less than or equal to zero were disregarded after zero-filling ({invalid_rows} row(s)).")

        if not notes:
            notes.append("No empty entries were found in the selected columns.")

    else:  # strategy == "drop"
        rows_with_missing_required = int(df[selected_cols].isna().any(axis=1).sum())
        if rows_with_missing_required > 0:
            df = df[~df[selected_cols].isna().any(axis=1)].copy()
            notes.append(f"Rows with empty required selected fields were disregarded ({rows_with_missing_required} row(s)).")
        else:
            notes.append("No empty rows were found in the selected columns.")

    final_rows = len(df)
    if final_rows != initial_rows:
        notes.append(f"Rows retained after empty-row handling: {final_rows} of {initial_rows}.")

    return df.reset_index(drop=True), notes


# ============================================================
# Column Mapping to Internal SPC Structure
# ============================================================
def build_raw_df_from_mapping(source_df: pd.DataFrame, mode: str, mapping: dict):
    """
    Convert user-selected columns into the internal structure expected by the SPC engine.

    Internal expected structure:
    - continuous: date, year, subgroup_id, value
    - defectives: date, year, subgroup_id, value, sample_size
    - defects: date, year, subgroup_id, value, inspection_units (optional)
    """
    validate_distinct_column_selection(mapping)

    df = source_df.copy()

    time_col = mapping.get("time_col")
    if time_col is None:
        raise ValueError("A time column must be selected.")

    if time_col not in df.columns:
        raise ValueError(f"Selected time column '{time_col}' was not found in the dataset.")

    df["date"] = pd.to_datetime(df[time_col], errors="coerce")
    invalid_time_rows = int(df["date"].isna().sum())
    if invalid_time_rows > 0:
        raise ValueError(
            f"The selected time column '{time_col}' contains {invalid_time_rows} invalid or unparseable date/time values."
        )

    if mode == "Continuous":
        value_col = mapping.get("value_col")
        subgroup_col = mapping.get("subgroup_col")

        if value_col is None or value_col not in df.columns:
            raise ValueError("A valid continuous measurement column must be selected.")

        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
        invalid_value_rows = int(df["value"].isna().sum())
        if invalid_value_rows > 0:
            raise ValueError(
                f"The selected continuous column '{value_col}' contains {invalid_value_rows} non-numeric or missing values."
            )

        if subgroup_col is not None and subgroup_col != "-- None --":
            if subgroup_col not in df.columns:
                raise ValueError(f"Selected subgroup column '{subgroup_col}' was not found.")
            df["subgroup_id"] = df[subgroup_col].astype(str)
        else:
            df = df.sort_values(["date"]).reset_index(drop=True)
            df["subgroup_id"] = df.groupby("date").cumcount() + 1

        raw_df = df[["date", "subgroup_id", "value"]].copy()
        raw_df["year"] = raw_df["date"].dt.year
        raw_df = raw_df.sort_values(["date", "subgroup_id"]).reset_index(drop=True)

        meta = {
            "kind": "continuous",
            "label": mapping.get("series_label", value_col),
            "source_columns": {
                "time": time_col,
                "value": value_col,
                "subgroup": None if subgroup_col == "-- None --" else subgroup_col
            }
        }
        return raw_df[["date", "year", "subgroup_id", "value"]], meta

    elif mode == "Defectives":
        defectives_col = mapping.get("defectives_col")
        sample_size_col = mapping.get("sample_size_col")

        if defectives_col is None or defectives_col not in df.columns:
            raise ValueError("A valid defectives count column must be selected.")
        if sample_size_col is None or sample_size_col not in df.columns:
            raise ValueError("A valid sample size column must be selected.")

        df["value"] = pd.to_numeric(df[defectives_col], errors="coerce")
        df["sample_size"] = pd.to_numeric(df[sample_size_col], errors="coerce")

        if df["value"].isna().any():
            raise ValueError(
                f"The defectives column '{defectives_col}' contains non-numeric or missing values."
            )
        if df["sample_size"].isna().any():
            raise ValueError(
                f"The sample size column '{sample_size_col}' contains non-numeric or missing values."
            )

        if not coerce_numeric_nonnegative_integer(df["value"]):
            raise ValueError("Defectives must be non-negative integers.")
        if not coerce_numeric_nonnegative_integer(df["sample_size"]):
            raise ValueError("Sample size must be positive integers.")
        if (df["sample_size"] <= 0).any():
            raise ValueError("Sample size must be greater than zero.")
        if (df["value"] > df["sample_size"]).any():
            raise ValueError("Defectives cannot exceed sample size.")

        df["subgroup_id"] = 1
        raw_df = df[["date", "subgroup_id", "value", "sample_size"]].copy()
        raw_df["year"] = raw_df["date"].dt.year
        raw_df = raw_df.sort_values(["date"]).reset_index(drop=True)

        meta = {
            "kind": "defectives",
            "label": mapping.get("series_label", defectives_col),
            "source_columns": {
                "time": time_col,
                "defectives": defectives_col,
                "sample_size": sample_size_col
            }
        }
        return raw_df[["date", "year", "subgroup_id", "value", "sample_size"]], meta

    elif mode == "Defects":
        defects_col = mapping.get("defects_col")
        inspection_units_col = mapping.get("inspection_units_col")

        if defects_col is None or defects_col not in df.columns:
            raise ValueError("A valid defects count column must be selected.")

        df["value"] = pd.to_numeric(df[defects_col], errors="coerce")
        if df["value"].isna().any():
            raise ValueError(
                f"The defects column '{defects_col}' contains non-numeric or missing values."
            )

        if not coerce_numeric_nonnegative_integer(df["value"]):
            raise ValueError("Defects must be non-negative integers.")

        if inspection_units_col is not None and inspection_units_col != "-- None --":
            if inspection_units_col not in df.columns:
                raise ValueError(f"Selected inspection/opportunity column '{inspection_units_col}' was not found.")
            df["inspection_units"] = pd.to_numeric(df[inspection_units_col], errors="coerce")
            if df["inspection_units"].isna().any():
                raise ValueError(
                    f"The inspection/opportunity column '{inspection_units_col}' contains non-numeric or missing values."
                )
            if (df["inspection_units"] <= 0).any():
                raise ValueError("Inspection units / opportunities must be greater than zero.")

        df["subgroup_id"] = 1

        keep_cols = ["date", "subgroup_id", "value"]
        if "inspection_units" in df.columns:
            keep_cols.append("inspection_units")

        raw_df = df[keep_cols].copy()
        raw_df["year"] = raw_df["date"].dt.year
        raw_df = raw_df.sort_values(["date"]).reset_index(drop=True)

        meta = {
            "kind": "defects",
            "label": mapping.get("series_label", defects_col),
            "source_columns": {
                "time": time_col,
                "defects": defects_col,
                "inspection_units": None if inspection_units_col == "-- None --" else inspection_units_col
            }
        }

        final_cols = ["date", "year", "subgroup_id", "value"]
        if "inspection_units" in raw_df.columns:
            final_cols.append("inspection_units")

        return raw_df[final_cols], meta

    else:
        raise ValueError(f"Unsupported mode: {mode}")


# ============================================================
# Automatic Chart Detection
# ============================================================
def get_valid_chart_options_for_continuous(subgroup_size: int):
    if subgroup_size == 1:
        return ["I-MR"]
    elif 2 <= subgroup_size <= 10:
        return ["Xbar-R"]
    elif 11 <= subgroup_size <= 25:
        return ["Xbar-S"]
    else:
        raise ValueError("Continuous subgroup size must be between 1 and 25.")


def detect_chart_options_for_variable(raw_df: pd.DataFrame, meta: dict) -> list:
    kind = meta["kind"]

    if kind == "continuous":
        subgroup_sizes = raw_df.groupby("date")["value"].size()
        if subgroup_sizes.empty:
            return []
        if subgroup_sizes.nunique() == 1:
            subgroup_size = int(subgroup_sizes.iloc[0])
        else:
            subgroup_size = int(round(subgroup_sizes.median()))

        try:
            return get_valid_chart_options_for_continuous(subgroup_size)
        except Exception:
            return []

    elif kind == "defectives":
        if "sample_size" not in raw_df.columns:
            return []

        if not coerce_numeric_nonnegative_integer(raw_df["value"]):
            return []

        n_vals = pd.to_numeric(raw_df["sample_size"], errors="coerce")
        if n_vals.isna().any() or (n_vals <= 0).any():
            return []

        if (pd.to_numeric(raw_df["value"], errors="coerce") > n_vals).any():
            return []

        if is_constant_series(n_vals):
            return ["np", "p"]
        return ["p"]

    elif kind == "defects":
        if not coerce_numeric_nonnegative_integer(raw_df["value"]):
            return []

        if "inspection_units" in raw_df.columns:
            units = pd.to_numeric(raw_df["inspection_units"], errors="coerce")
            if units.isna().any() or (units <= 0).any():
                return []
            if is_constant_series(units):
                return ["c", "u"]
            return ["u"]

        return ["c"]

    return []


def detect_subgroup_size_for_display(raw_df: pd.DataFrame, meta: dict):
    if meta["kind"] != "continuous":
        return None
    subgroup_sizes = raw_df.groupby("date")["value"].size()
    if subgroup_sizes.empty:
        return None
    if subgroup_sizes.nunique() == 1:
        return int(subgroup_sizes.iloc[0])
    return int(round(subgroup_sizes.median()))


def describe_variable_type(meta: dict, raw_df: pd.DataFrame) -> str:
    if meta["kind"] == "continuous":
        subgroup_size = detect_subgroup_size_for_display(raw_df, meta)
        return f"Continuous measurement over time (subgroup size = {subgroup_size})"
    elif meta["kind"] == "defectives":
        n_vals = raw_df["sample_size"] if "sample_size" in raw_df.columns else pd.Series(dtype=float)
        n_desc = "constant sample size" if len(n_vals) > 0 and is_constant_series(n_vals) else "varying sample size"
        return f"Attribute data: count of defectives ({n_desc})"
    elif meta["kind"] == "defects":
        if "inspection_units" in raw_df.columns:
            u_desc = "constant opportunity" if is_constant_series(raw_df["inspection_units"]) else "varying opportunity"
            return f"Attribute data: count of defects ({u_desc})"
        return "Attribute data: count of defects (constant opportunity assumed)"
    return "Unknown"


# ============================================================
# Aggregation Helpers
# ============================================================
def aggregate_for_chart(raw_df: pd.DataFrame, chart_type: str):
    if chart_type == "I-MR":
        temp = (
            raw_df.groupby("date", as_index=False)
            .agg(value=("value", "mean"), year=("year", "first"))
            .sort_values("date")
            .reset_index(drop=True)
        )
        temp["MR"] = temp["value"].diff().abs()
        return temp

    elif chart_type == "Xbar-R":
        temp = (
            raw_df.groupby("date", as_index=False)
            .agg(
                xbar=("value", "mean"),
                R=("value", lambda x: np.max(x) - np.min(x)),
                year=("year", "first"),
                n=("value", "size")
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        return temp

    elif chart_type == "Xbar-S":
        temp = (
            raw_df.groupby("date", as_index=False)
            .agg(
                xbar=("value", "mean"),
                S=("value", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
                year=("year", "first"),
                n=("value", "size")
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        return temp

    elif chart_type in ["np", "p"]:
        ensure_required_columns(raw_df, ["date", "year", "value", "sample_size"])
        temp = (
            raw_df.groupby("date", as_index=False)
            .agg(
                count=("value", "sum"),
                n=("sample_size", "sum"),
                year=("year", "first")
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        temp["p"] = np.where(temp["n"] > 0, temp["count"] / temp["n"], np.nan)
        return temp

    elif chart_type in ["c", "u"]:
        ensure_required_columns(raw_df, ["date", "year", "value"])

        if "inspection_units" in raw_df.columns:
            temp = (
                raw_df.groupby("date", as_index=False)
                .agg(defects=("value", "sum"), units=("inspection_units", "sum"), year=("year", "first"))
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            temp = (
                raw_df.groupby("date", as_index=False)
                .agg(defects=("value", "sum"), year=("year", "first"))
                .sort_values("date")
                .reset_index(drop=True)
            )
            temp["units"] = 1.0

        temp["u"] = np.where(temp["units"] > 0, temp["defects"] / temp["units"], np.nan)
        return temp

    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")


# ============================================================
# Control Limit Calculation
# ============================================================
def _repeat_line(value, n):
    return np.repeat(float(value), n)


def calc_chart_limits(chart_df: pd.DataFrame, chart_type: str):
    n_points = len(chart_df)

    if chart_type == "I-MR":
        x = chart_df["value"].values
        mr = chart_df["MR"].dropna().values
        xbar = np.mean(x)
        mrbar = np.mean(mr) if len(mr) > 0 else 0.0
        sigma = mrbar / 1.128 if 1.128 != 0 else 0.0

        ucl = xbar + 3 * sigma
        lcl = max(0.0, xbar - 3 * sigma) if np.all(x >= 0) else xbar - 3 * sigma

        return {
            "primary": {
                "label": "Individuals",
                "y_col": "value",
                "CL": xbar,
                "UCL": ucl,
                "LCL": lcl,
                "CL_series": _repeat_line(xbar, n_points),
                "UCL_series": _repeat_line(ucl, n_points),
                "LCL_series": _repeat_line(lcl, n_points),
                "sigma": sigma,
                "sigma_series": _repeat_line(sigma, n_points)
            },
            "secondary": {
                "label": "Moving Range",
                "y_col": "MR",
                "CL": mrbar,
                "UCL": 3.267 * mrbar,
                "LCL": 0.0,
                "CL_series": _repeat_line(mrbar, n_points),
                "UCL_series": _repeat_line(3.267 * mrbar, n_points),
                "LCL_series": _repeat_line(0.0, n_points),
                "sigma": None,
                "sigma_series": None
            }
        }

    elif chart_type == "Xbar-R":
        subgroup_sizes = chart_df["n"].dropna().astype(int).unique().tolist()
        if len(subgroup_sizes) != 1:
            raise ValueError("Xbar-R requires constant subgroup size.")
        n = subgroup_sizes[0]
        if n not in A2:
            raise ValueError("Xbar-R is only supported for subgroup sizes 2 to 10.")

        xbarbar = chart_df["xbar"].mean()
        rbar = chart_df["R"].mean()
        a2 = A2[n]
        d3 = D3[n]
        d4 = D4[n]

        sigma_xbar = (a2 * rbar) / 3.0
        ucl = xbarbar + a2 * rbar
        lcl = xbarbar - a2 * rbar

        return {
            "primary": {
                "label": "Xbar",
                "y_col": "xbar",
                "CL": xbarbar,
                "UCL": ucl,
                "LCL": lcl,
                "CL_series": _repeat_line(xbarbar, n_points),
                "UCL_series": _repeat_line(ucl, n_points),
                "LCL_series": _repeat_line(lcl, n_points),
                "sigma": sigma_xbar,
                "sigma_series": _repeat_line(sigma_xbar, n_points)
            },
            "secondary": {
                "label": "Range",
                "y_col": "R",
                "CL": rbar,
                "UCL": d4 * rbar,
                "LCL": d3 * rbar,
                "CL_series": _repeat_line(rbar, n_points),
                "UCL_series": _repeat_line(d4 * rbar, n_points),
                "LCL_series": _repeat_line(d3 * rbar, n_points),
                "sigma": None,
                "sigma_series": None
            }
        }

    elif chart_type == "Xbar-S":
        subgroup_sizes = chart_df["n"].dropna().astype(int).unique().tolist()
        if len(subgroup_sizes) != 1:
            raise ValueError("Xbar-S requires constant subgroup size.")
        n = subgroup_sizes[0]
        if n not in A3:
            raise ValueError("Xbar-S is only supported for subgroup sizes 2 to 25.")

        xbarbar = chart_df["xbar"].mean()
        sbar = chart_df["S"].mean()
        a3 = A3[n]
        b3 = B3[n]
        b4 = B4[n]

        sigma_xbar = (a3 * sbar) / 3.0
        ucl = xbarbar + a3 * sbar
        lcl = xbarbar - a3 * sbar

        return {
            "primary": {
                "label": "Xbar",
                "y_col": "xbar",
                "CL": xbarbar,
                "UCL": ucl,
                "LCL": lcl,
                "CL_series": _repeat_line(xbarbar, n_points),
                "UCL_series": _repeat_line(ucl, n_points),
                "LCL_series": _repeat_line(lcl, n_points),
                "sigma": sigma_xbar,
                "sigma_series": _repeat_line(sigma_xbar, n_points)
            },
            "secondary": {
                "label": "Std Dev",
                "y_col": "S",
                "CL": sbar,
                "UCL": b4 * sbar,
                "LCL": b3 * sbar,
                "CL_series": _repeat_line(sbar, n_points),
                "UCL_series": _repeat_line(b4 * sbar, n_points),
                "LCL_series": _repeat_line(b3 * sbar, n_points),
                "sigma": None,
                "sigma_series": None
            }
        }

    elif chart_type == "np":
        subgroup_sizes = chart_df["n"].dropna().astype(float)
        if subgroup_sizes.empty or not is_constant_series(subgroup_sizes):
            raise ValueError("np chart requires constant sample size.")

        n_val = float(subgroup_sizes.iloc[0])
        pbar = chart_df["count"].sum() / chart_df["n"].sum()
        cl = n_val * pbar
        sigma = np.sqrt(n_val * pbar * (1 - pbar))
        ucl = cl + 3 * sigma
        lcl = max(0.0, cl - 3 * sigma)

        return {
            "primary": {
                "label": "Defectives (np)",
                "y_col": "count",
                "CL": cl,
                "UCL": ucl,
                "LCL": lcl,
                "CL_series": _repeat_line(cl, n_points),
                "UCL_series": _repeat_line(ucl, n_points),
                "LCL_series": _repeat_line(lcl, n_points),
                "sigma": sigma,
                "sigma_series": _repeat_line(sigma, n_points)
            },
            "secondary": None
        }

    elif chart_type == "p":
        pbar = chart_df["count"].sum() / chart_df["n"].sum()
        sigma_series = np.sqrt((pbar * (1 - pbar)) / chart_df["n"].astype(float).values)
        cl_series = _repeat_line(pbar, n_points)
        ucl_series = np.minimum(1.0, cl_series + 3 * sigma_series)
        lcl_series = np.maximum(0.0, cl_series - 3 * sigma_series)

        return {
            "primary": {
                "label": "Proportion Defective (p)",
                "y_col": "p",
                "CL": pbar,
                "UCL": float(np.nanmean(ucl_series)),
                "LCL": float(np.nanmean(lcl_series)),
                "CL_series": cl_series,
                "UCL_series": ucl_series,
                "LCL_series": lcl_series,
                "sigma": None,
                "sigma_series": sigma_series
            },
            "secondary": None
        }

    elif chart_type == "c":
        if "units" in chart_df.columns and not is_constant_series(chart_df["units"]):
            raise ValueError("c chart requires constant inspection opportunity.")
        cbar = chart_df["defects"].mean()
        sigma = np.sqrt(cbar)
        ucl = cbar + 3 * sigma
        lcl = max(0.0, cbar - 3 * sigma)

        return {
            "primary": {
                "label": "Defects (c)",
                "y_col": "defects",
                "CL": cbar,
                "UCL": ucl,
                "LCL": lcl,
                "CL_series": _repeat_line(cbar, n_points),
                "UCL_series": _repeat_line(ucl, n_points),
                "LCL_series": _repeat_line(lcl, n_points),
                "sigma": sigma,
                "sigma_series": _repeat_line(sigma, n_points)
            },
            "secondary": None
        }

    elif chart_type == "u":
        if "units" not in chart_df.columns:
            raise ValueError("u chart requires inspection units / opportunities.")
        ubar = chart_df["defects"].sum() / chart_df["units"].sum()
        sigma_series = np.sqrt(ubar / chart_df["units"].astype(float).values)
        cl_series = _repeat_line(ubar, n_points)
        ucl_series = np.maximum(0.0, cl_series + 3 * sigma_series)
        lcl_series = np.maximum(0.0, cl_series - 3 * sigma_series)

        return {
            "primary": {
                "label": "Defects per Unit (u)",
                "y_col": "u",
                "CL": ubar,
                "UCL": float(np.nanmean(ucl_series)),
                "LCL": float(np.nanmean(lcl_series)),
                "CL_series": cl_series,
                "UCL_series": ucl_series,
                "LCL_series": lcl_series,
                "sigma": None,
                "sigma_series": sigma_series
            },
            "secondary": None
        }

    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")


# ============================================================
# SPC Rule Detection Helpers
# ============================================================
def _append_rule_hits(violations, dates, values, indices, rule_name):
    for idx in sorted(set(indices)):
        if idx < 0 or idx >= len(values):
            continue
        v = values[idx]
        if pd.isna(v):
            continue
        violations.append({
            "date": pd.to_datetime(dates[idx]),
            "rule": rule_name,
            "value": v,
            "rule_description": RULE_DISPLAY_TEXT.get(rule_name, rule_name)
        })


def _mark_run_same_side(values, cl, min_run_len):
    values = np.asarray(values, dtype=float)
    cl = np.asarray(cl, dtype=float)
    side = np.where(values > cl, 1, np.where(values < cl, -1, 0))
    flagged = set()
    n = len(side)

    i = 0
    while i < n:
        if side[i] == 0:
            i += 1
            continue

        j = i
        while j + 1 < n and side[j + 1] == side[i]:
            j += 1

        run_len = j - i + 1
        if run_len >= min_run_len:
            flagged.update(range(i, j + 1))

        i = j + 1

    return flagged


def _mark_monotonic_runs(values, min_points):
    flagged = set()
    n = len(values)

    if n < min_points:
        return flagged

    diffs = np.diff(values)

    i = 0
    while i < len(diffs):
        if pd.isna(diffs[i]) or diffs[i] == 0:
            i += 1
            continue

        sign = 1 if diffs[i] > 0 else -1
        j = i

        while (
            j + 1 < len(diffs)
            and not pd.isna(diffs[j + 1])
            and diffs[j + 1] != 0
            and ((diffs[j + 1] > 0 and sign == 1) or (diffs[j + 1] < 0 and sign == -1))
        ):
            j += 1

        points_in_run = (j - i + 1) + 1
        if points_in_run >= min_points:
            flagged.update(range(i, j + 2))

        i = j + 1

    return flagged


def _mark_alternating_runs(values, min_points):
    flagged = set()
    n = len(values)

    if n < min_points:
        return flagged

    diffs = np.diff(values)

    i = 0
    while i < len(diffs):
        if pd.isna(diffs[i]) or diffs[i] == 0:
            i += 1
            continue

        prev_sign = 1 if diffs[i] > 0 else -1
        j = i

        while j + 1 < len(diffs):
            next_diff = diffs[j + 1]
            if pd.isna(next_diff) or next_diff == 0:
                break

            next_sign = 1 if next_diff > 0 else -1
            if next_sign == -prev_sign:
                prev_sign = next_sign
                j += 1
            else:
                break

        points_in_run = (j - i + 1) + 1
        if points_in_run >= min_points:
            flagged.update(range(i, j + 2))

        i = j + 1

    return flagged


def _as_array(x, n):
    if np.isscalar(x) or x is None:
        return np.repeat(np.nan if x is None else float(x), n)
    arr = np.asarray(x, dtype=float)
    if len(arr) != n:
        raise ValueError("Input array length mismatch.")
    return arr


def detect_spc_rule_violations(df: pd.DataFrame, y_col: str, cl, sigma):
    values = pd.to_numeric(df[y_col], errors="coerce").values
    dates = pd.to_datetime(df["date"]).values
    n = len(values)

    if n == 0:
        return pd.DataFrame(columns=["date", "rule", "value", "rule_description"])

    cl_arr = _as_array(cl, n)
    sigma_arr = _as_array(sigma, n)

    if np.all(np.isnan(sigma_arr)) or np.nanmax(np.abs(sigma_arr)) == 0:
        return pd.DataFrame(columns=["date", "rule", "value", "rule_description"])

    violations = []

    upper_1 = cl_arr + 1 * sigma_arr
    lower_1 = cl_arr - 1 * sigma_arr
    upper_2 = cl_arr + 2 * sigma_arr
    lower_2 = cl_arr - 2 * sigma_arr
    upper_3 = cl_arr + 3 * sigma_arr
    lower_3 = cl_arr - 3 * sigma_arr

    rule1_idx = np.where((values > upper_3) | (values < lower_3))[0]
    _append_rule_hits(violations, dates, values, rule1_idx, "Rule 1")

    rule2_idx = _mark_run_same_side(values, cl_arr, min_run_len=9)
    _append_rule_hits(violations, dates, values, rule2_idx, "Rule 2")

    rule3_idx = _mark_monotonic_runs(values, min_points=6)
    _append_rule_hits(violations, dates, values, rule3_idx, "Rule 3")

    rule4_idx = _mark_alternating_runs(values, min_points=14)
    _append_rule_hits(violations, dates, values, rule4_idx, "Rule 4")

    rule5_idx = set()
    for i in range(2, n):
        window = values[i - 2:i + 1]
        if np.any(pd.isna(window)):
            continue

        above = [j for j in range(3) if window[j] > upper_2[i - 2 + j]]
        below = [j for j in range(3) if window[j] < lower_2[i - 2 + j]]

        if len(above) >= 2:
            rule5_idx.update((i - 2) + np.array(above))
        if len(below) >= 2:
            rule5_idx.update((i - 2) + np.array(below))

    _append_rule_hits(violations, dates, values, rule5_idx, "Rule 5")

    rule6_idx = set()
    for i in range(4, n):
        window = values[i - 4:i + 1]
        if np.any(pd.isna(window)):
            continue

        above = [j for j in range(5) if window[j] > upper_1[i - 4 + j]]
        below = [j for j in range(5) if window[j] < lower_1[i - 4 + j]]

        if len(above) >= 4:
            rule6_idx.update((i - 4) + np.array(above))
        if len(below) >= 4:
            rule6_idx.update((i - 4) + np.array(below))

    _append_rule_hits(violations, dates, values, rule6_idx, "Rule 6")

    rule7_idx = set()
    within_1 = np.abs(values - cl_arr) <= sigma_arr

    i = 0
    while i < n:
        if not within_1[i] or pd.isna(values[i]):
            i += 1
            continue

        j = i
        while j + 1 < n and within_1[j + 1] and not pd.isna(values[j + 1]):
            j += 1

        run_len = j - i + 1
        if run_len >= 15:
            rule7_idx.update(range(i, j + 1))

        i = j + 1

    _append_rule_hits(violations, dates, values, rule7_idx, "Rule 7")

    rule8_idx = set()
    for i in range(7, n):
        window = values[i - 7:i + 1]
        if np.any(pd.isna(window)):
            continue

        outside_1 = np.array(
            [abs(window[j] - cl_arr[i - 7 + j]) > sigma_arr[i - 7 + j] for j in range(8)]
        )
        sides = np.array(
            [
                1 if window[j] > cl_arr[i - 7 + j]
                else (-1 if window[j] < cl_arr[i - 7 + j] else 0)
                for j in range(8)
            ]
        )

        if np.all(outside_1) and (np.any(sides == 1) and np.any(sides == -1)):
            rule8_idx.update(range(i - 7, i + 1))

    _append_rule_hits(violations, dates, values, rule8_idx, "Rule 8")

    vdf = pd.DataFrame(violations)
    if not vdf.empty:
        vdf = (
            vdf.drop_duplicates(subset=["date", "rule"])
            .sort_values(
                ["date", "rule"],
                key=lambda s: s.map(RULE_SORT_ORDER) if s.name == "rule" else s
            )
            .reset_index(drop=True)
        )
    else:
        vdf = pd.DataFrame(columns=["date", "rule", "value", "rule_description"])

    return vdf


def detect_secondary_limit_breaches(df: pd.DataFrame, y_col: str, ucl, lcl):
    values = pd.to_numeric(df[y_col], errors="coerce").values
    dates = pd.to_datetime(df["date"]).values
    n = len(values)

    if n == 0:
        return pd.DataFrame(columns=["date", "rule", "value", "rule_description"])

    ucl_arr = _as_array(ucl, n)
    lcl_arr = _as_array(lcl, n)

    violations = []
    for i, v in enumerate(values):
        if pd.isna(v):
            continue
        if v > ucl_arr[i] or v < lcl_arr[i]:
            violations.append({
                "date": pd.to_datetime(dates[i]),
                "rule": "Secondary chart: point beyond control limit",
                "value": v,
                "rule_description": RULE_DISPLAY_TEXT["Secondary chart: point beyond control limit"]
            })

    vdf = pd.DataFrame(violations)
    if not vdf.empty:
        vdf = (
            vdf.drop_duplicates(subset=["date", "rule"])
            .sort_values(
                ["date", "rule"],
                key=lambda s: s.map(RULE_SORT_ORDER) if s.name == "rule" else s
            )
            .reset_index(drop=True)
        )
    else:
        vdf = pd.DataFrame(columns=["date", "rule", "value", "rule_description"])

    return vdf


# ============================================================
# Plot Helpers
# ============================================================
def add_limit_lines(fig, x_values, cl, ucl, lcl, sigma, row, col, show_legend_once=False):
    # Use x_values as provided (either dates or order indices)
    n = len(x_values)

    cl_arr = _as_array(cl, n)
    ucl_arr = _as_array(ucl, n)
    lcl_arr = _as_array(lcl, n)

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=cl_arr,
            mode="lines",
            name="Center Line",
            legendgroup="center_line",
            showlegend=show_legend_once,
            line=dict(color="#228B22", dash="dash")
        ),
        row=row, col=col
    )

    # Visual-only horizontal 1σ and 2σ reference lines.
    sigma_arr = _as_array(sigma, n)
    if not np.all(np.isnan(sigma_arr)) and np.nanmax(np.abs(sigma_arr)) != 0:
        sigma_ref = float(np.nanmean(sigma_arr))
        cl_ref = float(np.nanmean(cl_arr))

        upper_1_arr = _repeat_line(cl_ref + sigma_ref, n)
        lower_1_arr = _repeat_line(cl_ref - sigma_ref, n)
        upper_2_arr = _repeat_line(cl_ref + 2 * sigma_ref, n)
        lower_2_arr = _repeat_line(cl_ref - 2 * sigma_ref, n)

        sigma_line_color = "#6A5ACD"  # same color for both 1σ and 2σ

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=upper_1_arr,
                mode="lines",
                name="Upper 1 Sigma",
                legendgroup="sigma_1",
                showlegend=show_legend_once,
                line=dict(color=sigma_line_color, dash="dash")
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=lower_1_arr,
                mode="lines",
                name="Lower 1 Sigma",
                legendgroup="sigma_1",
                showlegend=False,
                line=dict(color=sigma_line_color, dash="dash")
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=upper_2_arr,
                mode="lines",
                name="Upper 2 Sigma",
                legendgroup="sigma_2",
                showlegend=show_legend_once,
                line=dict(color=sigma_line_color, dash="dot")
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=lower_2_arr,
                mode="lines",
                name="Lower 2 Sigma",
                legendgroup="sigma_2",
                showlegend=False,
                line=dict(color=sigma_line_color, dash="dot")
            ),
            row=row, col=col
        )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=ucl_arr,
            mode="lines",
            name="Upper Control Limit",
            legendgroup="ucl",
            showlegend=show_legend_once,
            line=dict(color="#B22222", dash="dot")
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=lcl_arr,
            mode="lines",
            name="Lower Control Limit",
            legendgroup="lcl",
            showlegend=show_legend_once,
            line=dict(color="#B22222", dash="dot")
        ),
        row=row, col=col
    )


def add_rule_markers(fig, source_df, violations_df, y_col, row, col, legend_shown_rules, x_col: str, x_axis_mode: str):
    if violations_df.empty:
        return legend_shown_rules

    # Merge to get x_col for the violated dates
    merge_cols = ["date", y_col]
    if x_col not in merge_cols:
        merge_cols.append(x_col)

    merged = (
        source_df[merge_cols]
        .merge(
            violations_df[["date", "rule", "rule_description"]].drop_duplicates(),
            on="date",
            how="inner"
        )
        .drop_duplicates(subset=["date", "rule"])
        .sort_values(
            ["rule", "date"],
            key=lambda s: s.map(RULE_SORT_ORDER) if s.name == "rule" else s
        )
    )

    multi_rule_df = (
        violations_df.groupby("date", as_index=False)
        .agg(
            rule_count=("rule", "nunique"),
            rules=("rule", lambda x: sorted(set(x), key=lambda r: RULE_SORT_ORDER.get(r, 999)))
        )
    )
    multi_rule_df = multi_rule_df[multi_rule_df["rule_count"] > 1].copy()
    multi_rule_dates = set(pd.to_datetime(multi_rule_df["date"]))

    single_rule_points = merged.copy()

    for rule_name in single_rule_points["rule"].unique():
        rule_points = single_rule_points[single_rule_points["rule"] == rule_name].copy()
        style = RULE_STYLE_MAP.get(rule_name, DEFAULT_RULE_STYLE)
        show_legend = rule_name not in legend_shown_rules

        # Hover template depends on x-axis mode
        if x_axis_mode == "Time":
            hovertemplate = (
                "<b>%{x|%Y-%m-%d}</b><br>"
                f"Value: %{{y:.5f}}<br>"
                f"Rule: {rule_name}<br>"
                f"Description: {RULE_DISPLAY_TEXT.get(rule_name, rule_name)}"
                "<extra></extra>"
            )
            x_vals = rule_points["date"]
        else:
            hovertemplate = (
                "<b>Order: %{x}</b><br>"
                f"Value: %{{y:.5f}}<br>"
                f"Rule: {rule_name}<br>"
                f"Description: {RULE_DISPLAY_TEXT.get(rule_name, rule_name)}"
                "<extra></extra>"
            )
            x_vals = rule_points[x_col]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=rule_points[y_col],
                mode="markers",
                name=style["label"],
                legendgroup=rule_name,
                showlegend=show_legend,
                marker=dict(
                    color=style["color"],
                    size=10,
                    symbol="x",
                    line=dict(width=2, color=style["color"])
                ),
                hovertemplate=hovertemplate
            ),
            row=row, col=col
        )

        legend_shown_rules.add(rule_name)

    if not multi_rule_df.empty:
        multi_rule_df = multi_rule_df.merge(
            source_df[merge_cols],
            on="date",
            how="left"
        )

        style = RULE_STYLE_MAP["Multiple rules"]
        show_legend = "Multiple rules" not in legend_shown_rules

        hover_text = []
        for _, r in multi_rule_df.iterrows():
            rules_text = "<br>".join([f"- {rule}" for rule in r["rules"]])
            date_str = pd.to_datetime(r["date"]).strftime('%Y-%m-%d')
            if x_axis_mode == "Time":
                base = f"<b>{date_str}</b><br>"
            else:
                base = f"<b>Order: {int(r[x_col]) if pd.notna(r[x_col]) else ''}</b><br>Date: {date_str}<br>"
            hover_text.append(
                base +
                f"Value: {r[y_col]:.5f}<br>"
                f"Multiple rules triggered:<br>{rules_text}"
            )

        x_vals = multi_rule_df["date"] if x_axis_mode == "Time" else multi_rule_df[x_col]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=multi_rule_df[y_col],
                mode="markers",
                name=style["label"],
                legendgroup="Multiple rules",
                showlegend=show_legend,
                marker=dict(
                    color=style["color"],
                    size=10,
                    symbol="diamond-x",
                    line=dict(width=2, color=style["color"])
                ),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text
            ),
            row=row, col=col
        )

        legend_shown_rules.add("Multiple rules")

    return legend_shown_rules


def plot_spc_chart(chart_df, limits, chart_type, title, primary_violations, secondary_violations, x_axis_mode: str = "Order"):
    primary = limits["primary"]
    secondary = limits["secondary"]

    x_col = "order" if x_axis_mode == "Order" else "date"
    x_axis_title = "Order" if x_axis_mode == "Order" else "Date / Time"

    if secondary is None:
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(f"{primary['label']} Chart",)
        )

        fig.add_trace(
            go.Scatter(
                x=chart_df[x_col],
                y=chart_df[primary["y_col"]],
                mode="lines+markers",
                name=primary["label"],
                legendgroup="primary_series",
                showlegend=True,
                line=dict(color="#5B5B5B", width=1.8),
                marker=dict(size=5, color="#5B5B5B")
            ),
            row=1, col=1
        )

        add_limit_lines(
            fig=fig,
            x_values=chart_df[x_col],
            cl=primary["CL_series"],
            ucl=primary["UCL_series"],
            lcl=primary["LCL_series"],
            sigma=primary["sigma_series"],
            row=1,
            col=1,
            show_legend_once=True
        )

        legend_shown_rules = set()
        add_rule_markers(
            fig=fig,
            source_df=chart_df,
            violations_df=primary_violations,
            y_col=primary["y_col"],
            row=1,
            col=1,
            legend_shown_rules=legend_shown_rules,
            x_col=x_col,
            x_axis_mode=x_axis_mode
        )

        fig.update_layout(
            height=520,
            title=dict(text=title, x=0.5, xanchor="center"),
            legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0),
            margin=dict(l=40, r=20, t=120, b=40),
            hovermode="closest"
        )
        fig.update_yaxes(title_text=primary["label"], row=1, col=1)
        fig.update_xaxes(title_text=x_axis_title, row=1, col=1)
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"{primary['label']} Chart", f"{secondary['label']} Chart")
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df[x_col],
            y=chart_df[primary["y_col"]],
            mode="lines+markers",
            name=primary["label"],
            legendgroup="primary_series",
            showlegend=True,
            line=dict(color="#5B5B5B", width=1.8),
            marker=dict(size=5, color="#5B5B5B")
        ),
        row=1, col=1
    )

    add_limit_lines(
        fig=fig,
        x_values=chart_df[x_col],
        cl=primary["CL_series"],
        ucl=primary["UCL_series"],
        lcl=primary["LCL_series"],
        sigma=primary["sigma_series"],
        row=1,
        col=1,
        show_legend_once=True
    )

    legend_shown_rules = set()
    legend_shown_rules = add_rule_markers(
        fig=fig,
        source_df=chart_df,
        violations_df=primary_violations,
        y_col=primary["y_col"],
        row=1,
        col=1,
        legend_shown_rules=legend_shown_rules,
        x_col=x_col,
        x_axis_mode=x_axis_mode
    )

    sec_df = chart_df.dropna(subset=[secondary["y_col"]]).copy()

    fig.add_trace(
        go.Scatter(
            x=sec_df[x_col],
            y=sec_df[secondary["y_col"]],
            mode="lines+markers",
            name=secondary["label"],
            legendgroup="secondary_series",
            showlegend=True,
            line=dict(color="#8C6D1F", width=1.8),
            marker=dict(size=5, color="#8C6D1F")
        ),
        row=2, col=1
    )

    add_limit_lines(
        fig=fig,
        x_values=sec_df[x_col],
        cl=secondary["CL_series"][:len(sec_df)],
        ucl=secondary["UCL_series"][:len(sec_df)],
        lcl=secondary["LCL_series"][:len(sec_df)],
        sigma=secondary["sigma_series"][:len(sec_df)] if secondary["sigma_series"] is not None else None,
        row=2,
        col=1,
        show_legend_once=False
    )

    add_rule_markers(
        fig=fig,
        source_df=sec_df,
        violations_df=secondary_violations,
        y_col=secondary["y_col"],
        row=2,
        col=1,
        legend_shown_rules=legend_shown_rules,
        x_col=x_col,
        x_axis_mode=x_axis_mode
    )

    fig.update_layout(
        height=780,
        title=dict(text=title, x=0.5, xanchor="center", y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=140, b=40),
        hovermode="closest"
    )

    fig.update_yaxes(title_text=primary["label"], row=1, col=1)
    fig.update_yaxes(title_text=secondary["label"], row=2, col=1)
    fig.update_xaxes(title_text=x_axis_title, row=2, col=1)

    return fig


# ============================================================
# Summary Helpers
# ============================================================
def summarize_violations(primary_vdf, secondary_vdf):
    combined = pd.concat([primary_vdf, secondary_vdf], ignore_index=True)

    if combined.empty:
        summary = pd.DataFrame(columns=["rule", "rule_description", "count"])
    else:
        summary = (
            combined.groupby(["rule", "rule_description"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        summary["rule_order"] = summary["rule"].map(RULE_SORT_ORDER)
        summary = (
            summary.sort_values(["rule_order", "count"], ascending=[True, False])
            .drop(columns=["rule_order"])
            .reset_index(drop=True)
        )

    return combined, summary


# ============================================================
# Period Helpers
# ============================================================
def add_period_columns(chart_df: pd.DataFrame) -> pd.DataFrame:
    df = chart_df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["quarter_period"] = pd.to_datetime(df["date"]).dt.to_period("Q")
    df["quarter_label"] = df["quarter_period"].astype(str)
    return df


def filter_chart_df_by_selected_period(chart_df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    temp = chart_df.copy()
    dt_series = pd.to_datetime(temp["date"])
    mask = (dt_series.dt.date >= start_date) & (dt_series.dt.date <= end_date)
    return temp.loc[mask].copy().reset_index(drop=True)


# ============================================================
# Sidebar Controls
# ============================================================
st.sidebar.header("Upload & Mapping")

uploaded_file = st.sidebar.file_uploader(
    "Upload one data file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV or Excel file to continue.")
    st.stop()

file_name = uploaded_file.name.lower()

try:
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        try:
            sheet_names = load_uploaded_table(uploaded_file, selected_sheet=None)
        except Exception as e:
            st.error(f"Unable to read Excel workbook: {e}")
            st.stop()

        if not sheet_names:
            st.error("No sheets were found in the uploaded Excel file.")
            st.stop()

        selected_sheet = st.sidebar.selectbox("Select Excel sheet", sheet_names)
        source_df = load_uploaded_table(uploaded_file, selected_sheet=selected_sheet)
    else:
        selected_sheet = None
        source_df = load_uploaded_table(uploaded_file)

except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

if source_df is None or source_df.empty:
    st.error("The uploaded file/sheet is empty.")
    st.stop()

source_df = normalize_column_names(source_df)
column_report = validate_uploaded_columns(source_df)

if column_report["fatal_errors"]:
    for msg in column_report["fatal_errors"]:
        st.error(msg)
    st.stop()

for warn in column_report["warnings"]:
    st.sidebar.warning(warn)

all_cols, numeric_cols, non_empty_cols = get_candidate_columns(source_df)
invalid_all_null_cols = set(column_report["all_null_columns"])
available_cols = [c for c in non_empty_cols if c not in invalid_all_null_cols]

if not available_cols:
    st.error("No usable columns remain after invalid column checks.")
    st.stop()

st.sidebar.markdown("### Dataset structure")
mode = st.sidebar.radio(
    "Select the type of data you want to chart",
    ["Continuous", "Defectives", "Defects"],
    help=(
        "Continuous = measured values over time.\n"
        "Defectives = count of defective units with sample size.\n"
        "Defects = count of defects, optionally with inspection units/opportunities."
    )
)

st.sidebar.markdown("### Empty row handling")
missing_strategy_label = st.sidebar.radio(
    "If selected columns contain empty entries, what should be done?",
    [
        "Replace empty entries with zeros",
        "Disregard rows with empty required fields"
    ]
)
missing_strategy = "zeros" if "zeros" in missing_strategy_label.lower() else "drop"

series_label = st.sidebar.text_input(
    "Series / chart label",
    value="Uploaded Series"
)

st.sidebar.markdown("### Column mapping")
time_col = st.sidebar.selectbox(
    "Time column",
    options=available_cols,
    help="Select the date/time column."
)

# NEW: X-axis mode selection (default to Order)
st.sidebar.markdown("### X-axis mode")
x_axis_mode = st.sidebar.radio(
    "Plot by",
    ["Order", "Time"],
    index=0,
    help="Default is by Order (sequence). Choose Time to use the time column on the x-axis."
)

mapping = {
    "time_col": time_col,
    "series_label": series_label
}

if mode == "Continuous":
    value_options = [c for c in available_cols if c != time_col]
    subgroup_options = ["-- None --"] + [c for c in available_cols if c != time_col]

    if not value_options:
        st.error("No usable measurement columns are available.")
        st.stop()

    value_col = st.sidebar.selectbox(
        "Continuous measurement column",
        options=value_options,
        help="Select the numeric measurement column."
    )
    subgroup_col = st.sidebar.selectbox(
        "Subgroup column (optional)",
        options=subgroup_options,
        help=(
            "Optional. Use this if your dataset contains subgroup identifiers. "
            "If omitted, subgrouping is inferred from repeated time values."
        )
    )

    mapping["value_col"] = value_col
    mapping["subgroup_col"] = subgroup_col

elif mode == "Defectives":
    candidate_options = [c for c in available_cols if c != time_col]
    if len(candidate_options) < 2:
        st.error("You need at least two usable non-time columns for defectives data.")
        st.stop()

    defectives_col = st.sidebar.selectbox(
        "Defectives count column",
        options=candidate_options,
        help="Select the column containing the number of defectives."
    )

    remaining_for_n = [c for c in candidate_options if c != defectives_col]
    if not remaining_for_n:
        st.error("You need one additional usable column for sample size.")
        st.stop()

    sample_size_col = st.sidebar.selectbox(
        "Sample size column",
        options=remaining_for_n,
        help="Select the sample size column."
    )

    mapping["defectives_col"] = defectives_col
    mapping["sample_size_col"] = sample_size_col

elif mode == "Defects":
    candidate_options = [c for c in available_cols if c != time_col]
    if not candidate_options:
        st.error("You need at least one usable non-time column for defects data.")
        st.stop()

    defects_col = st.sidebar.selectbox(
        "Defects count column",
        options=candidate_options,
        help="Select the column containing the number of defects."
    )

    inspection_options = ["-- None --"] + [c for c in candidate_options if c != defects_col]

    inspection_units_col = st.sidebar.selectbox(
        "Inspection units / opportunities column (optional)",
        options=inspection_options,
        help=(
            "Optional. If omitted, the app assumes constant opportunity and only c-chart logic applies."
        )
    )

    mapping["defects_col"] = defects_col
    mapping["inspection_units_col"] = inspection_units_col

# ============================================================
# Empty handling before mapping
# ============================================================
try:
    cleaned_source_df, empty_handling_notes = apply_missing_value_strategy(
        source_df=source_df,
        mode=mode,
        mapping=mapping,
        strategy=missing_strategy
    )
except Exception as e:
    st.error(f"Error while handling empty rows: {e}")
    st.stop()

if cleaned_source_df.empty:
    st.error("No rows remain after the selected empty-row handling was applied.")
    st.stop()

# ============================================================
# Build internal SPC dataset from selected mapping
# ============================================================
try:
    raw_df, meta = build_raw_df_from_mapping(cleaned_source_df, mode, mapping)
except Exception as e:
    st.error(f"Column mapping error: {e}")
    st.stop()

valid_chart_options = detect_chart_options_for_variable(raw_df, meta)

st.sidebar.markdown("### Valid SPC chart options")
if valid_chart_options:
    st.sidebar.success(", ".join(valid_chart_options))
else:
    st.sidebar.error("No valid SPC chart detected for the current data mapping.")

if not valid_chart_options:
    st.error(
        "No valid SPC charts could be detected for the selected data and column mapping. "
        "Please review the mapped columns and data structure."
    )
    st.stop()

selected_chart = st.sidebar.selectbox(
    "Select SPC chart",
    options=valid_chart_options,
    index=0
)

# ============================================================
# Main Dataset Overview
# ============================================================
st.header(meta["label"])
st.markdown(f"**Detected data type**: {describe_variable_type(meta, raw_df)}")
st.markdown(f"**Valid SPC chart option(s)**: {', '.join(valid_chart_options)}")

with st.expander("File / mapping summary", expanded=False):
    file_type = "Excel" if (file_name.endswith(".xlsx") or file_name.endswith(".xls")) else "CSV"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**File**: {uploaded_file.name}")
    with c2:
        st.write(f"**File type**: {file_type}")
    with c3:
        st.write(f"**Sheet**: {selected_sheet if selected_sheet is not None else 'N/A'}")

    st.write("**Mapped source columns:**")
    st.json(meta["source_columns"])

    st.write("**Note on empty rows / missing entries:**")
    for note in empty_handling_notes:
        st.info(note)

    if column_report["all_null_columns"]:
        st.warning(
            f"Completely empty columns detected in the uploaded file: {column_report['all_null_columns']}"
        )

# ============================================================
# Compute SPC on full aggregated data (before period filter)
# ============================================================
try:
    chart_df = aggregate_for_chart(raw_df, selected_chart)
    chart_df = add_period_columns(chart_df)
except Exception as e:
    st.error(f"Unable to compute {selected_chart} chart for '{meta['label']}': {e}")
    st.stop()

if chart_df.empty:
    st.error("No aggregated SPC observations are available.")
    st.stop()

# ============================================================
# Selected Time Period Controls
# ============================================================
chart_min_date = pd.to_datetime(chart_df["date"]).min().date()
chart_max_date = pd.to_datetime(chart_df["date"]).max().date()

today_date = pd.Timestamp.now().date()
default_end_date = min(chart_max_date, today_date)
default_start_candidate = (pd.Timestamp(today_date) - pd.DateOffset(months=12)).date()
default_start_date = max(chart_min_date, default_start_candidate)

# Fallback if the requested default 12-month window does not overlap available data
if default_start_date > default_end_date:
    default_start_date = chart_min_date
    default_end_date = chart_max_date

st.sidebar.markdown("### Analysis period")
selected_start_date = st.sidebar.date_input(
    "Start date for main SPC chart",
    value=default_start_date,
    min_value=chart_min_date,
    max_value=chart_max_date
)

selected_end_date = st.sidebar.date_input(
    "End date for main SPC chart",
    value=default_end_date,
    min_value=chart_min_date,
    max_value=chart_max_date
)

if selected_start_date > selected_end_date:
    st.error("The selected start date cannot be later than the selected end date.")
    st.stop()

period_breakdown_mode = st.sidebar.radio(
    "Breakdown graphs for selected period",
    ["Yearly", "Quarterly"],
    help="Choose whether the selected time period should be broken down into yearly or quarterly SPC charts."
)

period_chart_df = filter_chart_df_by_selected_period(chart_df, selected_start_date, selected_end_date)

if period_chart_df.empty:
    st.error("No observations fall within the selected time period.")
    st.stop()

# Add order index for the selected period (used when plotting by Order)
period_chart_df = period_chart_df.copy().reset_index(drop=True)
period_chart_df["order"] = np.arange(1, len(period_chart_df) + 1)

# ============================================================
# Compute SPC for selected period
# ============================================================
try:
    limits = calc_chart_limits(period_chart_df, selected_chart)
except Exception as e:
    st.error(f"Unable to compute {selected_chart} chart for the selected period: {e}")
    st.stop()

primary = limits["primary"]
secondary = limits["secondary"]

primary_violations = detect_spc_rule_violations(
    period_chart_df,
    y_col=primary["y_col"],
    cl=primary["CL_series"],
    sigma=primary["sigma_series"]
)

if secondary is not None:
    sec_input = period_chart_df.dropna(subset=[secondary["y_col"]]).copy()
    secondary_violations = detect_secondary_limit_breaches(
        sec_input,
        y_col=secondary["y_col"],
        ucl=secondary["UCL_series"][:len(sec_input)],
        lcl=secondary["LCL_series"][:len(sec_input)]
    )
else:
    secondary_violations = pd.DataFrame(
        columns=["date", "rule", "value", "rule_description"]
    )

combined_v, summary_v = summarize_violations(primary_violations, secondary_violations)

# ============================================================
# Selected Period Summary
# ============================================================
selected_period_text = f"{selected_start_date} to {selected_end_date}"
st.markdown(f"**Selected analysis period**: {selected_period_text}")
st.markdown(f"**Breakdown mode for selected period**: {period_breakdown_mode}")

# ============================================================
# KPI Metrics
# ============================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Observations (time points)", int(period_chart_df["date"].nunique()))
with c2:
    st.metric("Primary CL", metric_value_or_range(primary["CL_series"]))
with c3:
    st.metric("Primary UCL", metric_value_or_range(primary["UCL_series"]))
with c4:
    st.metric("Primary LCL", metric_value_or_range(primary["LCL_series"]))

# ============================================================
# Main SPC Chart - Selected Period
# ============================================================
st.subheader("SPC Chart - Selected Period")
fig_all = plot_spc_chart(
    chart_df=period_chart_df,
    limits=limits,
    chart_type=selected_chart,
    title=f"{meta['label']} - {selected_chart} Chart ({selected_start_date} to {selected_end_date})",
    primary_violations=primary_violations,
    secondary_violations=secondary_violations,
    x_axis_mode=x_axis_mode
)
st.plotly_chart(fig_all, use_container_width=True)

# ============================================================
# Breakdown Period Selection (NEW)
# ============================================================
if period_breakdown_mode == "Yearly":
    available_breakdown_periods = sorted(period_chart_df["year"].dropna().unique().tolist())
    breakdown_period_label = "years"
else:
    available_breakdown_periods = sorted(period_chart_df["quarter_period"].dropna().unique().tolist())
    breakdown_period_label = "quarters"

if not available_breakdown_periods:
    st.error("No breakdown periods are available within the selected time period.")
    st.stop()

periods_back_to_show = st.sidebar.number_input(
    f"How many {breakdown_period_label} back would you like to create charts for?",
    min_value=1,
    max_value=len(available_breakdown_periods),
    value=len(available_breakdown_periods),
    step=1
)

# ============================================================
# Violations Summary - Selected Period
# ============================================================
st.subheader("Detected Special Cause Variation - Selected Period")
if combined_v.empty:
    st.success("No SPC rule violations detected in the selected period.")
else:
    st.markdown("The following SPC rules were broken:")
    st.dataframe(summary_v, use_container_width=True)

    st.markdown("Violation details:")
    display_v = combined_v.copy()
    display_v["date"] = pd.to_datetime(display_v["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    display_v["rule_order"] = display_v["rule"].map(RULE_SORT_ORDER)
    display_v = (
        display_v.sort_values(["date", "rule_order"])
        .drop(columns=["rule_order"])
        .reset_index(drop=True)
    )
    st.dataframe(display_v, use_container_width=True)

# ============================================================
# SPC Charts by Selected Period Breakdown
# ============================================================
breakdown_title = "SPC Charts by Year (Selected Period)" if period_breakdown_mode == "Yearly" else "SPC Charts by Quarter (Selected Period)"
st.subheader(breakdown_title)

if period_breakdown_mode == "Yearly":
    years = sorted(period_chart_df["year"].dropna().unique().tolist())
    years = years[-int(periods_back_to_show):]

    for yr in years:
        year_df = period_chart_df[period_chart_df["year"] == yr].copy().reset_index(drop=True)
        if year_df.empty:
            continue

        # Recompute order for the year subset
        year_df["order"] = np.arange(1, len(year_df) + 1)

        try:
            year_limits = calc_chart_limits(year_df, selected_chart)
        except Exception:
            continue

        year_primary = year_limits["primary"]
        year_secondary = year_limits["secondary"]

        year_primary_viol = detect_spc_rule_violations(
            year_df,
            y_col=year_primary["y_col"],
            cl=year_primary["CL_series"],
            sigma=year_primary["sigma_series"]
        )

        if year_secondary is not None:
            sec_year_df = year_df.dropna(subset=[year_secondary["y_col"]]).copy()
            year_secondary_viol = detect_secondary_limit_breaches(
                sec_year_df,
                y_col=year_secondary["y_col"],
                ucl=year_secondary["UCL_series"][:len(sec_year_df)],
                lcl=year_secondary["LCL_series"][:len(sec_year_df)]
            )
        else:
            year_secondary_viol = pd.DataFrame(
                columns=["date", "rule", "value", "rule_description"]
            )

        year_combined_v, year_summary_v = summarize_violations(year_primary_viol, year_secondary_viol)

        with st.expander(f"Year {yr}", expanded=False):
            fig_year = plot_spc_chart(
                chart_df=year_df,
                limits=year_limits,
                chart_type=selected_chart,
                title=f"{meta['label']} - {selected_chart} Chart ({yr})",
                primary_violations=year_primary_viol,
                secondary_violations=year_secondary_viol,
                x_axis_mode=x_axis_mode
            )
            st.plotly_chart(fig_year, use_container_width=True)

            if year_combined_v.empty:
                st.success(f"No SPC rule violations detected for {yr}.")
            else:
                st.markdown(f"**SPC rules broken in {yr}:**")
                st.dataframe(year_summary_v, use_container_width=True)

                year_display_v = year_combined_v.copy()
                year_display_v["date"] = pd.to_datetime(year_display_v["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                year_display_v["rule_order"] = year_display_v["rule"].map(RULE_SORT_ORDER)
                year_display_v = (
                    year_display_v.sort_values(["date", "rule_order"])
                    .drop(columns=["rule_order"])
                    .reset_index(drop=True)
                )
                st.dataframe(year_display_v, use_container_width=True)

else:
    quarter_periods = sorted(period_chart_df["quarter_period"].dropna().unique().tolist())
    quarter_periods = quarter_periods[-int(periods_back_to_show):]

    for qtr in quarter_periods:
        quarter_df = period_chart_df[period_chart_df["quarter_period"] == qtr].copy().reset_index(drop=True)
        if quarter_df.empty:
            continue

        # Recompute order for the quarter subset
        quarter_df["order"] = np.arange(1, len(quarter_df) + 1)

        try:
            quarter_limits = calc_chart_limits(quarter_df, selected_chart)
        except Exception:
            continue

        quarter_primary = quarter_limits["primary"]
        quarter_secondary = quarter_limits["secondary"]

        quarter_primary_viol = detect_spc_rule_violations(
            quarter_df,
            y_col=quarter_primary["y_col"],
            cl=quarter_primary["CL_series"],
            sigma=quarter_primary["sigma_series"]
        )

        if quarter_secondary is not None:
            sec_quarter_df = quarter_df.dropna(subset=[quarter_secondary["y_col"]]).copy()
            quarter_secondary_viol = detect_secondary_limit_breaches(
                sec_quarter_df,
                y_col=quarter_secondary["y_col"],
                ucl=quarter_secondary["UCL_series"][:len(sec_quarter_df)],
                lcl=quarter_secondary["LCL_series"][:len(sec_quarter_df)]
            )
        else:
            quarter_secondary_viol = pd.DataFrame(
                columns=["date", "rule", "value", "rule_description"]
            )

        quarter_combined_v, quarter_summary_v = summarize_violations(quarter_primary_viol, quarter_secondary_viol)
        quarter_label = str(qtr)

        with st.expander(f"Quarter {quarter_label}", expanded=False):
            fig_quarter = plot_spc_chart(
                chart_df=quarter_df,
                limits=quarter_limits,
                chart_type=selected_chart,
                title=f"{meta['label']} - {selected_chart} Chart ({quarter_label})",
                primary_violations=quarter_primary_viol,
                secondary_violations=quarter_secondary_viol,
                x_axis_mode=x_axis_mode
            )
            st.plotly_chart(fig_quarter, use_container_width=True)

            if quarter_combined_v.empty:
                st.success(f"No SPC rule violations detected for {quarter_label}.")
            else:
                st.markdown(f"**SPC rules broken in {quarter_label}:**")
                st.dataframe(quarter_summary_v, use_container_width=True)

                quarter_display_v = quarter_combined_v.copy()
                quarter_display_v["date"] = pd.to_datetime(quarter_display_v["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                quarter_display_v["rule_order"] = quarter_display_v["rule"].map(RULE_SORT_ORDER)
                quarter_display_v = (
                    quarter_display_v.sort_values(["date", "rule_order"])
                    .drop(columns=["rule_order"])
                    .reset_index(drop=True)
                )
                st.dataframe(quarter_display_v, use_container_width=True)

# ============================================================
# Raw Data Preview
# ============================================================
with st.expander("Preview original uploaded data"):
    st.dataframe(source_df.head(50), use_container_width=True)

with st.expander("Preview cleaned uploaded data (after empty-row handling)"):
    st.dataframe(cleaned_source_df.head(50), use_container_width=True)

with st.expander("Preview mapped SPC input data"):
    st.dataframe(raw_df.head(50), use_container_width=True)

with st.expander("Preview aggregated SPC chart data (all available dates)"):
    st.dataframe(chart_df.head(50), use_container_width=True)

with st.expander("Preview aggregated SPC chart data (selected period)"):
    st.dataframe(period_chart_df.head(50), use_container_width=True)

# ============================================================
# Footer Notes
# ============================================================
st.markdown("---")
with st.expander("Reference: SPC logic, rules & behavior (details)", expanded=False):
    st.markdown(
        """
### Automatic SPC Chart Detection Logic

#### Continuous charts
- **I-MR** is enabled for **continuous data with one observation per time point**.
- **X̄-R** is enabled for **continuous subgrouped data with subgroup sizes from 2 to 10**.
- **X̄-S** is enabled for **continuous subgrouped data with subgroup sizes from 11 to 25**.

#### Attribute charts
- **np** is enabled for **counts of defectives** where the **sample size is constant**.
- **p** is enabled for **proportions defective** where the **sample size may vary or remain constant**.
- **c** is enabled for **counts of defects** where the **inspection opportunity is constant**.
- **u** is enabled for **defects per unit** where the **inspection opportunity may vary or remain constant**.

### Empty Row Handling
- The app asks the user whether to:
  1. **Replace empty entries with zeros**, or
  2. **Disregard rows with empty required fields**
- The app then applies that choice and only makes a note of what was done.
- Rows with empty time values are always disregarded because time cannot be zero-filled for SPC time-series analysis.

### Time Period Selection
- The user selects a **start date** and **end date** for the main SPC chart.
- By default, the main SPC chart date selectors attempt to use the period from **12 months ago up to today**, constrained to the available data range.
- The main SPC chart, KPI metrics, and violation summary use only that selected period.
- Breakdown charts are shown either **by year** or **by quarter**, based only on observations inside the selected period.

### SPC Rules Applied on the Primary Chart
1. **Rule 1**: One point is more than 3 standard deviations from the mean.
2. **Rule 2**: Nine (or more) points in a row are on the same side of the mean.
3. **Rule 3**: Six (or more) points in a row are continually increasing or decreasing.
4. **Rule 4**: Fourteen (or more) points in a row alternate in direction, increasing then decreasing.
5. **Rule 5**: Two (or three) out of three points in a row are more than 2 standard deviations from the mean in the same direction.
6. **Rule 6**: Four (or five) out of five points in a row are more than 1 standard deviation from the mean in the same direction.
7. **Rule 7**: Fifteen points in a row are all within 1 standard deviation of the mean on either side of the mean.
8. **Rule 8**: Eight points in a row exist, but none within 1 standard deviation of the mean, and the points are in both directions from the mean.

### Marker Behavior
- Each rule uses **one consistent marker color**, regardless of direction.
- **No blue** is used for violation markers.
- If **multiple rules** occur at the same point, an additional **red diamond-X marker** is overlaid.
- Continuous secondary charts (MR, R, S) also flag **control limit breaches**.

### Sigma Reference Lines
- For charts where sigma is available on the primary chart, the app now shows **horizontal 1σ and 2σ reference lines**.
- The **1σ and 2σ lines use the same color**.
- For charts with varying sigma by point (such as **p** and **u** charts), the displayed sigma reference lines use the **average sigma** to keep the references horizontal. This does **not** change the underlying SPC calculations or rule logic.
"""
    )
