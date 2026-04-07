from __future__ import annotations

import calendar
import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ============================================================
# Streamlit App Configuration
# ============================================================
APP_TITLE = "📈 SPC App: I‑MR, Xbar‑R, Xbar‑S"
APP_SUBTITLE = (
    "Upload your data, map columns, auto-detect valid charts, and visualize SPC rule "
    "detections with consistent indicators and review-friendly outputs."
)

st.set_page_config(
    page_title="SPC Charts (I-MR, Xbar-R, Xbar-S) with Rules",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Constants
# ============================================================

# ---- SPC Constants: Xbar-R
A2 = {
    2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308,
    11: 0.285, 12: 0.266, 13: 0.249, 14: 0.235, 15: 0.223, 16: 0.212, 17: 0.203, 18: 0.194,
    19: 0.187, 20: 0.180, 21: 0.173, 22: 0.167, 23: 0.162, 24: 0.157, 25: 0.153,
}
D3 = {
    2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.000, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223,
    11: 0.256, 12: 0.283, 13: 0.307, 14: 0.328, 15: 0.347, 16: 0.363, 17: 0.378, 18: 0.391,
    19: 0.403, 20: 0.415, 21: 0.425, 22: 0.434, 23: 0.443, 24: 0.451, 25: 0.459,
}
D4 = {
    2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777,
    11: 1.744, 12: 1.717, 13: 1.693, 14: 1.672, 15: 1.653, 16: 1.637, 17: 1.622, 18: 1.608,
    19: 1.596, 20: 1.585, 21: 1.575, 22: 1.566, 23: 1.557, 24: 1.548, 25: 1.541,
}

# ---- SPC Constants: I-MR
D2 = {2: 1.128}

# ---- SPC Constants: Xbar-S
A3 = {
    2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975,
    11: 0.927, 12: 0.886, 13: 0.850, 14: 0.817, 15: 0.789, 16: 0.763, 17: 0.739, 18: 0.718,
    19: 0.698, 20: 0.680, 21: 0.663, 22: 0.647, 23: 0.633, 24: 0.619, 25: 0.606,
}
B3 = {
    2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.030, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284,
    11: 0.328, 12: 0.354, 13: 0.377, 14: 0.399, 15: 0.419, 16: 0.437, 17: 0.454, 18: 0.469,
    19: 0.483, 20: 0.495, 21: 0.507, 22: 0.517, 23: 0.527, 24: 0.536, 25: 0.544,
}
B4 = {
    2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716,
    11: 1.682, 12: 1.649, 13: 1.618, 14: 1.590, 15: 1.565, 16: 1.541, 17: 1.519, 18: 1.499,
    19: 1.480, 20: 1.462, 21: 1.446, 22: 1.431, 23: 1.417, 24: 1.404, 25: 1.392,
}

# ---- Rule Style & Display Metadata
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
        "label": "Secondary limit breach",
    },
    "Multiple rules": {"color": "#FF0000", "label": "Multiple rules"},
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
    "Rule 8": "Eight points in a row exist, but none are within 1 standard deviation of the mean, and the points are in both directions from the mean.",
    "Secondary chart: point beyond control limit": "A point on the secondary chart is beyond the control limit.",
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
    "Secondary chart: point beyond control limit": 9,
    "Multiple rules": 10,
}

SUPPORTED_UPLOAD_TYPES = ["csv", "xlsx", "xls"]
PLOT_HEIGHT = 780
PLOT_WIDTH_DEFAULT = 1000
PLOT_HEIGHT_DEFAULT = 780

NULL_TREATMENT_OPTIONS = {
    "Discard null/empty measurement observations": "discard",
    "Make null/empty measurement observations zero": "zero",
}

BACKTRACK_OPTIONS = {
    "Backtrack over all periods": "all_periods",
    "Backtrack for the same period": "same_period",
}

# ============================================================
# Data Structures
# ============================================================
@dataclass(frozen=True)
class ChartEvaluation:
    valid_options: list[str]
    messages: list[str]


# ============================================================
# Generic Helpers
# ============================================================
@lru_cache(maxsize=None)
def is_excel(file_name: str) -> bool:
    """Return True if the uploaded file is an Excel file."""
    return file_name.lower().endswith((".xlsx", ".xlsm", ".xls", ".xltx", ".xltm"))


@lru_cache(maxsize=None)
def get_excel_engine(file_name: str) -> str:
    """Return the appropriate pandas engine for the uploaded Excel extension."""
    if file_name.lower().endswith(".xls"):
        return "xlrd"
    return "openpyxl"


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric, invalid values become NaN."""
    return pd.to_numeric(series, errors="coerce")


def parse_date(series: pd.Series) -> pd.Series:
    """Parse a Series to datetime, invalid values become NaT."""
    return pd.to_datetime(series, errors="coerce")


def all_groups_at_least_two(counts: pd.Series) -> bool:
    """Check whether all subgroup counts are at least 2."""
    return (counts >= 2).all()


@lru_cache(maxsize=None)
def supported_n(n: int) -> bool:
    """Check whether subgroup size n is supported by all needed SPC constant tables."""
    return (
        n in A2 and n in D3 and n in D4 and
        n in A3 and n in B3 and n in B4
    )


def repeat_line(value: float, length: int) -> np.ndarray:
    """Repeat a scalar value to create a line array."""
    return np.repeat(float(value), length)


def as_array(values: Any, length: int) -> np.ndarray:
    """
    Convert scalar/list/ndarray input to an array of the expected length.
    Scalars are repeated, None becomes all-NaN, arrays must match the expected length.
    """
    if np.isscalar(values) or values is None:
        return np.repeat(np.nan if values is None else float(values), length)

    arr = np.asarray(values, dtype=float)
    if len(arr) != length:
        raise ValueError("Input array length mismatch.")
    return arr


def empty_violations_df() -> pd.DataFrame:
    """Return a standard empty violations DataFrame."""
    return pd.DataFrame(columns=["date", "rule", "value", "rule_description"])


def format_metric_value(value: float | int | None, decimals: int = 5) -> str:
    """Format a numeric metric for UI display."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"


@lru_cache(maxsize=None)
def format_period_label(period: pd.Period, granularity: str) -> str:
    """Format a period label for display."""
    if granularity == "yearly":
        return str(period)
    if granularity == "quarterly":
        return f"{period.year} Q{period.quarter}"
    if granularity == "monthly":
        return period.strftime("%Y-%m")
    return str(period)


def format_focus_label(granularity: str, focus_value: int | None) -> str | None:
    """Format a selected focus value (month/quarter) for display."""
    if focus_value is None:
        return None
    if granularity == "quarterly":
        return f"Quarter {int(focus_value)}"
    if granularity == "monthly":
        return calendar.month_name[int(focus_value)]
    return None


def _get_uploaded_file_bytes(uploaded_file) -> bytes:
    """Read uploaded file into raw bytes."""
    uploaded_file.seek(0)
    return uploaded_file.getvalue()


def normalize_timestamp_to_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalize a timestamp to midnight for date-only comparisons."""
    return pd.Timestamp(ts).normalize()


def date_range_to_full_day_bounds(start_date: Any, end_date: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert date-like inputs to inclusive full-day timestamp bounds."""
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return start_ts, end_ts


@st.cache_data(show_spinner=False)
def get_valid_date_bounds(df_work: pd.DataFrame, date_col: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return min/max valid timestamps for the selected date column."""
    valid_dates = pd.to_datetime(df_work[date_col], errors="coerce").dropna().sort_values()
    if valid_dates.empty:
        return None, None
    return pd.Timestamp(valid_dates.iloc[0]), pd.Timestamp(valid_dates.iloc[-1])


@st.cache_data(show_spinner=False)
def get_default_imr_date_window(df_work: pd.DataFrame, date_col: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Return the default I-MR display window:
    - most recent year of data if possible
    - otherwise the full available date span
    """
    min_dt, max_dt = get_valid_date_bounds(df_work, date_col)
    if min_dt is None or max_dt is None:
        return None, None

    default_start = max(min_dt, max_dt - pd.DateOffset(years=1) + pd.Timedelta(days=1))
    return pd.Timestamp(default_start), pd.Timestamp(max_dt)


@st.cache_data(show_spinner=False)
def filter_df_by_date_range(
    df_work: pd.DataFrame,
    date_col: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Filter a dataframe to an inclusive timestamp range on the selected date column."""
    working_dates = pd.to_datetime(df_work[date_col], errors="coerce")
    mask = working_dates.between(start_ts, end_ts, inclusive="both")
    return df_work.loc[mask].copy()


def concat_violations(
    primary_violations: pd.DataFrame,
    secondary_violations: pd.DataFrame,
) -> pd.DataFrame:
    """Combine primary and secondary violations into a single DataFrame."""
    frames = []
    if primary_violations is not None and not primary_violations.empty:
        frames.append(primary_violations[["date", "rule"]].copy())
    if secondary_violations is not None and not secondary_violations.empty:
        frames.append(secondary_violations[["date", "rule"]].copy())
    if not frames:
        return pd.DataFrame(columns=["date", "rule"])
    return pd.concat(frames, ignore_index=True)


def get_most_common_rule(
    primary_violations: pd.DataFrame,
    secondary_violations: pd.DataFrame,
) -> str | None:
    """
    Determine the single most common rule across the figure.
    Ties are broken by RULE_SORT_ORDER.
    """
    all_violations = concat_violations(primary_violations, secondary_violations)
    if all_violations.empty:
        return None

    counts = (
        all_violations
        .groupby("rule", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    counts["sort_order"] = counts["rule"].map(lambda x: RULE_SORT_ORDER.get(x, 999))
    counts = counts.sort_values(["count", "sort_order"], ascending=[False, True]).reset_index(drop=True)

    return str(counts.loc[0, "rule"])


@st.cache_data(show_spinner=False)
def get_available_focus_values(df_work: pd.DataFrame, date_col: str, granularity: str) -> list[int]:
    """Return the available month or quarter values present in the dataset."""
    valid_dates = pd.to_datetime(df_work[date_col], errors="coerce").dropna()
    if valid_dates.empty:
        return []

    if granularity == "quarterly":
        return sorted(valid_dates.dt.quarter.dropna().astype(int).unique().tolist())
    if granularity == "monthly":
        return sorted(valid_dates.dt.month.dropna().astype(int).unique().tolist())
    return []


@st.cache_data(show_spinner=False)
def count_available_period_occurrences(
    df_work: pd.DataFrame,
    date_col: str,
    granularity: str,
    backtrack_mode: str = "all_periods",
    focus_value: int | None = None,
) -> int:
    """Count the number of available periods for the requested mode."""
    periods = get_selected_periods(
        df_work=df_work,
        date_col=date_col,
        granularity=granularity,
        requested_count=10_000_000,  # effectively all
        backtrack_mode=backtrack_mode,
        focus_value=focus_value,
    )
    return len(periods)


# ============================================================
# File Loading
# ============================================================
@st.cache_data(show_spinner=False)
def get_excel_sheet_names(file_bytes: bytes, file_name: str) -> list[str]:
    """Return Excel sheet names from raw uploaded bytes."""
    engine = get_excel_engine(file_name)
    xls = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def load_excel_sheet_from_bytes(file_bytes: bytes, file_name: str, sheet_name: str) -> pd.DataFrame:
    """Load a specific Excel sheet from raw uploaded bytes."""
    engine = get_excel_engine(file_name)
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine=engine)


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Load CSV from raw uploaded bytes."""
    return pd.read_csv(io.BytesIO(file_bytes))


def load_uploaded_file(uploaded_file) -> pd.DataFrame | None:
    """Load CSV or Excel data into a DataFrame."""
    if uploaded_file is None:
        return None

    try:
        file_bytes = _get_uploaded_file_bytes(uploaded_file)

        if is_excel(uploaded_file.name):
            sheet_names = get_excel_sheet_names(file_bytes, uploaded_file.name)
            sheet_name = st.sidebar.selectbox("2) Choose a sheet", options=sheet_names)
            return load_excel_sheet_from_bytes(file_bytes, uploaded_file.name, sheet_name)

        return load_csv_from_bytes(file_bytes)

    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return None


# ============================================================
# SPC Rule Detection Helpers
# ============================================================
def append_rule_hits(
    violations: list[dict[str, Any]],
    dates: np.ndarray,
    values: np.ndarray,
    indices: set[int] | np.ndarray | list[int],
    rule_name: str,
) -> None:
    """Append rule hits to the violations list with de-duplication by caller."""
    for idx in sorted(set(indices)):
        if idx < 0 or idx >= len(values):
            continue

        value = values[idx]
        if pd.isna(value):
            continue

        violations.append(
            {
                "date": pd.to_datetime(dates[idx]),
                "rule": rule_name,
                "value": value,
                "rule_description": RULE_DISPLAY_TEXT.get(rule_name, rule_name),
            }
        )


def mark_run_same_side(values: np.ndarray, center_line: np.ndarray, min_run_len: int) -> set[int]:
    """Mark consecutive points on the same side of the center line."""
    values = np.asarray(values, dtype=float)
    center_line = np.asarray(center_line, dtype=float)

    side = np.where(values > center_line, 1, np.where(values < center_line, -1, 0))
    flagged: set[int] = set()

    i = 0
    n = len(side)
    while i < n:
        if side[i] == 0:
            i += 1
            continue

        j = i
        while j + 1 < n and side[j + 1] == side[i]:
            j += 1

        if (j - i + 1) >= min_run_len:
            flagged.update(range(i, j + 1))

        i = j + 1

    return flagged


def mark_monotonic_runs(values: np.ndarray, min_points: int) -> set[int]:
    """Mark points in monotonic increasing or decreasing runs."""
    flagged: set[int] = set()
    if len(values) < min_points:
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


def mark_alternating_runs(values: np.ndarray, min_points: int) -> set[int]:
    """Mark points in alternating up/down runs."""
    flagged: set[int] = set()
    if len(values) < min_points:
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


# ============================================================
# SPC Rule Detection
# ============================================================
@st.cache_data(show_spinner=False)
def detect_spc_rule_violations(
    df: pd.DataFrame,
    y_col: str,
    cl: float | np.ndarray,
    sigma: float | np.ndarray,
) -> pd.DataFrame:
    """
    Detect SPC rule violations for a primary chart.
    Core rule logic preserved from the original implementation.
    """
    values = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    dates = pd.to_datetime(df["date"]).to_numpy()

    n = len(values)
    if n == 0:
        return empty_violations_df()

    cl_arr = as_array(cl, n)
    sigma_arr = as_array(sigma, n)

    if np.all(np.isnan(sigma_arr)) or np.nanmax(np.abs(sigma_arr)) == 0:
        return empty_violations_df()

    violations: list[dict[str, Any]] = []

    upper_1 = cl_arr + sigma_arr
    lower_1 = cl_arr - sigma_arr
    upper_2 = cl_arr + 2 * sigma_arr
    lower_2 = cl_arr - 2 * sigma_arr
    upper_3 = cl_arr + 3 * sigma_arr
    lower_3 = cl_arr - 3 * sigma_arr

    # Rule 1
    rule1_idx = np.where((values > upper_3) | (values < lower_3))[0]
    append_rule_hits(violations, dates, values, rule1_idx, "Rule 1")

    # Rule 2
    rule2_idx = mark_run_same_side(values, cl_arr, min_run_len=9)
    append_rule_hits(violations, dates, values, rule2_idx, "Rule 2")

    # Rule 3
    rule3_idx = mark_monotonic_runs(values, min_points=6)
    append_rule_hits(violations, dates, values, rule3_idx, "Rule 3")

    # Rule 4
    rule4_idx = mark_alternating_runs(values, min_points=14)
    append_rule_hits(violations, dates, values, rule4_idx, "Rule 4")

    # Rule 5
    rule5_idx: set[int] = set()
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

    append_rule_hits(violations, dates, values, rule5_idx, "Rule 5")

    # Rule 6
    rule6_idx: set[int] = set()
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

    append_rule_hits(violations, dates, values, rule6_idx, "Rule 6")

    # Rule 7
    rule7_idx: set[int] = set()
    within_1 = np.abs(values - cl_arr) <= sigma_arr

    i = 0
    while i < n:
        if not within_1[i] or pd.isna(values[i]):
            i += 1
            continue

        j = i
        while j + 1 < n and within_1[j + 1] and not pd.isna(values[j + 1]):
            j += 1

        if (j - i + 1) >= 15:
            rule7_idx.update(range(i, j + 1))

        i = j + 1

    append_rule_hits(violations, dates, values, rule7_idx, "Rule 7")

    # Rule 8
    rule8_idx: set[int] = set()
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

    append_rule_hits(violations, dates, values, rule8_idx, "Rule 8")

    violations_df = pd.DataFrame(violations)
    if violations_df.empty:
        return empty_violations_df()

    return (
        violations_df
        .drop_duplicates(subset=["date", "rule"])
        .sort_values(
            ["date", "rule"],
            key=lambda s: s.map(RULE_SORT_ORDER) if s.name == "rule" else s,
        )
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def detect_secondary_limit_breaches(
    df: pd.DataFrame,
    y_col: str,
    ucl: float | np.ndarray,
    lcl: float | np.ndarray,
) -> pd.DataFrame:
    """
    Detect secondary chart points beyond control limits.
    Core logic preserved from the original implementation.
    """
    values = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    dates = pd.to_datetime(df["date"]).to_numpy()

    n = len(values)
    if n == 0:
        return empty_violations_df()

    ucl_arr = as_array(ucl, n)
    lcl_arr = as_array(lcl, n)

    violations: list[dict[str, Any]] = []
    for i, value in enumerate(values):
        if pd.isna(value):
            continue

        if value > ucl_arr[i] or value < lcl_arr[i]:
            violations.append(
                {
                    "date": pd.to_datetime(dates[i]),
                    "rule": "Secondary chart: point beyond control limit",
                    "value": value,
                    "rule_description": RULE_DISPLAY_TEXT["Secondary chart: point beyond control limit"],
                }
            )

    violations_df = pd.DataFrame(violations)
    if violations_df.empty:
        return empty_violations_df()

    return (
        violations_df
        .drop_duplicates(subset=["date", "rule"])
        .sort_values(
            ["date", "rule"],
            key=lambda s: s.map(RULE_SORT_ORDER) if s.name == "rule" else s,
        )
        .reset_index(drop=True)
    )


# ============================================================
# Chart Data Builders
# ============================================================
@st.cache_data(show_spinner=False)
def build_imr_chart_df(df: pd.DataFrame, measurement_col: str, date_col: str | None) -> pd.DataFrame:
    """Build chart-ready data for I-MR."""
    if date_col:
        chart_df = df[[date_col, measurement_col]].dropna().copy()
        chart_df[date_col] = parse_date(chart_df[date_col])
        chart_df = chart_df.sort_values(by=date_col)
        chart_df = chart_df.rename(columns={date_col: "date", measurement_col: "value"})
    else:
        chart_df = df[[measurement_col]].dropna().copy().reset_index(drop=True)
        chart_df = chart_df.rename(columns={measurement_col: "value"})
        # Synthetic datetime for rule engine compatibility
        chart_df["date"] = pd.to_datetime(np.arange(len(chart_df)), unit="D", origin="unix")

    chart_df["MR"] = chart_df["value"].diff().abs()
    chart_df = chart_df.reset_index(drop=True)
    chart_df["Index"] = np.arange(1, len(chart_df) + 1)
    chart_df["subgroup_number"] = np.arange(1, len(chart_df) + 1)
    return chart_df


@st.cache_data(show_spinner=False)
def build_xbar_r_chart_df(df: pd.DataFrame, measurement_col: str, subgroup_col: str) -> pd.DataFrame:
    """Build chart-ready data for Xbar-R."""
    grouped = df[[subgroup_col, measurement_col]].dropna().groupby(subgroup_col, sort=False)
    stats = (
        grouped
        .agg(
            n=(measurement_col, "count"),
            xbar=(measurement_col, "mean"),
            min_=(measurement_col, "min"),
            max_=(measurement_col, "max"),
        )
        .reset_index()
    )
    stats["R"] = stats["max_"] - stats["min_"]
    stats = stats.rename(columns={subgroup_col: "subgroup"})
    stats["date"] = pd.to_datetime(np.arange(len(stats)), unit="D", origin="unix")
    stats["subgroup_number"] = np.arange(1, len(stats) + 1)
    return stats


@st.cache_data(show_spinner=False)
def build_xbar_s_chart_df(df: pd.DataFrame, measurement_col: str, subgroup_col: str) -> pd.DataFrame:
    """Build chart-ready data for Xbar-S."""
    grouped = df[[subgroup_col, measurement_col]].dropna().groupby(subgroup_col, sort=False)
    stats = grouped.agg(n=(measurement_col, "count"), xbar=(measurement_col, "mean")).reset_index()
    s_values = grouped[measurement_col].apply(lambda x: np.std(x, ddof=1)).reset_index(name="S")
    stats = stats.merge(s_values, on=subgroup_col)
    stats = stats.rename(columns={subgroup_col: "subgroup"})
    stats["date"] = pd.to_datetime(np.arange(len(stats)), unit="D", origin="unix")
    stats["subgroup_number"] = np.arange(1, len(stats) + 1)
    return stats


# ============================================================
# Limit Calculations
# ============================================================
@st.cache_data(show_spinner=False)
def calc_limits_imr(chart_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calculate control limits for I-MR."""
    values = chart_df["value"].to_numpy()
    moving_ranges = chart_df["MR"].dropna().to_numpy()

    xbar = np.mean(values)
    mrbar = np.mean(moving_ranges) if len(moving_ranges) > 0 else 0.0
    sigma = mrbar / D2[2] if D2[2] != 0 else 0.0

    ucl = xbar + 3 * sigma
    lcl = max(0.0, xbar - 3 * sigma) if np.all(values >= 0) else xbar - 3 * sigma
    n_points = len(chart_df)

    return {
        "primary": {
            "label": "Individuals",
            "y_col": "value",
            "CL": xbar,
            "UCL": ucl,
            "LCL": lcl,
            "CL_series": repeat_line(xbar, n_points),
            "UCL_series": repeat_line(ucl, n_points),
            "LCL_series": repeat_line(lcl, n_points),
            "sigma": sigma,
            "sigma_series": repeat_line(sigma, n_points),
        },
        "secondary": {
            "label": "Moving Range",
            "y_col": "MR",
            "CL": mrbar,
            "UCL": D4[2] * mrbar,
            "LCL": max(0.0, D3[2] * mrbar),
            "CL_series": repeat_line(mrbar, n_points),
            "UCL_series": repeat_line(D4[2] * mrbar, n_points),
            "LCL_series": repeat_line(max(0.0, D3[2] * mrbar), n_points),
            "sigma": None,
            "sigma_series": None,
        },
    }


@st.cache_data(show_spinner=False)
def calc_limits_xbar_r(chart_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calculate control limits for Xbar-R."""
    xbarbar = chart_df["xbar"].mean()
    rbar = chart_df["R"].mean()

    n_arr = chart_df["n"].astype(int).to_numpy()
    a2_arr = np.array([A2[n] if n in A2 else np.nan for n in n_arr], dtype=float)
    d3_arr = np.array([D3[n] if n in D3 else np.nan for n in n_arr], dtype=float)
    d4_arr = np.array([D4[n] if n in D4 else np.nan for n in n_arr], dtype=float)

    ucl_x = xbarbar + a2_arr * rbar
    lcl_x = xbarbar - a2_arr * rbar
    sigma_x = (a2_arr * rbar) / 3.0

    ucl_r = d4_arr * rbar
    lcl_r = d3_arr * rbar

    return {
        "primary": {
            "label": "Xbar",
            "y_col": "xbar",
            "CL": xbarbar,
            "UCL": float(np.nanmean(ucl_x)),
            "LCL": float(np.nanmean(lcl_x)),
            "CL_series": repeat_line(xbarbar, len(chart_df)),
            "UCL_series": ucl_x,
            "LCL_series": lcl_x,
            "sigma": float(np.nanmean(sigma_x)),
            "sigma_series": sigma_x,
        },
        "secondary": {
            "label": "Range",
            "y_col": "R",
            "CL": rbar,
            "UCL": float(np.nanmean(ucl_r)),
            "LCL": float(np.nanmean(lcl_r)),
            "CL_series": repeat_line(rbar, len(chart_df)),
            "UCL_series": ucl_r,
            "LCL_series": lcl_r,
            "sigma": None,
            "sigma_series": None,
        },
    }


@st.cache_data(show_spinner=False)
def calc_limits_xbar_s(chart_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calculate control limits for Xbar-S."""
    xbarbar = chart_df["xbar"].mean()
    sbar = chart_df["S"].mean()

    n_arr = chart_df["n"].astype(int).to_numpy()
    a3_arr = np.array([A3[n] if n in A3 else np.nan for n in n_arr], dtype=float)
    b3_arr = np.array([B3[n] if n in B3 else np.nan for n in n_arr], dtype=float)
    b4_arr = np.array([B4[n] if n in B4 else np.nan for n in n_arr], dtype=float)

    ucl_x = xbarbar + a3_arr * sbar
    lcl_x = xbarbar - a3_arr * sbar
    sigma_x = (a3_arr * sbar) / 3.0

    ucl_s = b4_arr * sbar
    lcl_s = b3_arr * sbar

    return {
        "primary": {
            "label": "Xbar",
            "y_col": "xbar",
            "CL": xbarbar,
            "UCL": float(np.nanmean(ucl_x)),
            "LCL": float(np.nanmean(lcl_x)),
            "CL_series": repeat_line(xbarbar, len(chart_df)),
            "UCL_series": ucl_x,
            "LCL_series": lcl_x,
            "sigma": float(np.nanmean(sigma_x)),
            "sigma_series": sigma_x,
        },
        "secondary": {
            "label": "Std Dev",
            "y_col": "S",
            "CL": sbar,
            "UCL": float(np.nanmean(ucl_s)),
            "LCL": float(np.nanmean(lcl_s)),
            "CL_series": repeat_line(sbar, len(chart_df)),
            "UCL_series": ucl_s,
            "LCL_series": lcl_s,
            "sigma": None,
            "sigma_series": None,
        },
    }


# ============================================================
# Plot Helpers
# ============================================================
def add_limit_lines(
    fig: go.Figure,
    x_values: pd.Series | np.ndarray,
    cl: float | np.ndarray,
    ucl: float | np.ndarray,
    lcl: float | np.ndarray,
    sigma: float | np.ndarray | None,
    row: int,
    col: int,
    show_legend_once: bool = False,
) -> None:
    """Add center line, sigma reference lines, and control limits to a subplot."""
    n = len(x_values)
    cl_arr = as_array(cl, n)
    ucl_arr = as_array(ucl, n)
    lcl_arr = as_array(lcl, n)

    # Center Line
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=cl_arr,
            mode="lines",
            name="Center Line",
            legendgroup="center_line",
            showlegend=show_legend_once,
            line=dict(color="#228B22", dash="dash"),
        ),
        row=row,
        col=col,
    )

    # Sigma Reference Lines (visual only; based on mean sigma)
    sigma_arr = as_array(sigma, n)
    if not np.all(np.isnan(sigma_arr)) and np.nanmax(np.abs(sigma_arr)) != 0:
        sigma_ref = float(np.nanmean(sigma_arr))
        cl_ref = float(np.nanmean(cl_arr))

        upper_1_arr = repeat_line(cl_ref + sigma_ref, n)
        lower_1_arr = repeat_line(cl_ref - sigma_ref, n)
        upper_2_arr = repeat_line(cl_ref + 2 * sigma_ref, n)
        lower_2_arr = repeat_line(cl_ref - 2 * sigma_ref, n)

        sigma_line_color = "#6A5ACD"

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=upper_1_arr,
                mode="lines",
                name="Upper 1 Sigma",
                legendgroup="sigma_1",
                showlegend=show_legend_once,
                line=dict(color=sigma_line_color, dash="dash"),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=lower_1_arr,
                mode="lines",
                name="Lower 1 Sigma",
                legendgroup="sigma_1",
                showlegend=False,
                line=dict(color=sigma_line_color, dash="dash"),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=upper_2_arr,
                mode="lines",
                name="Upper 2 Sigma",
                legendgroup="sigma_2",
                showlegend=show_legend_once,
                line=dict(color=sigma_line_color, dash="dot"),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=lower_2_arr,
                mode="lines",
                name="Lower 2 Sigma",
                legendgroup="sigma_2",
                showlegend=False,
                line=dict(color=sigma_line_color, dash="dot"),
            ),
            row=row,
            col=col,
        )

    # UCL / LCL
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=ucl_arr,
            mode="lines",
            name="Upper Control Limit",
            legendgroup="ucl",
            showlegend=show_legend_once,
            line=dict(color="#B22222", dash="dot"),
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=lcl_arr,
            mode="lines",
            name="Lower Control Limit",
            legendgroup="lcl",
            showlegend=show_legend_once,
            line=dict(color="#B22222", dash="dot"),
        ),
        row=row,
        col=col,
    )


def add_rule_markers(
    fig: go.Figure,
    source_df: pd.DataFrame,
    violations_df: pd.DataFrame,
    y_col: str,
    row: int,
    col: int,
    legend_shown_rules: set[str],
    x_col: str,
    x_axis_mode: str,
    default_visible_rule: str | None,
) -> set[str]:
    """
    Add rule markers and multi-rule segmented overlays to a subplot.

    Updated behavior:
    - only the single most common rule for the whole figure is visible by default
    - all other rule traces (including "Multiple rules") start as legend-only
    """
    if violations_df.empty:
        return legend_shown_rules

    merge_cols = ["date", y_col]
    if x_col not in merge_cols:
        merge_cols.append(x_col)

    merged = (
        source_df[merge_cols]
        .merge(
            violations_df[["date", "rule", "rule_description"]].drop_duplicates(),
            on="date",
            how="inner",
        )
        .drop_duplicates(subset=["date", "rule"])
        .sort_values(
            ["rule", "date"],
            key=lambda s: s.map(RULE_SORT_ORDER) if s.name == "rule" else s,
        )
    )

    multi_rule_df = (
        violations_df
        .groupby("date", as_index=False)
        .agg(
            rule_count=("rule", "nunique"),
            rules=("rule", lambda x: sorted(set(x), key=lambda r: RULE_SORT_ORDER.get(r, 999))),
        )
    )
    multi_rule_df = multi_rule_df[multi_rule_df["rule_count"] > 1].copy()

    # Single-rule markers
    for rule_name in merged["rule"].unique():
        rule_points = merged[merged["rule"] == rule_name].copy()
        style = RULE_STYLE_MAP.get(rule_name, DEFAULT_RULE_STYLE)
        show_legend = rule_name not in legend_shown_rules
        visible_state = True if rule_name == default_visible_rule else "legendonly"

        if x_axis_mode == "Time":
            x_vals = rule_points["date"]
            hovertemplate = (
                "<b>%{x|%Y-%m-%d}</b><br>"
                f"Value: %{{y:.5f}}<br>"
                f"Rule: {rule_name}<br>"
                f"Description: {RULE_DISPLAY_TEXT.get(rule_name, rule_name)}"
                "<extra></extra>"
            )
        elif x_axis_mode == "Index":
            x_vals = rule_points[x_col]
            hovertemplate = (
                "<b>Index: %{x}</b><br>"
                f"Value: %{{y:.5f}}<br>"
                f"Rule: {rule_name}<br>"
                f"Description: {RULE_DISPLAY_TEXT.get(rule_name, rule_name)}"
                "<extra></extra>"
            )
        else:
            x_vals = rule_points[x_col]
            hovertemplate = (
                "<b>Subgroup #: %{x}</b><br>"
                f"Value: %{{y:.5f}}<br>"
                f"Rule: {rule_name}<br>"
                f"Description: {RULE_DISPLAY_TEXT.get(rule_name, rule_name)}"
                "<extra></extra>"
            )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=rule_points[y_col],
                mode="markers",
                name=style["label"],
                legendgroup=rule_name,
                showlegend=show_legend,
                visible=visible_state,
                marker=dict(
                    color=style["color"],
                    size=10,
                    symbol="x",
                    line=dict(width=2, color=style["color"]),
                ),
                hovertemplate=hovertemplate,
            ),
            row=row,
            col=col,
        )
        legend_shown_rules.add(rule_name)

    # Multi-rule segmented ring overlay
    if not multi_rule_df.empty:
        multi_rule_df = multi_rule_df.merge(source_df[merge_cols], on="date", how="left")

        style = RULE_STYLE_MAP["Multiple rules"]
        show_legend = "Multiple rules" not in legend_shown_rules
        multiple_visible_state = True if default_visible_rule == "Multiple rules" else "legendonly"

        hover_text = []
        for _, record in multi_rule_df.iterrows():
            rules_text = "<br>".join([f"- {rule}" for rule in record["rules"]])
            date_str = pd.to_datetime(record["date"]).strftime("%Y-%m-%d")

            if x_axis_mode == "Time":
                base = f"<b>{date_str}</b><br>"
            elif x_axis_mode == "Index":
                base = (
                    f"<b>Index: {int(record[x_col]) if pd.notna(record[x_col]) else ''}</b><br>"
                    f"Date: {date_str}<br>"
                )
            else:
                base = (
                    f"<b>Subgroup #: {int(record[x_col]) if pd.notna(record[x_col]) else ''}</b><br>"
                    f"Date: {date_str}<br>"
                )

            hover_text.append(
                base +
                f"Value: {record[y_col]:.5f}<br>"
                f"Multiple rules triggered:<br>{rules_text}"
            )

        x_vals = multi_rule_df["date"] if x_axis_mode == "Time" else multi_rule_df[x_col]

        # Invisible anchor for hover + legend
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=multi_rule_df[y_col],
                mode="markers",
                name=style["label"],
                legendgroup="Multiple rules",
                showlegend=show_legend,
                visible=multiple_visible_state,
                marker=dict(
                    size=12,
                    color="rgba(0,0,0,0)",
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                ),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
            ),
            row=row,
            col=col,
        )
        legend_shown_rules.add("Multiple rules")

        # Center cross
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=multi_rule_df[y_col],
                mode="markers",
                name=None,
                legendgroup="Multiple rules",
                showlegend=False,
                visible=multiple_visible_state,
                marker=dict(
                    symbol="x",
                    size=10,
                    color="#5B5B5B",
                    line=dict(width=2, color="#5B5B5B"),
                ),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Draw segmented rings
        ring_px = 12
        plot_h = int(fig.layout.height) if fig.layout.height else PLOT_HEIGHT_DEFAULT
        plot_w = int(fig.layout.width) if fig.layout.width else PLOT_WIDTH_DEFAULT

        y_vals = pd.to_numeric(source_df[y_col], errors="coerce")
        y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
        y_span = float(y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) != 0 else 1.0

        if x_axis_mode in ("Subgroup", "Index"):
            x_vals_num = pd.to_numeric(source_df[x_col], errors="coerce")
            x_min, x_max = np.nanmin(x_vals_num), np.nanmax(x_vals_num)
            x_span = float(x_max - x_min) if np.isfinite(x_max - x_min) and (x_max - x_min) != 0 else 1.0
        else:
            dts = pd.to_datetime(source_df["date"])
            if len(dts) > 1:
                x_span = (pd.to_datetime(dts.max()) - pd.to_datetime(dts.min())).total_seconds()
                x_span = x_span if x_span != 0 else 1.0
            else:
                x_span = 1.0

        r_y = ring_px / (plot_h / y_span)
        r_x = ring_px / (plot_w / x_span)

        for _, record in multi_rule_df.reset_index(drop=True).iterrows():
            rules_list = record["rules"]
            if not isinstance(rules_list, (list, tuple)) or len(rules_list) == 0:
                continue

            k = len(rules_list)
            x0 = record["date"] if x_axis_mode == "Time" else record[x_col]
            y0 = record[y_col]

            for j, rule_name in enumerate(rules_list):
                color = RULE_STYLE_MAP.get(rule_name, DEFAULT_RULE_STYLE)["color"]

                theta0 = 2 * np.pi * (j / k)
                theta1 = 2 * np.pi * ((j + 1) / k)
                thetas = np.linspace(theta0, theta1, 24)

                if x_axis_mode in ("Subgroup", "Index"):
                    seg_x = x0 + (r_x * np.cos(thetas))
                else:
                    offsets = r_x * np.cos(thetas)
                    seg_x = [
                        pd.to_datetime(x0) + pd.to_timedelta(seconds, unit="s")
                        for seconds in offsets
                    ]

                seg_y = y0 + (r_y * np.sin(thetas))

                fig.add_trace(
                    go.Scatter(
                        x=seg_x,
                        y=seg_y,
                        mode="lines",
                        name=None,
                        legendgroup="Multiple rules",
                        showlegend=False,
                        visible=multiple_visible_state,
                        line=dict(color=color, width=1),
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

    return legend_shown_rules


def plot_spc_chart(
    chart_df: pd.DataFrame,
    limits: dict[str, dict[str, Any]],
    title: str,
    primary_violations: pd.DataFrame,
    secondary_violations: pd.DataFrame,
    x_axis_mode: str = "Subgroup",
) -> go.Figure:
    """Create the SPC chart figure with primary and secondary subplots."""
    primary = limits["primary"]
    secondary = limits["secondary"]

    if x_axis_mode == "Time":
        x_col = "date"
        x_axis_title = "Date / Time"
    elif x_axis_mode == "Index":
        x_col = "Index"
        x_axis_title = "Index"
    else:
        x_col = "subgroup_number"
        x_axis_title = "Subgroup Number"

    default_visible_rule = get_most_common_rule(primary_violations, secondary_violations)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{primary['label']} Chart",
            f"{secondary['label']} Chart" if secondary else "",
        ),
    )

    # Primary series
    fig.add_trace(
        go.Scatter(
            x=chart_df[x_col],
            y=chart_df[primary["y_col"]],
            mode="lines+markers",
            name=primary["label"],
            legendgroup="primary_series",
            showlegend=True,
            line=dict(color="#5B5B5B", width=1.8),
            marker=dict(size=5, color="#5B5B5B"),
        ),
        row=1,
        col=1,
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
        show_legend_once=True,
    )

    legend_shown_rules: set[str] = set()
    legend_shown_rules = add_rule_markers(
        fig=fig,
        source_df=chart_df,
        violations_df=primary_violations,
        y_col=primary["y_col"],
        row=1,
        col=1,
        legend_shown_rules=legend_shown_rules,
        x_col=x_col,
        x_axis_mode=x_axis_mode,
        default_visible_rule=default_visible_rule,
    )

    # Secondary series
    if secondary is not None:
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
                marker=dict(size=5, color="#8C6D1F"),
            ),
            row=2,
            col=1,
        )

        add_limit_lines(
            fig=fig,
            x_values=sec_df[x_col],
            cl=secondary["CL_series"][: len(sec_df)],
            ucl=secondary["UCL_series"][: len(sec_df)],
            lcl=secondary["LCL_series"][: len(sec_df)],
            sigma=secondary["sigma_series"][: len(sec_df)] if secondary["sigma_series"] is not None else None,
            row=2,
            col=1,
            show_legend_once=False,
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
            x_axis_mode=x_axis_mode,
            default_visible_rule=default_visible_rule,
        )

    fig.update_layout(
        height=PLOT_HEIGHT,
        title=dict(text=title, x=0.5, xanchor="center", y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=140, b=40),
        hovermode="closest",
    )
    fig.update_yaxes(title_text=primary["label"], row=1, col=1)
    if secondary is not None:
        fig.update_yaxes(title_text=secondary["label"], row=2, col=1)
    fig.update_xaxes(title_text=x_axis_title, row=2, col=1)

    return fig


# ============================================================
# Validation & Processing
# ============================================================
@st.cache_data(show_spinner=False)
def clean_working_data(
    df: pd.DataFrame,
    measurement_col: str,
    date_col: str | None,
    subgroup_col: str | None,
    null_treatment: str,
) -> pd.DataFrame:
    """Prepare the working dataset based on selected mappings."""
    df_work = df.copy()
    df_work[measurement_col] = coerce_numeric(df_work[measurement_col])

    if null_treatment == "zero":
        df_work[measurement_col] = df_work[measurement_col].fillna(0)
    else:
        df_work = df_work.dropna(subset=[measurement_col])

    if date_col:
    df_work[date_col] = parse_date(df_work[date_col])

    # aggregate duplicate dates
    duplicate_counts = df_work[date_col].value_counts()
    has_duplicates = (duplicate_counts > 1).any()

    if has_duplicates:
        df_work = (
            df_work
            .groupby(date_col, as_index=False, sort=True)
            .agg({measurement_col: "sum"})
        )

        # annotate aggregation for UI
        df_work.attrs["dates_aggregated"] = True
        df_work.attrs["aggregated_date_count"] = int((duplicate_counts > 1).sum())
    else:
        df_work.attrs["dates_aggregated"] = False

    if subgroup_col:
        df_work[subgroup_col] = df_work[subgroup_col].astype(str)

    return df_work


@st.cache_data(show_spinner=False)
def evaluate_chart_validity(
    df_work: pd.DataFrame,
    measurement_col: str,
    subgroup_col: str | None,
) -> ChartEvaluation:
    """
    Determine which chart types are valid.
    Core validity rules preserved, including the original demo rule:
    - Xbar-R for 1–11 unique subgroup labels
    - Xbar-S for > 11 unique subgroup labels
    """
    messages: list[str] = []
    valid_options: list[str] = []

    # I-MR
    imr_valid = len(df_work) >= 3
    if imr_valid:
        valid_options.append("I-MR")
        messages.append("✅ **I‑MR**: valid (measurement present; ≥ 3 rows).")
    else:
        messages.append("❌ **I‑MR**: not enough data (need ≥ 3 measurements).")

    # Xbar-R / Xbar-S
    if subgroup_col is None:
        messages.append("❌ **Xbar‑R**: subgroup not selected.")
        messages.append("❌ **Xbar‑S**: subgroup not selected.")
        return ChartEvaluation(valid_options=valid_options, messages=messages)

    subgroup_counts = df_work.groupby(subgroup_col)[measurement_col].count().sort_index()
    unique_subgroups = subgroup_counts.shape[0]
    at_least_two_per_group = all_groups_at_least_two(subgroup_counts)
    at_least_two_groups = unique_subgroups >= 2

    # Preserve original demo rule exactly
    if 1 <= unique_subgroups <= 11:
        if not at_least_two_groups:
            messages.append(f"❌ **Xbar‑R**: only {unique_subgroups} subgroup(s); need ≥ 2 to chart.")
        elif not at_least_two_per_group:
            messages.append("❌ **Xbar‑R**: some subgroup(s) have < 2 observations (range undefined).")
        else:
            valid_options.append("Xbar-R")
            messages.append(f"✅ **Xbar‑R**: valid ({unique_subgroups} subgroup labels; each subgroup has ≥ 2).")
    else:
        messages.append(f"❌ **Xbar‑R**: requires 1–11 unique subgroups by your rule; found {unique_subgroups}.")

    if unique_subgroups > 11:
        if not at_least_two_per_group:
            messages.append("❌ **Xbar‑S**: some subgroup(s) have < 2 observations (std dev undefined).")
        else:
            valid_options.append("Xbar-S")
            messages.append("✅ **Xbar‑S**: valid (> 11 subgroup labels; each subgroup has ≥ 2).")
    else:
        messages.append(f"❌ **Xbar‑S**: requires > 11 unique subgroups by your rule; found {unique_subgroups}.")

    return ChartEvaluation(valid_options=valid_options, messages=messages)


@st.cache_data(show_spinner=False)
def detect_violations_for_chart(
    chart_df: pd.DataFrame,
    limits: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect primary SPC rule violations and secondary limit breaches."""
    primary_violations = detect_spc_rule_violations(
        chart_df,
        y_col=limits["primary"]["y_col"],
        cl=limits["primary"]["CL_series"],
        sigma=limits["primary"]["sigma_series"],
    )

    secondary_col = limits["secondary"]["y_col"]
    sec_df = chart_df.dropna(subset=[secondary_col]).copy()
    secondary_violations = detect_secondary_limit_breaches(
        sec_df,
        y_col=secondary_col,
        ucl=limits["secondary"]["UCL_series"][: len(sec_df)],
        lcl=limits["secondary"]["LCL_series"][: len(sec_df)],
    )
    return primary_violations, secondary_violations


@st.cache_data(show_spinner=False)
def check_unsupported_group_sizes(df_work: pd.DataFrame, measurement_col: str, subgroup_col: str) -> list[int]:
    """Return unsupported subgroup sizes for Xbar charts."""
    group_sizes = df_work.groupby(subgroup_col)[measurement_col].count()
    return sorted({int(n) for n in group_sizes.unique() if not supported_n(int(n))})


@st.cache_data(show_spinner=False)
def get_selected_periods(
    df_work: pd.DataFrame,
    date_col: str,
    granularity: str,
    requested_count: int,
    backtrack_mode: str = "all_periods",
    focus_value: int | None = None,
) -> list[pd.Period]:
    """
    Return the selected periods for additional I-MR chart creation.

    Modes:
    - all_periods: current behavior, take the most recent N distinct periods
    - same_period:
        * quarterly -> same quarter across years
        * monthly -> same month across years
        * yearly -> behaves the same as all_periods
    """
    freq_map = {"yearly": "Y", "quarterly": "Q", "monthly": "M"}
    freq = freq_map[granularity]

    valid_dates = pd.to_datetime(df_work[date_col], errors="coerce").dropna()
    if valid_dates.empty:
        return []

    periods = pd.Series(valid_dates.dt.to_period(freq).unique()).sort_values().tolist()

    if backtrack_mode == "same_period":
        if granularity == "quarterly" and focus_value is not None:
            periods = [p for p in periods if int(p.quarter) == int(focus_value)]
        elif granularity == "monthly" and focus_value is not None:
            periods = [p for p in periods if int(p.month) == int(focus_value)]
        # yearly intentionally falls back to normal behavior

    return periods[-requested_count:] if requested_count > 0 else []


@st.cache_data(show_spinner=False)
def get_imr_period_chart_payloads(
    df_work: pd.DataFrame,
    measurement_col: str,
    date_col: str,
    granularity: str,
    requested_count: int,
    backtrack_mode: str = "all_periods",
    focus_value: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build chart payloads for requested additional periods.

    Supports:
    - backtrack over all periods
    - backtrack for the same period (same quarter or same month across years)
    """
    freq_map = {"yearly": "Y", "quarterly": "Q", "monthly": "M"}
    working_dates = pd.to_datetime(df_work[date_col], errors="coerce")
    selected_periods = get_selected_periods(
        df_work=df_work,
        date_col=date_col,
        granularity=granularity,
        requested_count=requested_count,
        backtrack_mode=backtrack_mode,
        focus_value=focus_value,
    )

    if not selected_periods:
        return []

    freq = freq_map[granularity]
    payloads: list[dict[str, Any]] = []

    for period in selected_periods:
        period_mask = working_dates.dt.to_period(freq) == period
        period_df = df_work.loc[period_mask].copy()

        if period_df.empty:
            continue

        chart_df = build_imr_chart_df(period_df, measurement_col=measurement_col, date_col=date_col)

        if len(chart_df) < 3:
            payloads.append(
                {
                    "granularity": granularity,
                    "period": period,
                    "period_label": format_period_label(period, granularity),
                    "status": "skipped",
                    "message": (
                        f"Skipping {format_period_label(period, granularity)} because fewer than 3 "
                        f"valid observations remain for an I‑MR chart."
                    ),
                }
            )
            continue

        limits = calc_limits_imr(chart_df)
        primary_violations, secondary_violations = detect_violations_for_chart(chart_df, limits)

        fig = plot_spc_chart(
            chart_df=chart_df,
            limits=limits,
            title=f"I-MR Chart — {format_period_label(period, granularity)}",
            primary_violations=primary_violations,
            secondary_violations=secondary_violations,
            x_axis_mode="Time",
        )

        payloads.append(
            {
                "granularity": granularity,
                "period": period,
                "period_label": format_period_label(period, granularity),
                "status": "ready",
                "fig": fig,
                "limits": limits,
                "primary_violations": primary_violations,
                "secondary_violations": secondary_violations,
            }
        )

    return payloads


def render_imr_periodic_options(df_work: pd.DataFrame, date_col: str) -> dict[str, dict[str, Any]]:
    """Render post-violation controls for additional yearly/quarterly/monthly I-MR charts."""
    st.markdown("### Additional time-based I‑MR charts")

    create_periodic = st.checkbox(
        "Create yearly / quarterly / monthly I‑MR charts as well",
        value=False,
    )

    if not create_periodic:
        return {}

    backtrack_mode_label = st.radio(
        "How should the additional charts backtrack?",
        options=list(BACKTRACK_OPTIONS.keys()),
        index=0,
        horizontal=True,
        key="imr_backtrack_mode",
    )
    backtrack_mode = BACKTRACK_OPTIONS[backtrack_mode_label]

    selections = st.multiselect(
        "Select period types",
        options=["yearly", "quarterly", "monthly"],
        default=[],
    )

    period_requests: dict[str, dict[str, Any]] = {}
    freq_map = {"yearly": "Y", "quarterly": "Q", "monthly": "M"}

    for granularity in selections:
        available_periods_total = (
            pd.to_datetime(df_work[date_col], errors="coerce")
            .dropna()
            .dt.to_period(freq_map[granularity])
            .nunique()
        )

        if available_periods_total == 0:
            st.info(f"No valid dates are available for {granularity} chart creation.")
            continue

        focus_value: int | None = None
        focus_label: str | None = None
        available_periods = int(available_periods_total)

        if backtrack_mode == "same_period" and granularity in ["quarterly", "monthly"]:
            focus_options = get_available_focus_values(df_work, date_col, granularity)

            if not focus_options:
                st.info(f"No valid {granularity[:-2]} values are available for {granularity} chart creation.")
                continue

            if granularity == "quarterly":
                option_labels = {f"Quarter {q}": q for q in focus_options}
                selected_focus_label = st.selectbox(
                    "Which quarter should be tracked across years?",
                    options=list(option_labels.keys()),
                    key=f"focus_value_{granularity}",
                )
                focus_value = int(option_labels[selected_focus_label])
            else:
                option_labels = {calendar.month_name[m]: m for m in focus_options}
                selected_focus_label = st.selectbox(
                    "Which month should be tracked across years?",
                    options=list(option_labels.keys()),
                    key=f"focus_value_{granularity}",
                )
                focus_value = int(option_labels[selected_focus_label])

            focus_label = format_focus_label(granularity, focus_value)
            available_periods = count_available_period_occurrences(
                df_work=df_work,
                date_col=date_col,
                granularity=granularity,
                backtrack_mode=backtrack_mode,
                focus_value=focus_value,
            )

            if available_periods == 0:
                st.info(f"No historical {focus_label} periods are available in the data.")
                continue

        label_map = {
            "yearly": "How many recent years should have separate SPC charts created?",
            "quarterly": (
                f"How many periods should be created for {focus_label} across years?"
                if backtrack_mode == "same_period" and focus_label
                else "How many recent quarters should have separate SPC charts created?"
            ),
            "monthly": (
                f"How many periods should be created for {focus_label} across years?"
                if backtrack_mode == "same_period" and focus_label
                else "How many recent months should have separate SPC charts created?"
            ),
        }

        period_requests[granularity] = {
            "count": int(
                st.number_input(
                    label_map[granularity],
                    min_value=1,
                    max_value=int(available_periods),
                    value=min(1, int(available_periods)),
                    step=1,
                    key=f"period_count_{granularity}_{backtrack_mode}_{focus_value}",
                )
            ),
            "backtrack_mode": backtrack_mode,
            "focus_value": focus_value,
            "focus_label": focus_label,
        }

    return period_requests


def render_imr_periodic_charts(
    df_work: pd.DataFrame,
    measurement_col: str,
    date_col: str,
    period_requests: dict[str, dict[str, Any]],
) -> None:
    """
    Create additional I-MR charts for selected yearly / quarterly / monthly periods.

    Updated behavior:
    - supports backtracking over all periods
    - supports backtracking for the same quarter or month across years
    - each chart is rendered inside its own Streamlit tab
    - the chart, limit summary, and violation dataframes stay together in the same tab
    """
    if not period_requests:
        return

    tab_payloads: list[dict[str, Any]] = []

    for granularity, config in period_requests.items():
        payloads = get_imr_period_chart_payloads(
            df_work=df_work,
            measurement_col=measurement_col,
            date_col=date_col,
            granularity=granularity,
            requested_count=int(config["count"]),
            backtrack_mode=str(config["backtrack_mode"]),
            focus_value=config.get("focus_value"),
        )
        tab_payloads.extend(payloads)

    if not tab_payloads:
        st.info("No additional yearly, quarterly, or monthly I‑MR charts are available.")
        return

    st.markdown("## Additional I‑MR Charts")

    tab_labels = [
        f"{payload['granularity'].capitalize()} · {payload['period_label']}"
        for payload in tab_payloads
    ]
    tabs = st.tabs(tab_labels)

    for tab, payload in zip(tabs, tab_payloads):
        with tab:
            if payload["status"] == "skipped":
                st.warning(payload["message"])
                continue

            st.plotly_chart(payload["fig"], use_container_width=True)
            render_limit_summary(payload["limits"])
            render_violations_section(
                payload["primary_violations"],
                payload["secondary_violations"],
            )


# ============================================================
# UI Rendering
# ============================================================
def render_header() -> None:
    """Render the app header."""
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)


def render_sidebar_file_upload() -> Any:
    """Render file upload control in the sidebar."""
    st.sidebar.header("Setup")
    return st.sidebar.file_uploader(
        "1) Upload a CSV or Excel file",
        type=SUPPORTED_UPLOAD_TYPES,
    )


def render_data_preview(df: pd.DataFrame) -> None:
    """Render dataset preview and summary."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    with st.expander("Preview (first 10 rows)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)


def render_column_mapping(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Render column mapping controls in the sidebar."""
    st.sidebar.subheader("Map your columns")
    all_cols = list(df.columns)

    measurement_col = st.sidebar.selectbox(
        "Measurement (required, numeric)",
        options=["—"] + all_cols,
        index=0,
    )
    date_col = st.sidebar.selectbox(
        "Date (optional)",
        options=["(none)"] + all_cols,
        index=0,
    )
    subgroup_col = st.sidebar.selectbox(
        "Subgroup (optional, categorical)",
        options=["(none)"] + all_cols,
        index=0,
    )

    m_col = None if measurement_col == "—" else measurement_col
    d_col = None if date_col == "(none)" else date_col
    g_col = None if subgroup_col == "(none)" else subgroup_col

    return m_col, d_col, g_col


def render_null_treatment_option() -> str:
    """Render null/empty measurement treatment options in the sidebar."""
    st.sidebar.subheader("Null / Empty observation handling")
    selected_label = st.sidebar.radio(
        "How should empty/null measurement observations be treated?",
        options=list(NULL_TREATMENT_OPTIONS.keys()),
        index=0,
    )
    return NULL_TREATMENT_OPTIONS[selected_label]


def render_validity_messages(evaluation: ChartEvaluation) -> None:
    """Render chart validity assessment results."""
    st.subheader("Valid SPC chart types for this data")
    for message in evaluation.messages:
        st.markdown(message)


def render_limit_summary(limits: dict[str, dict[str, Any]]) -> None:
    """Render a compact summary of the current chart limits."""
    primary = limits["primary"]
    secondary = limits["secondary"]

    st.markdown("### Limit Summary")

    p1, p2, p3 = st.columns(3)
    p1.metric(f"{primary['label']} CL", format_metric_value(primary["CL"]))
    p2.metric(f"{primary['label']} UCL", format_metric_value(primary["UCL"]))
    p3.metric(f"{primary['label']} LCL", format_metric_value(primary["LCL"]))

    if secondary:
        s1, s2, s3 = st.columns(3)
        s1.metric(f"{secondary['label']} CL", format_metric_value(secondary["CL"]))
        s2.metric(f"{secondary['label']} UCL", format_metric_value(secondary["UCL"]))
        s3.metric(f"{secondary['label']} LCL", format_metric_value(secondary["LCL"]))


def render_violations_section(
    primary_violations: pd.DataFrame,
    secondary_violations: pd.DataFrame,
) -> None:
    """Render violation tables."""
    st.markdown("### Violations")

    tab1, tab2 = st.tabs(["Primary Chart Rules", "Secondary Chart Limit Breaches"])

    with tab1:
        if primary_violations.empty:
            st.success("No primary-chart SPC rule violations detected.")
        else:
            st.dataframe(primary_violations, use_container_width=True)

    with tab2:
        if secondary_violations.empty:
            st.success("No secondary-chart limit breaches detected.")
        else:
            st.dataframe(secondary_violations, use_container_width=True)


def render_imr_main_date_selector(df_work: pd.DataFrame, date_col: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Render the date selector for the main I-MR chart.

    Updated per request:
    - defaults to a year's worth of data
    - still allows manual date selection
    """
    min_dt, max_dt = get_valid_date_bounds(df_work, date_col)
    default_start, default_end = get_default_imr_date_window(df_work, date_col)

    if min_dt is None or max_dt is None or default_start is None or default_end is None:
        st.info("No valid dates are available for date-based I‑MR chart filtering.")
        return None

    st.markdown("### Main I‑MR chart date range")

    selected_range = st.date_input(
        "Select the date range for the main I‑MR chart",
        value=(default_start.date(), default_end.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
        key="imr_main_date_range",
    )

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    elif isinstance(selected_range, list) and len(selected_range) == 2:
        start_date, end_date = selected_range[0], selected_range[1]
    else:
        st.warning("Please select both a start date and an end date.")
        return None

    start_ts, end_ts = date_range_to_full_day_bounds(start_date, end_date)

    if start_ts > end_ts:
        st.warning("The selected start date must be earlier than or equal to the end date.")
        return None

    # Notify user if date aggregation occurred
    if df_work.attrs.get("dates_aggregated", False):
        agg_count = df_work.attrs.get("aggregated_date_count", 0)
        st.sidebar.info(
            f"ℹ️ Duplicate dates were detected in the selected range.\n\n"
            f"Measurements for the same date were automatically **aggregated (summed)**.\n\n"
            f"Affected dates: {agg_count}"
        )

    return start_ts, end_ts


# ============================================================
# Main Chart Execution
# ============================================================
def run_imr_flow(df_work: pd.DataFrame, measurement_col: str, date_col: str | None) -> None:
    """Execute I-MR build, limits, violations, and rendering."""
    df_for_main_chart = df_work.copy()

    if date_col:
        selected_bounds = render_imr_main_date_selector(df_work=df_work, date_col=date_col)
        if selected_bounds is None:
            st.stop()

        start_ts, end_ts = selected_bounds
        df_for_main_chart = filter_df_by_date_range(
            df_work=df_work,
            date_col=date_col,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        if df_for_main_chart.empty:
            st.warning("No observations fall within the selected date range for the main I‑MR chart.")
            st.stop()

        if len(df_for_main_chart) < 3:
            st.warning("Fewer than 3 valid observations remain in the selected date range for the main I‑MR chart.")
            st.stop()

    chart_df = build_imr_chart_df(df_for_main_chart, measurement_col=measurement_col, date_col=date_col)
    limits = calc_limits_imr(chart_df)
    x_axis_mode = "Time" if date_col else "Index"

    primary_violations, secondary_violations = detect_violations_for_chart(chart_df, limits)

    fig = plot_spc_chart(
        chart_df=chart_df,
        limits=limits,
        title="I-MR Chart",
        primary_violations=primary_violations,
        secondary_violations=secondary_violations,
        x_axis_mode=x_axis_mode,
    )

    st.plotly_chart(fig, use_container_width=True)
    render_limit_summary(limits)
    render_violations_section(primary_violations, secondary_violations)

    if date_col:
        period_requests = render_imr_periodic_options(df_work=df_work, date_col=date_col)
        render_imr_periodic_charts(
            df_work=df_work,
            measurement_col=measurement_col,
            date_col=date_col,
            period_requests=period_requests,
        )


def run_xbar_r_flow(df_work: pd.DataFrame, measurement_col: str, subgroup_col: str) -> None:
    """Execute Xbar-R build, limits, violations, and rendering."""
    unsupported_sizes = check_unsupported_group_sizes(df_work, measurement_col, subgroup_col)
    if unsupported_sizes:
        st.error(
            f"Some subgroup sizes are unsupported for constants (n in {unsupported_sizes}). "
            f"Supported n is 2–25."
        )
        st.stop()

    chart_df = build_xbar_r_chart_df(df_work, measurement_col=measurement_col, subgroup_col=subgroup_col)
    limits = calc_limits_xbar_r(chart_df)
    primary_violations, secondary_violations = detect_violations_for_chart(chart_df, limits)

    fig = plot_spc_chart(
        chart_df=chart_df,
        limits=limits,
        title="Xbar–R Chart",
        primary_violations=primary_violations,
        secondary_violations=secondary_violations,
        x_axis_mode="Subgroup",
    )

    st.plotly_chart(fig, use_container_width=True)
    render_limit_summary(limits)
    render_violations_section(primary_violations, secondary_violations)


def run_xbar_s_flow(df_work: pd.DataFrame, measurement_col: str, subgroup_col: str) -> None:
    """Execute Xbar-S build, limits, violations, and rendering."""
    unsupported_sizes = check_unsupported_group_sizes(df_work, measurement_col, subgroup_col)
    if unsupported_sizes:
        st.error(
            f"Some subgroup sizes are unsupported for constants (n in {unsupported_sizes}). "
            f"Supported n is 2–25."
        )
        st.stop()

    chart_df = build_xbar_s_chart_df(df_work, measurement_col=measurement_col, subgroup_col=subgroup_col)
    limits = calc_limits_xbar_s(chart_df)
    primary_violations, secondary_violations = detect_violations_for_chart(chart_df, limits)

    fig = plot_spc_chart(
        chart_df=chart_df,
        limits=limits,
        title="Xbar–S Chart",
        primary_violations=primary_violations,
        secondary_violations=secondary_violations,
        x_axis_mode="Subgroup",
    )

    st.plotly_chart(fig, use_container_width=True)
    render_limit_summary(limits)
    render_violations_section(primary_violations, secondary_violations)


# ============================================================
# Main App
# ============================================================
def main() -> None:
    """Run the Streamlit SPC application."""
    render_header()

    uploaded_file = render_sidebar_file_upload()
    df = load_uploaded_file(uploaded_file)

    if df is None:
        st.info("Upload a CSV or Excel file to get started.")
        return

    st.success(f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns.")
    render_data_preview(df)

    measurement_col, date_col, subgroup_col = render_column_mapping(df)
    null_treatment = render_null_treatment_option()

    if measurement_col is None:
        st.info("Please select a Measurement column to continue.")
        st.stop()

    df_work = clean_working_data(
        df=df,
        measurement_col=measurement_col,
        date_col=date_col,
        subgroup_col=subgroup_col,
        null_treatment=null_treatment,
    )

    if df_work.empty:
        st.warning("No valid numeric measurement data remains after cleaning.")
        st.stop()

    evaluation = evaluate_chart_validity(
        df_work=df_work,
        measurement_col=measurement_col,
        subgroup_col=subgroup_col,
    )
    render_validity_messages(evaluation)

    if not evaluation.valid_options:
        st.warning("No valid SPC chart options based on the current selection. Adjust your column mapping or data.")
        st.stop()

    st.subheader("Choose a chart to graph")
    chosen_chart = st.radio(
        "Select one of the valid options:",
        options=evaluation.valid_options,
        index=0,
        horizontal=True,
    )

    st.subheader("Chart(s) with SPC Rules & Indicators")

    if chosen_chart == "I-MR":
        run_imr_flow(df_work, measurement_col, date_col)

    elif chosen_chart == "Xbar-R":
        if subgroup_col is None:
            st.error("Subgroup column is required for Xbar-R.")
            st.stop()
        run_xbar_r_flow(df_work, measurement_col, subgroup_col)

    elif chosen_chart == "Xbar-S":
        if subgroup_col is None:
            st.error("Subgroup column is required for Xbar-S.")
            st.stop()
        run_xbar_s_flow(df_work, measurement_col, subgroup_col)


if __name__ == "__main__":
    main()
