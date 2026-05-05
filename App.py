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
from numpy import linspace, exp, sqrt, pi

from datetime import date, timedelta

# Easter calculation
from dateutil.easter import easter, EASTER_WESTERN

# Public holidays (optional dependency)
try:
    import holidays as pyholidays
except ImportError:
    pyholidays = None

# ============================================================
# Streamlit App Configuration
# ============================================================
APP_TITLE = "SPC App"
APP_SUBTITLE = (
    "Use this app to upload your data, choose the relevant columns, and create SPC charts "
    "to monitor process behaviour over time. The app highlights possible special-cause "
    "signals and helps you review control limits and rule-break summaries."
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
    "All Rule Breaks": {"color": "#FF0000", "label": "All Rule Breaks"},
}
DEFAULT_RULE_STYLE = {"color": "#333333", "label": "Special cause"}
DATE_FMT_D3 = "%d %b %Y"   # 29 Apr 2026  (3-letter month)
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
    "All Rule Breaks": "All points where one or more SPC rule breaks occur.",
}

def get_rule_description_dynamic(
    rule_name: str,
    rule_points: dict[str, int],
    rule_window_threshold: dict[str, dict[str, int]],
) -> str:
    """
    Return a rule description string that reflects the current editable settings.
    Used for UI display in the rules expander (and optionally elsewhere).
    """
    # Fallback to the original text for rules we don't parameterize here
    base = RULE_DISPLAY_TEXT.get(rule_name, rule_name)

    # Single-parameter rules: "points used" changes the run/window length
    if rule_name == "Rule 2":
        n = int(rule_points.get("Rule 2", 9))
        return f"{n} (or more) points in a row are on the same side of the mean."

    if rule_name == "Rule 3":
        n = int(rule_points.get("Rule 3", 6))
        return f"{n} (or more) points in a row are continually increasing (or decreasing)."

    if rule_name == "Rule 4":
        n = int(rule_points.get("Rule 4", 14))
        return f"{n} (or more) points in a row alternate in direction, increasing then decreasing."

    if rule_name == "Rule 7":
        n = int(rule_points.get("Rule 7", 15))
        return (
            f"{n} points in a row are all within 1 standard deviation of the mean "
            f"on either side of the mean."
        )

    if rule_name == "Rule 8":
        n = int(rule_points.get("Rule 8", 8))
        return (
            f"{n} points in a row exist, but none are within 1 standard deviation of the mean, "
            f"and the points are in both directions from the mean."
        )

    # Two-parameter rules: window + threshold
    if rule_name == "Rule 5":
        cfg = rule_window_threshold.get("Rule 5", {"window": 3, "threshold": 2})
        w = max(2, int(cfg.get("window", 3)))
        t = min(max(1, int(cfg.get("threshold", 2))), w)
        return (
            f"{t} (or more) out of {w} points in a row are more than 2 standard deviations "
            f"from the mean in the same direction."
        )

    if rule_name == "Rule 6":
        cfg = rule_window_threshold.get("Rule 6", {"window": 5, "threshold": 4})
        w = max(2, int(cfg.get("window", 5)))
        t = min(max(1, int(cfg.get("threshold", 4))), w)
        return (
            f"{t} (or more) out of {w} points in a row are more than 1 standard deviation "
            f"from the mean in the same direction."
        )

    return base


# ---- Editable rule parameters (defaults)
DEFAULT_RULE_ENABLED = {
    "Rule 1": True,
    "Rule 2": True,
    "Rule 3": True,
    "Rule 4": False,
    "Rule 5": False,
    "Rule 6": False,
    "Rule 7": False,
    "Rule 8": False,
    "Secondary chart: point beyond control limit": True,
}

# Rules whose "points used" are a single number (run length/window size)
DEFAULT_RULE_POINTS = {
    "Rule 2": 7,
    "Rule 3": 7,
    "Rule 4": 14,
    "Rule 7": 15,
    "Rule 8": 8,
}

# Rules with BOTH a window and a threshold (count within that window)
DEFAULT_RULE_WINDOW_THRESHOLD = {
    "Rule 5": {"window": 3, "threshold": 2},
    "Rule 6": {"window": 5, "threshold": 4},
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
    "All Rule Breaks": 10,
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

STRUCTURAL_BREAK_DEFAULTS = {
    "min_history": 25,              # baseline length used to estimate segment mean/std
    "mean_allowance": 0.50,         # k for standardized mean CUSUM
    "mean_decision_interval": 5.0,  # h for mean CUSUM
    "var_allowance": 0.25,          # k for variance CUSUM on (z^2 - 1)
    "var_decision_interval": 6.0,   # h for variance CUSUM
    "confirmations": 2,             # number of consecutive alarming points to confirm a break
    "min_segment_length": 15,       # minimum number of observations allowed in any segment
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
    return pd.DataFrame(columns=["date", "rule", "value", "rule_description", "count_as_break"])

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
def build_holiday_calendar(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    country_code: str = "ZA",
    subdiv: str | None = None,
    observed: bool = True,
    christmas_start_md: tuple[int, int] = (12, 24),
    christmas_end_md: tuple[int, int] = (1, 6),
    easter_week_mode: str = "iso_week",  # "iso_week" or "goodfri_to_mon"
) -> pd.DataFrame:
    """
    Build a calendar of dates between start_dt and end_dt with holiday flags.
    Returns a DataFrame with:
      - date
      - is_public_holiday, public_holiday_name
      - is_christmas_holiday
      - is_easter_week
      - holiday_type (public/christmas/easter_week/none)
    """

    start_dt = pd.Timestamp(start_dt).normalize()
    end_dt = pd.Timestamp(end_dt).normalize()

    all_days = pd.date_range(start_dt, end_dt, freq="D")
    cal = pd.DataFrame({"date": all_days})

    # -------- Public holidays
    pub_names = {}
    if pyholidays is not None:
        years = range(start_dt.year, end_dt.year + 1)
        hdays = pyholidays.country_holidays(country_code, years=years, subdiv=subdiv, observed=observed)
        # dict-like: keys are date objects; values are holiday names
        pub_names = {pd.Timestamp(d): name for d, name in hdays.items()}
    else:
        # If library missing, leave empty (but app can warn)
        pub_names = {}

    cal["public_holiday_name"] = cal["date"].map(pub_names)
    cal["is_public_holiday"] = cal["public_holiday_name"].notna()

    # -------- Christmas holiday window (spans year boundary)
    # Example default: Dec 24 -> Jan 6
    christmas_dates = set()
    for y in range(start_dt.year - 1, end_dt.year + 1):
        start = pd.Timestamp(date(y, christmas_start_md[0], christmas_start_md[1]))
        # end is in following year if month/day is Jan...
        end_year = y if christmas_end_md[0] >= christmas_start_md[0] else y + 1
        end = pd.Timestamp(date(end_year, christmas_end_md[0], christmas_end_md[1]))
        for d in pd.date_range(start, end, freq="D"):
            christmas_dates.add(pd.Timestamp(d).normalize())

    cal["is_christmas_holiday"] = cal["date"].isin(christmas_dates)

    # -------- Easter week
    easter_week_dates = set()
    for y in range(start_dt.year, end_dt.year + 1):
        easter_sun = pd.Timestamp(easter(y, method=EASTER_WESTERN))

        if easter_week_mode == "goodfri_to_mon":
            # Good Friday (Sun-2) through Easter Monday (Sun+1)
            start = easter_sun - pd.Timedelta(days=2)
            end = easter_sun + pd.Timedelta(days=1)
        else:
            # ISO week containing Easter Sunday (Mon..Sun)
            start = easter_sun - pd.Timedelta(days=int(easter_sun.weekday()))  # Monday
            end = start + pd.Timedelta(days=6)

        for d in pd.date_range(start.normalize(), end.normalize(), freq="D"):
            easter_week_dates.add(pd.Timestamp(d).normalize())

    cal["is_easter_week"] = cal["date"].isin(easter_week_dates)

    # -------- A simple categorical label (useful for saving/export)
    def _label_row(r):
        if r["is_public_holiday"]:
            return "public_holiday"
        if r["is_christmas_holiday"]:
            return "christmas_holiday"
        if r["is_easter_week"]:
            return "easter_week"
        return "none"

    cal["holiday_type"] = cal.apply(_label_row, axis=1)
    return cal


def save_holiday_calendar_to_session(holiday_df: pd.DataFrame) -> None:
    """Store holiday calendar in session_state."""
    st.session_state["holiday_calendar_df"] = holiday_df.copy()


@st.cache_data(show_spinner=False)
def fill_missing_dates_with_zero(
    df: pd.DataFrame,
    date_col: str,
    measurement_col: str,
    freq: str = "D",
    agg: str = "sum",
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Ensures every date in [min(date), max(date)] exists.
    Missing dates get measurement=0.
    Returns (new_df, missing_dates_index).

    If multiple rows per date exist, aggregates them first (sum/mean).
    """
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work[measurement_col] = pd.to_numeric(work[measurement_col], errors="coerce")

    work = work.dropna(subset=[date_col])

    if agg == "mean":
        daily = work.groupby(date_col, as_index=False)[measurement_col].mean()
    else:
        daily = work.groupby(date_col, as_index=False)[measurement_col].sum()

    if daily.empty:
        return daily, pd.DatetimeIndex([])

    full = pd.date_range(daily[date_col].min(), daily[date_col].max(), freq=freq)
    daily = daily.set_index(date_col).reindex(full)

    missing = daily[daily[measurement_col].isna()].index
    daily[measurement_col] = daily[measurement_col].fillna(0.0)

    daily = daily.rename_axis(date_col).reset_index()
    daily["imputed_zero_missing_date"] = daily[date_col].isin(missing)
    return daily, missing


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

    #default_start = max(min_dt, max_dt - pd.DateOffset(years=1) + pd.Timedelta(days=1))
    return pd.Timestamp(min_dt), pd.Timestamp(max_dt)


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
            sheet_name = st.selectbox("Choose a sheet", options=sheet_names)
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
    count_as_break: bool = True,
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
                "count_as_break": bool(count_as_break),
            }
        )


def mark_run_same_side(values: np.ndarray, center_line: np.ndarray, min_run_len: int) -> set[int]:
    """Mark consecutive points on the same side of the center line."""
    values = np.asarray(values, dtype=float)
    center_line = np.asarray(center_line, dtype=float)

    side = np.where(values > center_line, 1, np.where(values <= center_line, -1, 0))
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
def apply_uniform_date_format(fig, date_fmt_d3: str = DATE_FMT_D3) -> None:
    """
    Enforce consistent date display across all x-axes and hover tooltips in a Plotly figure.
    Works for subplots too (applies to all x-axes).
    """
    # Axis tick labels
    fig.update_xaxes(tickformat=date_fmt_d3)

    # Hover date formatting (axis-level); traces can override if they have custom hovertemplate
    fig.update_xaxes(hoverformat=date_fmt_d3)

    # If any traces define their own hovertemplate, enforce date format there too
    for tr in fig.data:
        # Only apply to traces that actually use x (time-series)
        if hasattr(tr, "hovertemplate") and tr.hovertemplate:
            # Replace any existing x formatting with our format if present,
            # otherwise prepend an x line with our format.
            if "%{x|" in tr.hovertemplate:
                # crude but effective: normalize any x date formatting to our format
                import re
                tr.hovertemplate = re.sub(r"%\{x\|[^}]+\}", f"%{{x|{date_fmt_d3}}}", tr.hovertemplate)
            else:
                tr.hovertemplate = f"%{{x|{date_fmt_d3}}}<br>" + tr.hovertemplate
        elif hasattr(tr, "hoverinfo") and tr.hoverinfo:
            # leave hoverinfo-only traces as-is (not all traces support hovertemplate cleanly)
            pass


@st.cache_data(show_spinner=False)
def detect_spc_rule_violations(
    df: pd.DataFrame,
    y_col: str,
    cl: float | np.ndarray,
    sigma: float | np.ndarray,
    enabled_rules: dict[str, bool],
    rule_points: dict[str, int],
    rule_window_threshold: dict[str, dict[str, int]],
    mark_all_sequence_points: bool = True,
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

    # --- Helpers to split indices into contributory vs break points ---

    def _split_contiguous_runs_threshold(idxs, threshold: int) -> tuple[set[int], set[int]]:
        """
        For each contiguous run in idxs:
          - first (threshold-1) points are contributory
          - from threshold-th point onward are breaks
        Returns: (breaks, contributory)
        """
        s = sorted({int(i) for i in idxs})
        breaks: set[int] = set()
        contrib: set[int] = set()

        if not s:
            return breaks, contrib
        threshold = max(2, int(threshold))  # safety

        run_start = prev = s[0]
        for cur in list(s[1:]) + [None]:  # sentinel to flush last run
            if cur is not None and cur == prev + 1:
                prev = cur
                continue

            run_len = prev - run_start + 1
            if run_len >= threshold:
                contrib.update(range(run_start, run_start + threshold - 1))
                breaks.update(range(run_start + threshold - 1, prev + 1))

            if cur is None:
                break
            run_start = prev = cur

        return breaks, contrib

    def _split_window_end_runs_with_leading_contrib(end_idxs, window: int) -> tuple[set[int], set[int]]:
        """
        For rules defined by a moving window (Rule 8):
          - breaks are the window END indices that satisfy the rule
          - contributory are the first (window-1) points leading into the first break
            of each contiguous streak of qualifying window ends.
        Returns: (breaks, contributory)
        """
        ends = sorted({int(i) for i in end_idxs})
        breaks: set[int] = set()
        contrib: set[int] = set()
        if not ends:
            return breaks, contrib

        window = max(2, int(window))  # safety
        run_start = prev = ends[0]

        for cur in list(ends[1:]) + [None]:
            if cur is not None and cur == prev + 1:
                prev = cur
                continue

            # Flush run of qualifying window-ends: [run_start .. prev]
            breaks.update(range(run_start, prev + 1))

            # Leading contributory points before the first break in this run
            lead_start = max(0, run_start - (window - 1))
            contrib.update(range(lead_start, run_start))

            if cur is None:
                break
            run_start = prev = cur

        # Make sure contrib doesn't overlap breaks
        contrib -= breaks
        return breaks, contrib

    def _last_of_contiguous_runs(idxs) -> set[int]:
        s = sorted({int(i) for i in idxs})
        if not s:
            return set()

        last = set()
        prev = s[0]

        for cur in s[1:]:
            if cur == prev + 1:
                prev = cur
            else:
                last.add(prev)   # end of the previous contiguous run
                prev = cur

        last.add(prev)           # end of the final run
        return last

    # Rule 1
    if enabled_rules.get("Rule 1", True):
        rule1_idx = np.where((values > upper_3) | (values < lower_3))[0]
        append_rule_hits(violations, dates, values, rule1_idx, "Rule 1")

    # Rule 2
    if enabled_rules.get("Rule 2", True):
        r2 = int(rule_points.get("Rule 2", 9))
        rule2_idx = mark_run_same_side(values, cl_arr, min_run_len=r2)

        if mark_all_sequence_points:
            # Current behavior: all points in qualifying runs are breaks
            append_rule_hits(violations, dates, values, rule2_idx, "Rule 2", count_as_break=True)
        else:
            # New behavior: first (r2-1) contributory, from r2th onward are breaks
            rule2_breaks, rule2_contrib = _split_contiguous_runs_threshold(rule2_idx, r2)
            append_rule_hits(violations, dates, values, rule2_contrib, "Rule 2", count_as_break=False)
            append_rule_hits(violations, dates, values, rule2_breaks, "Rule 2", count_as_break=True)
    
    #print(rule2_contrib)

    # Rule 3
    if enabled_rules.get("Rule 3", True):
        r3 = int(rule_points.get("Rule 3", 6))
        rule3_idx = mark_monotonic_runs(values, min_points=r3)

        if mark_all_sequence_points:
            append_rule_hits(violations, dates, values, rule3_idx, "Rule 3", count_as_break=True)
        else:
            rule3_breaks, rule3_contrib = _split_contiguous_runs_threshold(rule3_idx, r3)
            append_rule_hits(violations, dates, values, rule3_contrib, "Rule 3", count_as_break=False)
            append_rule_hits(violations, dates, values, rule3_breaks, "Rule 3", count_as_break=True)

    # Rule 4
    if enabled_rules.get("Rule 4", True):
        r4 = int(rule_points.get("Rule 4", 14))
        rule4_idx = mark_alternating_runs(values, min_points=r4)

        if mark_all_sequence_points:
            append_rule_hits(violations, dates, values, rule4_idx, "Rule 4", count_as_break=True)
        else:
            rule4_breaks, rule4_contrib = _split_contiguous_runs_threshold(rule4_idx, r4)
            append_rule_hits(violations, dates, values, rule4_contrib, "Rule 4", count_as_break=False)
            append_rule_hits(violations, dates, values, rule4_breaks, "Rule 4", count_as_break=True)

    # Rule 5 (threshold + window)
    if enabled_rules.get("Rule 5", True):
        cfg5 = rule_window_threshold.get("Rule 5", {"window": 3, "threshold": 2})
        w5 = int(cfg5.get("window", 3))
        t5 = int(cfg5.get("threshold", 2))

        # Safety clamps
        w5 = max(2, w5)
        t5 = min(max(1, t5), w5)

        rule5_idx: set[int] = set()
        rule5_triggers_raw: set[int] = set()

        for i in range(w5 - 1, n):
            window = values[i - (w5 - 1): i + 1]
            if np.any(pd.isna(window)):
                continue

            above = [j for j in range(w5) if window[j] > upper_2[i - (w5 - 1) + j]]
            below = [j for j in range(w5) if window[j] < lower_2[i - (w5 - 1) + j]]

            if len(above) >= t5:
                rule5_idx.update((i - (w5 - 1)) + np.array(above))
                rule5_triggers_raw.add(i)
            if len(below) >= t5:
                rule5_idx.update((i - (w5 - 1)) + np.array(below))
                rule5_triggers_raw.add(i)
        
        if mark_all_sequence_points:
            append_rule_hits(violations, dates, values, rule7_idx, "Rule 7", count_as_break=True)
        else:
            rule7_breaks, rule7_contrib = _split_contiguous_runs_threshold(rule7_idx, r7)
            append_rule_hits(violations, dates, values, rule7_contrib, "Rule 7", count_as_break=False)
            append_rule_hits(violations, dates, values, rule7_breaks, "Rule 7", count_as_break=True)


    # Rule 6 (threshold + window)
    if enabled_rules.get("Rule 6", True):
        cfg6 = rule_window_threshold.get("Rule 6", {"window": 5, "threshold": 4})
        w6 = int(cfg6.get("window", 5))
        t6 = int(cfg6.get("threshold", 4))

        # Safety clamps
        w6 = max(2, w6)
        t6 = min(max(1, t6), w6)

        rule6_idx: set[int] = set()
        rule6_triggers_raw: set[int] = set()
        for i in range(w6 - 1, n):
            window = values[i - (w6 - 1): i + 1]
            if np.any(pd.isna(window)):
                continue

            above = [j for j in range(w6) if window[j] > upper_1[i - (w6 - 1) + j]]
            below = [j for j in range(w6) if window[j] < lower_1[i - (w6 - 1) + j]]

            if len(above) >= t6:
                rule6_idx.update((i - (w6 - 1)) + np.array(above))
                rule6_triggers_raw.add(i)
            if len(below) >= t6:
                rule6_idx.update((i - (w6 - 1)) + np.array(below))
                rule6_triggers_raw.add(i)

        if mark_all_sequence_points:
            append_rule_hits(violations, dates, values, rule6_idx, "Rule 6")
        else:
            trigger = _last_of_contiguous_runs(rule6_triggers_raw)
            contrib = (rule6_idx | rule6_triggers_raw) - trigger
            append_rule_hits(violations, dates, values, contrib, "Rule 6", count_as_break=False)
            append_rule_hits(violations, dates, values, trigger, "Rule 6", count_as_break=True)

    # Rule 7
    if enabled_rules.get("Rule 7", True):
        r7 = int(rule_points.get("Rule 7", 15))
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

            if (j - i + 1) >= r7:
                rule7_idx.update(range(i, j + 1))

            i = j + 1

        if mark_all_sequence_points:
            append_rule_hits(violations, dates, values, rule7_idx, "Rule 7")
        else:
            trigger = _last_of_contiguous_runs(rule7_idx)
            contrib = set(rule7_idx) - trigger
            append_rule_hits(violations, dates, values, contrib, "Rule 7", count_as_break=False)
            append_rule_hits(violations, dates, values, trigger, "Rule 7", count_as_break=True)

    # Rule 8
    if enabled_rules.get("Rule 8", True):
        w8 = int(rule_points.get("Rule 8", 8))
        w8 = max(2, w8)

        # If ticked, preserve current behavior (mark all points in each qualifying window)
        rule8_idx: set[int] = set()

        # Always track qualifying window-end indices (these are the "break events" when unticked)
        rule8_end_idx: set[int] = set()

        for i in range(w8 - 1, n):
            window = values[i - (w8 - 1): i + 1]
            if np.any(pd.isna(window)):
                continue

            outside_1 = np.array(
                [abs(window[j] - cl_arr[i - (w8 - 1) + j]) > sigma_arr[i - (w8 - 1) + j] for j in range(w8)]
            )
            sides = np.array(
                [
                    1 if window[j] > cl_arr[i - (w8 - 1) + j]
                    else (-1 if window[j] < cl_arr[i - (w8 - 1) + j] else 0)
                    for j in range(w8)
                ]
            )

            if np.all(outside_1) and (np.any(sides == 1) and np.any(sides == -1)):
                rule8_end_idx.add(i)
                if mark_all_sequence_points:
                    rule8_idx.update(range(i - (w8 - 1), i + 1))

        if mark_all_sequence_points:
            append_rule_hits(violations, dates, values, rule8_idx, "Rule 8", count_as_break=True)
        else:
            rule8_breaks, rule8_contrib = _split_window_end_runs_with_leading_contrib(rule8_end_idx, w8)
            append_rule_hits(violations, dates, values, rule8_contrib, "Rule 8", count_as_break=False)
            append_rule_hits(violations, dates, values, rule8_breaks, "Rule 8", count_as_break=True)

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
    enabled: bool,
) -> pd.DataFrame:
    """
    Detect secondary chart points beyond control limits.
    Core logic preserved from the original implementation.
    """
    if not enabled:
        return empty_violations_df()
    
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
                    "count_as_break": True,
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
# Sequential Structural Break Detection (Joint / Adaptive CUSUM)
# ============================================================
def _estimate_segment_baseline(
    values: np.ndarray,
    start: int,
    baseline_end: int,
) -> tuple[float, float] | None:
    """
    Estimate the in-control baseline (mean, std) for the current segment using
    the first part of that segment only.

    Returns:
        (mu0, sigma0) if enough valid data exist, else None
    """
    baseline = values[start:baseline_end]
    baseline = baseline[~np.isnan(baseline)]

    if len(baseline) < 2:
        return None

    mu0 = float(np.mean(baseline))
    sigma0 = float(np.std(baseline, ddof=1))

    if not np.isfinite(sigma0) or sigma0 <= 1e-8:
        sigma0 = 1e-8

    return mu0, sigma0


def detect_structural_breaks_sequential(
    values: pd.Series | np.ndarray,
    min_history: int = STRUCTURAL_BREAK_DEFAULTS["min_history"],
    mean_allowance: float = STRUCTURAL_BREAK_DEFAULTS["mean_allowance"],
    mean_decision_interval: float = STRUCTURAL_BREAK_DEFAULTS["mean_decision_interval"],
    var_allowance: float = STRUCTURAL_BREAK_DEFAULTS["var_allowance"],
    var_decision_interval: float = STRUCTURAL_BREAK_DEFAULTS["var_decision_interval"],
    confirmations: int = STRUCTURAL_BREAK_DEFAULTS["confirmations"],
    min_segment_length: int = STRUCTURAL_BREAK_DEFAULTS["min_segment_length"],
) -> list[int]:
    """
    Detect structural breaks sequentially using a joint/adaptive CUSUM-type method.

    Method:
    - The first `min_history` observations in each segment estimate the baseline mean/std.
    - Thereafter, four tabular CUSUMs are updated sequentially:
        1) mean shift upward
        2) mean shift downward
        3) variance shift upward
        4) variance shift downward
    - A break is confirmed only after `confirmations` consecutive alarming points.
    - After a break, a fresh segment begins and a new baseline is estimated.

    Notes:
    - Uses only past and present values (no future leakage).
    - Returns 0-based indices marking the first point of each new segment.
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    n = len(arr)

    if n < max(min_history + 1, min_segment_length + 1):
        return []

    breaks: list[int] = []
    segment_start = 0

    while segment_start + min_history < n:
        baseline_end = segment_start + min_history
        baseline = _estimate_segment_baseline(arr, segment_start, baseline_end)

        if baseline is None:
            break

        mu0, sigma0 = baseline

        # Two-sided mean CUSUMs
        c_mean_pos = 0.0
        c_mean_neg = 0.0

        # Two-sided variance CUSUMs
        c_var_pos = 0.0
        c_var_neg = 0.0

        # Consecutive alarm tracking for confirmation
        alarm_streak = 0
        first_alarm_idx: int | None = None

        break_confirmed = False

        for t in range(baseline_end, n):
            x_t = arr[t]

            if pd.isna(x_t):
                alarm_streak = 0
                first_alarm_idx = None
                continue

            # Standardized residual relative to the current segment baseline
            z_t = (x_t - mu0) / sigma0

            # -------------------------
            # Mean CUSUM (two-sided)
            # -------------------------
            c_mean_pos = max(0.0, c_mean_pos + z_t - mean_allowance)
            c_mean_neg = max(0.0, c_mean_neg - z_t - mean_allowance)

            # -------------------------
            # Variance CUSUM (two-sided)
            # Monitor departures of z^2 from 1
            # -------------------------
            v_t = (z_t ** 2) - 1.0
            c_var_pos = max(0.0, c_var_pos + v_t - var_allowance)
            c_var_neg = max(0.0, c_var_neg - v_t - var_allowance)

            mean_alarm = (
                c_mean_pos >= mean_decision_interval or
                c_mean_neg >= mean_decision_interval
            )
            var_alarm = (
                c_var_pos >= var_decision_interval or
                c_var_neg >= var_decision_interval
            )

            alarm_now = mean_alarm or var_alarm

            if alarm_now:
                alarm_streak += 1
                if first_alarm_idx is None:
                    first_alarm_idx = t
            else:
                alarm_streak = 0
                first_alarm_idx = None

            if alarm_streak >= confirmations and first_alarm_idx is not None:
                candidate_break_idx = int(first_alarm_idx)

                # Enforce minimum segment length
                if (candidate_break_idx - segment_start) >= min_segment_length:
                    breaks.append(candidate_break_idx)
                    segment_start = candidate_break_idx
                    break_confirmed = True
                    break
                else:
                    # Too short to allow a break here; reset evidence and continue
                    c_mean_pos = 0.0
                    c_mean_neg = 0.0
                    c_var_pos = 0.0
                    c_var_neg = 0.0
                    alarm_streak = 0
                    first_alarm_idx = None

        if not break_confirmed:
            break

    return sorted(set(idx for idx in breaks if 0 < idx < n))


def build_segment_ranges(length: int, break_indices: list[int]) -> list[tuple[int, int]]:
    """Convert break indices into [(start, end), ...] ranges."""
    valid_breaks = sorted({int(i) for i in break_indices if 0 < int(i) < length})
    bounds = [0] + valid_breaks + [length]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def calc_segmented_limits(
    chart_df: pd.DataFrame,
    base_calc_func,
    break_indices: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Calculate segment-wise control limits using the existing chart-specific limit logic.

    This preserves the current chart formulas, but applies them independently
    within each detected segment.
    """
    break_indices = break_indices or []
    n = len(chart_df)

    if n == 0:
        return base_calc_func(chart_df)

    segment_ranges = build_segment_ranges(n, break_indices)

    if len(segment_ranges) == 1:
        base_limits = base_calc_func(chart_df)
        base_limits["break_indices"] = []
        base_limits["segment_ranges"] = segment_ranges
        base_limits["segment_summaries"] = [{
            "segment": 1,
            "start_obs": 1,
            "end_obs": n,
            "primary_CL": base_limits["primary"]["CL"],
            "primary_UCL": base_limits["primary"]["UCL"],
            "primary_LCL": base_limits["primary"]["LCL"],
            "secondary_CL": base_limits["secondary"]["CL"],
            "secondary_UCL": base_limits["secondary"]["UCL"],
            "secondary_LCL": base_limits["secondary"]["LCL"],
        }]
        return base_limits

    first_seg_limits = base_calc_func(chart_df.iloc[segment_ranges[0][0]:segment_ranges[0][1]].copy())

    segmented_limits = {
        "primary": {
            "label": first_seg_limits["primary"]["label"],
            "y_col": first_seg_limits["primary"]["y_col"],
            "CL": first_seg_limits["primary"]["CL"],
            "UCL": first_seg_limits["primary"]["UCL"],
            "LCL": first_seg_limits["primary"]["LCL"],
            "CL_series": np.full(n, np.nan, dtype=float),
            "UCL_series": np.full(n, np.nan, dtype=float),
            "LCL_series": np.full(n, np.nan, dtype=float),
            "sigma": first_seg_limits["primary"]["sigma"],
            "sigma_series": (
                np.full(n, np.nan, dtype=float)
                if first_seg_limits["primary"]["sigma_series"] is not None
                else None
            ),
        },
        "secondary": {
            "label": first_seg_limits["secondary"]["label"],
            "y_col": first_seg_limits["secondary"]["y_col"],
            "CL": first_seg_limits["secondary"]["CL"],
            "UCL": first_seg_limits["secondary"]["UCL"],
            "LCL": first_seg_limits["secondary"]["LCL"],
            "CL_series": np.full(n, np.nan, dtype=float),
            "UCL_series": np.full(n, np.nan, dtype=float),
            "LCL_series": np.full(n, np.nan, dtype=float),
            "sigma": first_seg_limits["secondary"]["sigma"],
            "sigma_series": (
                np.full(n, np.nan, dtype=float)
                if first_seg_limits["secondary"]["sigma_series"] is not None
                else None
            ),
        },
        "break_indices": sorted(set(break_indices)),
        "segment_ranges": segment_ranges,
        "segment_summaries": [],
    }

    last_seg_limits = first_seg_limits

    for seg_no, (start, end) in enumerate(segment_ranges, start=1):
        seg_df = chart_df.iloc[start:end].copy()
        seg_limits = base_calc_func(seg_df)
        seg_len = end - start

        # Primary
        segmented_limits["primary"]["CL_series"][start:end] = as_array(seg_limits["primary"]["CL_series"], seg_len)
        segmented_limits["primary"]["UCL_series"][start:end] = as_array(seg_limits["primary"]["UCL_series"], seg_len)
        segmented_limits["primary"]["LCL_series"][start:end] = as_array(seg_limits["primary"]["LCL_series"], seg_len)
        if segmented_limits["primary"]["sigma_series"] is not None and seg_limits["primary"]["sigma_series"] is not None:
            segmented_limits["primary"]["sigma_series"][start:end] = as_array(seg_limits["primary"]["sigma_series"], seg_len)

        # Secondary
        segmented_limits["secondary"]["CL_series"][start:end] = as_array(seg_limits["secondary"]["CL_series"], seg_len)
        segmented_limits["secondary"]["UCL_series"][start:end] = as_array(seg_limits["secondary"]["UCL_series"], seg_len)
        segmented_limits["secondary"]["LCL_series"][start:end] = as_array(seg_limits["secondary"]["LCL_series"], seg_len)
        if segmented_limits["secondary"]["sigma_series"] is not None and seg_limits["secondary"]["sigma_series"] is not None:
            segmented_limits["secondary"]["sigma_series"][start:end] = as_array(seg_limits["secondary"]["sigma_series"], seg_len)

        segmented_limits["segment_summaries"].append(
            {
                "segment": seg_no,
                "start_obs": start + 1,
                "end_obs": end,
                "primary_CL": seg_limits["primary"]["CL"],
                "primary_UCL": seg_limits["primary"]["UCL"],
                "primary_LCL": seg_limits["primary"]["LCL"],
                "secondary_CL": seg_limits["secondary"]["CL"],
                "secondary_UCL": seg_limits["secondary"]["UCL"],
                "secondary_LCL": seg_limits["secondary"]["LCL"],
            }
        )

        last_seg_limits = seg_limits

    # Use the most recent segment's scalar limits for summary metrics
    segmented_limits["primary"]["CL"] = last_seg_limits["primary"]["CL"]
    segmented_limits["primary"]["UCL"] = last_seg_limits["primary"]["UCL"]
    segmented_limits["primary"]["LCL"] = last_seg_limits["primary"]["LCL"]
    segmented_limits["primary"]["sigma"] = last_seg_limits["primary"]["sigma"]

    segmented_limits["secondary"]["CL"] = last_seg_limits["secondary"]["CL"]
    segmented_limits["secondary"]["UCL"] = last_seg_limits["secondary"]["UCL"]
    segmented_limits["secondary"]["LCL"] = last_seg_limits["secondary"]["LCL"]
    segmented_limits["secondary"]["sigma"] = last_seg_limits["secondary"]["sigma"]

    return segmented_limits


def get_limits_with_optional_structural_breaks(
    chart_df: pd.DataFrame,
    chart_type: str,
    enable_structural_break_detection: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Return chart_df (possibly adjusted) and control limits.
    If enabled, limits are recalculated segment-by-segment after sequential break detection.
    """
    calc_map = {
        "I-MR": calc_limits_imr,
        "Xbar-R": calc_limits_xbar_r,
        "Xbar-S": calc_limits_xbar_s,
        "P": calc_limits_p,
    }
    primary_col_map = {
        "I-MR": "value",
        "Xbar-R": "xbar",
        "Xbar-S": "xbar",
        "P": "p",
    }

    if chart_type not in calc_map:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    working_chart_df = chart_df.copy()
    break_indices: list[int] = []

    if enable_structural_break_detection:
        break_indices = detect_structural_breaks_sequential(
            working_chart_df[primary_col_map[chart_type]].to_numpy()
        )

        # For I-MR, the first MR in each new segment must not bridge across regimes
        if chart_type == "I-MR" and break_indices:
            valid_breaks = [idx for idx in break_indices if 0 <= idx < len(working_chart_df)]
            if valid_breaks:
                working_chart_df.loc[valid_breaks, "MR"] = np.nan
    
    limits = calc_segmented_limits(
        chart_df=working_chart_df,
        base_calc_func=calc_map[chart_type],
        break_indices=break_indices,
    )

    return working_chart_df, limits


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
    # If a holiday calendar is present in session_state, annotate chart_df
    holiday_df = st.session_state.get("holiday_calendar_df")
    if holiday_df is not None and not holiday_df.empty:
        chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce").dt.normalize()
        chart_df = chart_df.merge(holiday_df, on="date", how="left")
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

    xbar = round(np.mean(values),1)
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
@st.cache_data(show_spinner=False)
def build_p_chart_df(
    df: pd.DataFrame,
    inspected_col: str,
    defects_col: str,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Build chart-ready data for a P chart (proportion nonconforming)."""
    use_cols = [inspected_col, defects_col] + ([date_col] if date_col else [])
    work = df[use_cols].dropna().copy()

    # Ensure valid numeric types
    work[inspected_col] = coerce_numeric(work[inspected_col])
    work[defects_col] = coerce_numeric(work[defects_col])
    work = work.dropna(subset=[inspected_col, defects_col])

    # Validate constraints
    work = work[work[inspected_col] > 0]
    work = work[(work[defects_col] >= 0) & (work[defects_col] <= work[inspected_col])]

    if date_col:
        work[date_col] = parse_date(work[date_col])
        work = work.dropna(subset=[date_col]).sort_values(by=date_col)
        work = work.rename(columns={date_col: "date"})
    else:
        work = work.reset_index(drop=True)
        work["date"] = pd.to_datetime(np.arange(len(work)), unit="D", origin="unix")

    work = work.rename(columns={inspected_col: "n", defects_col: "defects"})
    work["p"] = work["defects"] / work["n"]
    work["subgroup_number"] = np.arange(1, len(work) + 1)

    return work


@st.cache_data(show_spinner=False)
def calc_limits_p(chart_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Calculate control limits for a P chart.

    p_i = defects_i / n_i
    pbar = sum(defects) / sum(n)
    sigma_i = sqrt(pbar*(1-pbar)/n_i)
    UCL_i = pbar + 3*sigma_i
    LCL_i = pbar - 3*sigma_i
    Clamp to [0, 1].
    """
    n = pd.to_numeric(chart_df["n"], errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(chart_df["defects"], errors="coerce").to_numpy(dtype=float)

    n_sum = np.nansum(n)
    if n_sum <= 0:
        pbar = np.nan
    else:
        pbar = np.nansum(d) / n_sum

    sigma = np.sqrt((pbar * (1.0 - pbar)) / n)
    ucl = pbar + 3.0 * sigma
    lcl = pbar - 3.0 * sigma

    ucl = np.clip(ucl, 0.0, 1.0)
    lcl = np.clip(lcl, 0.0, 1.0)

    # IMPORTANT: match existing app keys ("CL", "UCL", "LCL", "sigma")
    primary = {
        "y_col": "p",
        "label": "p",
        "CL": repeat_line(pbar, len(chart_df)),
        "UCL": ucl,
        "LCL": lcl,
        "sigma": sigma,
    }

    # Keep secondary disabled, but still provide expected keys so other code doesn't KeyError
    secondary = {
        "enabled": False,
        "y_col": None,
        "label": "",
        "CL": None,
        "UCL": None,
        "LCL": None,
        "sigma": None,
    }

    return {"primary": primary, "secondary": secondary}

def compute_kde(x: np.ndarray, grid_size: int = 200):
    """
    Compute a Gaussian KDE for plotting.
    Returns x_grid, density
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return None, None

    std = np.std(x, ddof=1)
    if std == 0:
        return None, None

    # Silverman's rule of thumb
    bandwidth = 1.06 * std * n ** (-1 / 5)

    x_grid = linspace(x.min(), x.max(), grid_size)
    density = sum(
        exp(-0.5 * ((x_grid - xi) / bandwidth) ** 2)
        for xi in x
    ) / (n * bandwidth * sqrt(2 * pi))

    return x_grid, density

def apply_plot_line_gaps(line_values: np.ndarray, break_positions: list[int] | None = None) -> np.ndarray:
    """Insert NaN at break positions so plotted lines visually split across segments."""
    arr = np.asarray(line_values, dtype=float).copy()
    if break_positions:
        for idx in break_positions:
            if 0 <= idx < len(arr):
                arr[idx] = np.nan
    return arr


def add_structural_break_lines(
    fig: go.Figure,
    break_x_values: list[Any],
    rows: list[int],
    col: int = 1,
) -> None:
    """Add vertical dashed lines marking structural breaks."""
    for x_val in break_x_values:
        for row in rows:
            fig.add_vline(
                x=x_val,
                line_width=1.5,
                line_dash="dash",
                line_color="#1E90FF",
                opacity=0.85,
                row=row,
                col=col,
            )

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
    break_positions: list[int] | None = None,
) -> None:
    """Add center line, sigma reference lines, and control limits to a subplot."""
    n = len(x_values)
    cl_arr = as_array(cl, n)
    ucl_arr = as_array(ucl, n)
    lcl_arr = as_array(lcl, n)
    sigma_arr = as_array(sigma, n)
    plot_cl_arr = apply_plot_line_gaps(cl_arr, break_positions)
    plot_ucl_arr = apply_plot_line_gaps(ucl_arr, break_positions)
    plot_lcl_arr = apply_plot_line_gaps(lcl_arr, break_positions)

    # Center Line
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=plot_cl_arr,
            mode="lines",
            name="Center Line",
            legendgroup="center_line",
            showlegend=show_legend_once,
            line=dict(color="#228B22", dash="dash"),
        ),
        row=row,
        col=col,
    )

    # Sigma Reference Lines (segment-aware)
    if not np.all(np.isnan(sigma_arr)) and np.nanmax(np.abs(sigma_arr)) != 0:
        upper_1_arr = cl_arr + sigma_arr
        lower_1_arr = cl_arr - sigma_arr
        upper_2_arr = cl_arr + 2 * sigma_arr
        lower_2_arr = cl_arr - 2 * sigma_arr

        upper_1_arr = apply_plot_line_gaps(upper_1_arr, break_positions)
        lower_1_arr = apply_plot_line_gaps(lower_1_arr, break_positions)
        upper_2_arr = apply_plot_line_gaps(upper_2_arr, break_positions)
        lower_2_arr = apply_plot_line_gaps(lower_2_arr, break_positions)

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

        try:
            for i in range(len(lower_2_arr)):
                if lcl[i] > lower_2_arr[i]:
                    lower_2_arr[i] = lcl[i]
            for i in range(len(lower_1_arr)):
                if lower_2_arr[i] > lower_1_arr[i]:
                    lower_1_arr[i] = lower_2_arr[i]
        except:
            pass

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
            y=plot_ucl_arr,
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
            y=plot_lcl_arr,
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
    Add rule markers and an all-rule-break overlay to a subplot.

    Updated behavior:
    - only the single most common rule for the whole figure is visible by default
    - all other rule traces (including "All Rule Breaks") start as legend-only
    """
    if violations_df.empty:
        return legend_shown_rules
    
    break_df = violations_df
    contrib_df = violations_df.iloc[0:0].copy()
    if "count_as_break" in violations_df.columns:
        break_df = violations_df[violations_df["count_as_break"] == True].copy()
        contrib_df = violations_df[violations_df["count_as_break"] == False].copy()

    if break_df.empty and contrib_df.empty:
        return legend_shown_rules

    merge_cols = ["date", y_col]
    if x_col not in merge_cols:
        merge_cols.append(x_col)

    merged = (
        source_df[merge_cols]
        .merge(
            break_df[["date", "rule", "rule_description"]].drop_duplicates(),
            on="date",
            how="inner",
        )
        .drop_duplicates(subset=["date", "rule"])
        .sort_values(
            ["rule", "date"],
            key=lambda s: s.map(RULE_SORT_ORDER) if s.name == "rule" else s,
        )
    )

    all_rule_breaks_df = (
        break_df
        .groupby("date", as_index=False)
        .agg(
            rule_count=("rule", "nunique"),
            rules=("rule", lambda x: sorted(set(x), key=lambda r: RULE_SORT_ORDER.get(r, 999))),
        )
        .copy()
    )

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

    # Contributory (non-counting) points: white squares with red outline
    if not contrib_df.empty:
        contrib_points = (
            source_df[merge_cols]
            .merge(contrib_df[["date"]].drop_duplicates(), on="date", how="inner")
            .drop_duplicates(subset=["date"])
        )

        x_vals = contrib_points["date"] if x_axis_mode == "Time" else contrib_points[x_col]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=contrib_points[y_col],
                mode="markers",
                name="Contributory points",
                legendgroup="Contributory points",
                showlegend=("Contributory points" not in legend_shown_rules),
                visible="legendonly",
                marker=dict(
                    symbol="square",
                    size=11,
                    color="white",
                    line=dict(width=2, color="#FF0000"),
                ),
                hovertemplate="Contributory (non-counting)<br>Value: %{y:.5f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        legend_shown_rules.add("Contributory points")

    # All-rule-break overlay
    if not all_rule_breaks_df.empty:
        all_rule_breaks_df = all_rule_breaks_df.merge(source_df[merge_cols], on="date", how="left")

        style = RULE_STYLE_MAP["All Rule Breaks"]
        show_legend = "All Rule Breaks" not in legend_shown_rules
        all_breaks_visible_state = True if default_visible_rule == "All Rule Breaks" else "legendonly"

        hover_text = []
        for _, record in all_rule_breaks_df.iterrows():
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
                f"Rule break(s) triggered:<br>{rules_text}"
            )

        x_vals = all_rule_breaks_df["date"] if x_axis_mode == "Time" else all_rule_breaks_df[x_col]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=all_rule_breaks_df[y_col],
                mode="markers",
                name=style["label"],
                legendgroup="All Rule Breaks",
                showlegend=show_legend,
                visible=all_breaks_visible_state,
                marker=dict(
                    symbol="square",
                    size=11,
                    color=style["color"],          # red fill
                    line=dict(
                        width=1.5,
                        color="#000000",            # black border
                    ),
                ),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
            ),
            row=row,
            col=col,
        )
        legend_shown_rules.add("All Rule Breaks")

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
    break_indices = limits.get("break_indices", [])

    if x_axis_mode == "Time":
        x_col = "date"
        x_axis_title = "Date / Time"
    elif x_axis_mode == "Index":
        x_col = "Index"
        x_axis_title = "Index"
    else:
        x_col = "subgroup_number"
        x_axis_title = "Subgroup Number"

    default_visible_rule = "All Rule Breaks"

    try:
        flag = secondary["enabled"]
    except:
        flag = True
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(
            f"{primary['label']} Chart",
            f"{secondary['label']} Chart" if flag else "",
        ),
    )

    # Resolve x values for break lines
    break_x_values = [
        chart_df.iloc[idx][x_col]
        for idx in break_indices
        if 0 <= idx < len(chart_df)
    ]
    
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

    try:
        add_limit_lines(
            fig=fig,
            x_values=chart_df[x_col],
            cl=primary["CL_series"],
            ucl=primary["UCL_series"],
            lcl=primary["LCL_series"],
            sigma=primary.get("sigma", None),
            row=1,
            col=1,
            show_legend_once=True,
            break_positions=break_indices,
        )
    except:
        add_limit_lines(
            fig=fig,
            x_values=chart_df[x_col],
            cl=primary["CL"],
            ucl=primary["UCL"],
            lcl=primary["LCL"],
            sigma=primary.get("sigma", None),
            row=1,
            col=1,
            show_legend_once=True,
            break_positions=break_indices,
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
    if secondary["CL"] is not None:
        sec_df = chart_df.dropna(subset=[secondary["y_col"]]).copy()
        sec_idx = sec_df.index.to_numpy()

        # First available point at/after each break for visual line splitting
        secondary_break_positions: list[int] = []
        for break_idx in break_indices:
            local_pos = next((pos for pos, orig_idx in enumerate(sec_idx) if orig_idx >= break_idx), None)
            if local_pos is not None:
                secondary_break_positions.append(local_pos)

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
            cl=np.asarray(secondary["CL_series"], dtype=float)[sec_idx],
            ucl=np.asarray(secondary["UCL_series"], dtype=float)[sec_idx],
            lcl=np.asarray(secondary["LCL_series"], dtype=float)[sec_idx],
            sigma=(
                np.asarray(secondary["sigma_series"], dtype=float)[sec_idx]
                if secondary["sigma_series"] is not None
                else None
            ),
            row=2,
            col=1,
            show_legend_once=False,
            break_positions=secondary_break_positions,
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

    # Structural break lines on both subplots
    if break_x_values:
        add_structural_break_lines(fig, break_x_values=break_x_values, rows=[1, 2], col=1)

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


    MAX_MONTH_TICKS = 36  # e.g., allow up to 36 monthly labels (3 years)
    DATE_TICK_FMT_D3 = "%b %Y"  # 29 Apr 2026 (Plotly d3 time format)
    # If you prefer full month: DATE_TICK_FMT_D3 = "%d %B %Y"
    # Only apply dense monthly tick labels when it won't overcrowd the axis
    if x_axis_mode == "Time":
        x_dates = pd.to_datetime(chart_df[x_col], errors="coerce").dropna()
        if not x_dates.empty:
            min_d = x_dates.min()
            max_d = x_dates.max()

            # How many monthly ticks would "M1" produce across the range (inclusive)?
            month_count = (max_d.year - min_d.year) * 12 + (max_d.month - min_d.month) + 1

            if month_count <= MAX_MONTH_TICKS:
                # Apply to the shared x-axis (row=2) is usually enough,
                # but we can apply to all x-axes for safety.
                fig.update_xaxes(dtick="M1", tickformat=DATE_TICK_FMT_D3)
                
    fig.update_layout(
        xaxis_showticklabels=True,
        xaxis2_showticklabels=True,
    )

    return fig



# ============================================================
# Validation & Processing
# ============================================================
@st.cache_data(show_spinner=False)
def clean_working_data(
    df: pd.DataFrame,
    measurement_col: str | None,
    date_col: str | None,
    subgroup_col: str | None,
    inspected_col: str | None,
    defects_col: str | None,
    null_treatment: str,
) -> pd.DataFrame:
    """Prepare the working dataset based on selected mappings (variable + attribute charts)."""
    df_work = df.copy()

    # --- Coerce numeric columns ---
    if measurement_col:
        df_work[measurement_col] = coerce_numeric(df_work[measurement_col])
        if null_treatment == "zero":
            df_work[measurement_col] = df_work[measurement_col].fillna(0)
        else:
            df_work = df_work.dropna(subset=[measurement_col])

    if inspected_col:
        df_work[inspected_col] = coerce_numeric(df_work[inspected_col])

    if defects_col:
        df_work[defects_col] = coerce_numeric(df_work[defects_col])

    # --- Parse date if provided ---
    if date_col:
        df_work[date_col] = parse_date(df_work[date_col])

        # If duplicate dates exist, aggregate appropriately depending on available columns.
        duplicate_counts = df_work[date_col].value_counts(dropna=True)
        has_duplicates = (duplicate_counts > 1).any()

        if has_duplicates:
            agg_dict: dict[str, str] = {}
            if measurement_col:
                agg_dict[measurement_col] = "sum"
            if inspected_col:
                agg_dict[inspected_col] = "sum"
            if defects_col:
                agg_dict[defects_col] = "sum"

            # group only if we have something to aggregate
            if agg_dict:
                df_work = (
                    df_work
                    .dropna(subset=[date_col])
                    .groupby(date_col, as_index=False, sort=True)
                    .agg(agg_dict)
                )

            df_work.attrs["dates_aggregated"] = True
            df_work.attrs["aggregated_date_count"] = int((duplicate_counts > 1).sum())
        else:
            df_work.attrs["dates_aggregated"] = False

    # --- Subgroup is categorical ---
    if subgroup_col:
        df_work[subgroup_col] = df_work[subgroup_col].astype(str)

    # --- Basic P-chart sanity filters (avoid divide-by-zero / invalid rows) ---
    if inspected_col and defects_col:
        df_work = df_work.dropna(subset=[inspected_col, defects_col])
        df_work = df_work[df_work[inspected_col] > 0]
        df_work = df_work[(df_work[defects_col] >= 0) & (df_work[defects_col] <= df_work[inspected_col])]

    return df_work


@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def evaluate_chart_validity(
    df_work: pd.DataFrame,
    measurement_col: str | None,
    subgroup_col: str | None,
    inspected_col: str | None,
    defects_col: str | None,
) -> ChartEvaluation:
    """
    Determine which chart types are valid.

    Variable charts:
    - I-MR requires a numeric measurement column and >= 3 measurements.
    - Xbar-R/Xbar-S require subgroup column, >=2 subgroups, >=2 per subgroup,
      supported subgroup sizes; selection rule uses max subgroup size.

    Attribute chart:
    - P requires inspected (n_i) and defects (D_i), with n_i > 0 and 0 <= D_i <= n_i,
      and >= 3 samples.
    """
    messages: list[str] = []
    valid_options: list[str] = []

    # -------------------------
    # I-MR
    # -------------------------
    if measurement_col is None:
        messages.append("❌ **I‑MR**: measurement not selected.")
    else:
        imr_valid = df_work[measurement_col].dropna().shape[0] >= 3
        if imr_valid:
            valid_options.append("I-MR")
            messages.append("✅ **I‑MR**: valid (measurement present; ≥ 3 rows).")
        else:
            messages.append("❌ **I‑MR**: not enough data (need ≥ 3 measurements).")

    # -------------------------
    # Xbar-R / Xbar-S
    # -------------------------
    if measurement_col is None:
        messages.append("❌ **Xbar‑R**: measurement not selected.")
        messages.append("❌ **Xbar‑S**: measurement not selected.")
    elif subgroup_col is None:
        messages.append("❌ **Xbar‑R**: subgroup not selected.")
        messages.append("❌ **Xbar‑S**: subgroup not selected.")
    else:
        subgroup_counts = df_work.groupby(subgroup_col)[measurement_col].count().sort_index()
        unique_subgroups = subgroup_counts.shape[0]
        subgroup_sizes = subgroup_counts.astype(int)

        at_least_two_groups = unique_subgroups >= 2
        at_least_two_per_group = all_groups_at_least_two(subgroup_sizes)

        unsupported_sizes = sorted({int(n) for n in subgroup_sizes.unique() if not supported_n(int(n))})

        if not at_least_two_groups:
            messages.append(f"❌ **Xbar‑R**: only {unique_subgroups} subgroup(s); need ≥ 2 to chart.")
            messages.append(f"❌ **Xbar‑S**: only {unique_subgroups} subgroup(s); need ≥ 2 to chart.")
        elif not at_least_two_per_group:
            messages.append("❌ **Xbar‑R**: some subgroup(s) have < 2 observations (range undefined).")
            messages.append("❌ **Xbar‑S**: some subgroup(s) have < 2 observations (std dev undefined).")
        elif unsupported_sizes:
            messages.append(f"❌ **Xbar‑R**: unsupported subgroup size(s) found {unsupported_sizes}. Supported n is 2–25.")
            messages.append(f"❌ **Xbar‑S**: unsupported subgroup size(s) found {unsupported_sizes}. Supported n is 2–25.")
        else:
            max_n = int(subgroup_sizes.max())
            min_n = int(subgroup_sizes.min())

            if max_n <= 7:
                valid_options.append("Xbar-R")
                messages.append(
                    f"✅ **Xbar‑R**: valid (all subgroup sizes are between {min_n} and {max_n}; "
                    f"use Xbar‑R when subgroup size is 2–7)."
                )
                messages.append("❌ **Xbar‑S**: not selected by rule because all subgroup sizes are ≤ 7.")
            else:
                valid_options.append("Xbar-S")
                messages.append(
                    f"❌ **Xbar‑R**: not selected by rule because one or more subgroup sizes are > 7 "
                    f"(observed range: {min_n} to {max_n})."
                )
                messages.append(
                    f"✅ **Xbar‑S**: valid (one or more subgroup sizes are > 7; "
                    f"use Xbar‑S when subgroup size exceeds 7)."
                )

    # -------------------------
    # P chart
    # -------------------------
    if inspected_col is None or defects_col is None:
        messages.append("❌ **P**: inspected and/or defects not selected.")
    else:
        # after clean_working_data we already filtered invalid rows; just ensure enough remain
        usable = df_work[[inspected_col, defects_col]].dropna()
        p_valid = usable.shape[0] >= 3

        if p_valid:
            valid_options.append("P")
            messages.append("✅ **P**: valid (nᵢ and Dᵢ present; ≥ 3 samples; nᵢ>0; 0≤Dᵢ≤nᵢ).")
        else:
            messages.append("❌ **P**: not enough valid samples (need ≥ 3 rows with nᵢ and Dᵢ).")

    return ChartEvaluation(valid_options=valid_options, messages=messages)


@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def detect_violations_for_chart(
    chart_df: pd.DataFrame,
    limits: dict[str, dict[str, Any]],
    enabled_rules: dict[str, bool],
    rule_points: dict[str, int],
    rule_window_threshold: dict[str, dict[str, int]],
    enable_secondary: bool,
    mark_all_sequence_points: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect primary SPC rule violations and (optional) secondary limit breaches."""
    primary_cfg = limits["primary"]
    secondary_cfg = limits.get("secondary")

    primary_violations = detect_spc_rule_violations(
        chart_df,
        y_col=primary_cfg["y_col"],
        cl=primary_cfg["CL"],
        sigma=primary_cfg["sigma"],
        enabled_rules=enabled_rules,
        rule_points=rule_points,
        rule_window_threshold=rule_window_threshold,
        mark_all_sequence_points=mark_all_sequence_points,
    )

    # Default: no secondary violations unless enabled and configured
    secondary_violations = empty_violations_df()

    has_secondary = (
        enable_secondary
        and secondary_cfg is not None
        and secondary_cfg.get("enabled", True)
        and secondary_cfg.get("y_col") is not None
    )

    if has_secondary:
        secondary_violations = detect_secondary_limit_breaches(
            chart_df,
            y_col=secondary_cfg["y_col"],
            ucl=secondary_cfg["UCL_series"],
            lcl=secondary_cfg["LCL_series"],
            enabled=True,
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
    enable_structural_break_detection: bool = False,
    enabled_rules: dict[str, bool] = {},
    rule_points: dict[str, int] = {},
    rule_window_threshold: dict[str, dict[str, int]] = {},
    enable_secondary: bool = True,
    mark_all_sequence_points: bool = True,
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

        chart_df, limits = get_limits_with_optional_structural_breaks(
            chart_df=chart_df,
            chart_type="I-MR",
            enable_structural_break_detection=enable_structural_break_detection,
        )
        primary_violations, secondary_violations = detect_violations_for_chart(
            chart_df,
            limits,
            enabled_rules=enabled_rules,
            rule_points=rule_points,
            rule_window_threshold=rule_window_threshold,
            enable_secondary=enable_secondary,
            mark_all_sequence_points=mark_all_sequence_points,
        )


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
                "chart_df": chart_df,
                "fig": fig,
                "limits": limits,
                "primary_violations": primary_violations,
                "secondary_violations": secondary_violations,
            }
        )

    return payloads


def render_imr_periodic_options(df_work: pd.DataFrame, date_col: str) -> dict[str, dict[str, Any]]:
    """
    Render controls for additional yearly/quarterly/monthly I-MR charts.

    Updated behavior:
    - When enabled, automatically includes ALL available years, quarters, and months.
    - Returns period_requests with count equal to the number of available periods per granularity.
    - Uses backtrack_mode="all_periods" (because we want each historical period as its own chart).
    """

    create_periodic = st.checkbox(
        "Create yearly / quarterly / monthly I‑MR charts as well",
        value=False,
    )

    if not create_periodic:
        return {}

    freq_map = {"yearly": "Y", "quarterly": "Q", "monthly": "M"}

    # Coerce and validate dates once
    dates = pd.to_datetime(df_work[date_col], errors="coerce").dropna()
    if dates.empty:
        st.info("No valid dates are available for yearly/quarterly/monthly chart creation.")
        return {}

    period_requests: dict[str, dict[str, Any]] = {}

    for granularity, freq in freq_map.items():
        available_periods_total = dates.dt.to_period(freq).nunique()

        if available_periods_total == 0:
            st.info(f"No valid dates are available for {granularity} chart creation.")
            continue

        period_requests[granularity] = {
            # include ALL periods
            "count": int(available_periods_total),
            # force "all_periods" so each year / year-quarter / year-month becomes a chart
            "backtrack_mode": "all_periods",
            # not used in all_periods mode
            "focus_value": None,
            "focus_label": None,
        }

    return period_requests

def render_imr_periodic_charts(
    df_work: pd.DataFrame,
    measurement_col: str,
    date_col: str,
    period_requests: dict[str, dict[str, Any]],
    enable_structural_break_detection: bool,
    split_histograms_by_structure: bool,
    scale_segmented_histograms: bool,
    enabled_rules: dict[str, bool],
    rule_points: dict[str, int],
    rule_window_threshold: dict[str, dict[str, int]],
    enable_secondary: bool,
    show_info: bool,
) -> None:
    """
    Create additional I-MR charts for selected yearly / quarterly / monthly periods.

    Updated behavior:
    - Always uses three top-level tabs: Yearly, Quarterly, Monthly
    - Under each: one sub-tab per available period (ALL periods included)
    - Chart + histogram + limit summary + violations remain together in each sub-tab
    """
    if not period_requests:
        return

    # Build payloads grouped by granularity
    payloads_by_granularity: dict[str, list[dict[str, Any]]] = {"yearly": [], "quarterly": [], "monthly": []}

    for granularity, config in period_requests.items():
        payloads = get_imr_period_chart_payloads(
            df_work=df_work,
            measurement_col=measurement_col,
            date_col=date_col,
            granularity=granularity,
            requested_count=int(config["count"]),  # already set to ALL available
            backtrack_mode=str(config["backtrack_mode"]),  # "all_periods"
            focus_value=config.get("focus_value"),
            enable_structural_break_detection=enable_structural_break_detection,
            enabled_rules=enabled_rules,
            rule_points=rule_points,
            rule_window_threshold=rule_window_threshold,
            enable_secondary=enable_secondary,
        )
        payloads_by_granularity.setdefault(granularity, []).extend(payloads)

    any_payloads = any(payloads_by_granularity.get(g) for g in ["yearly", "quarterly", "monthly"])
    if not any_payloads:
        st.info("No additional yearly, quarterly, or monthly I‑MR charts are available.")
        return

    st.markdown("## Additional I‑MR Charts")

    # Top-level tabs in required order
    top_labels = ["Yearly", "Quarterly", "Monthly"]
    top_keys = ["yearly", "quarterly", "monthly"]
    top_tabs = st.tabs(top_labels)

    for top_tab, granularity in zip(top_tabs, top_keys):
        with top_tab:
            payloads = payloads_by_granularity.get(granularity, [])
            if not payloads:
                st.info(f"No {granularity} charts are available.")
                continue

            # Sub-tabs: one per period label
            sub_labels = [p["period_label"] for p in payloads]
            sub_tabs = st.tabs(sub_labels)

            for sub_tab, payload in zip(sub_tabs, payloads):
                with sub_tab:
                    if payload.get("status") == "skipped":
                        st.warning(payload.get("message", "This chart was skipped."))
                        continue

                    render_limit_summary(
                        chart_df=payload["chart_df"],
                        limits=payload["limits"],
                        split_by_structure=split_histograms_by_structure,
                        use_date_labels=True,
                    )

                    # Plotly: use_container_width is the correct Streamlit arg
                    st.plotly_chart(payload["fig"], use_container_width=True)

                    render_histograms_section(
                        chart_df=payload["chart_df"],
                        limits=payload["limits"],
                        chart_title=f"I-MR Chart — {payload['period_label']}",
                        split_by_structure=split_histograms_by_structure,
                        use_date_labels=True,
                        scale_segmented_histograms=scale_segmented_histograms,
                    )

                    if show_info:
                        render_violations_section(
                            payload.get("primary_violations", []),
                            payload.get("secondary_violations", []),
                        )


# ============================================================
# UI Rendering
# ============================================================
def render_spc_explainer(show_info) -> tuple[dict[str, bool], dict[str, int], dict[str, dict[str, int]], bool]:
    """
    Render SPC explanation and editable rule controls.

    Returns:
      enabled_rules: rule_name -> bool
      rule_points: rule_name -> int (Rules 2,3,4,7,8)
      rule_window_threshold: rule_name -> {"window": int, "threshold": int} (Rules 5,6)
      enable_secondary: bool
    """
    primary_rules = ["Rule 1","Rule 2","Rule 3","Rule 4","Rule 5","Rule 6","Rule 7","Rule 8"]
    if show_info:
        with st.expander("What is SPC and what rules does this app use?", expanded=False):
            st.markdown(
                """
                **Statistical Process Control (SPC)** is a way of monitoring a process over time
                to distinguish normal variation from unusual variation that may need attention.

                This app supports:
                - **I-MR** charts for individual observations
                - **Xbar-R** charts for subgroup averages and ranges
                - **Xbar-S** charts for subgroup averages and standard deviations

                The app checks for the following rule signals on the **primary chart**:
                """
            )

            st.markdown("#### Rule controls")

            enabled_rules: dict[str, bool] = {}
            rule_points: dict[str, int] = {}
            rule_window_threshold: dict[str, dict[str, int]] = {}

            for rule_name in primary_rules:
                # Layout:
                # [Enable] [Inputs...] [Description]
                if rule_name in DEFAULT_RULE_WINDOW_THRESHOLD:
                    c1, c2, c3, c4 = st.columns([0.12, 0.14, 0.14, 0.60])
                elif rule_name in DEFAULT_RULE_POINTS:
                    c1, c2, c4 = st.columns([0.12, 0.18, 0.70])
                    c3 = None
                else:
                    c1, c4 = st.columns([0.12, 0.88])
                    c2 = c3 = None

                with c1:
                    enabled_rules[rule_name] = st.checkbox(
                        "On",
                        value=DEFAULT_RULE_ENABLED.get(rule_name, True),
                        key=f"rule_enabled_{rule_name}",
                    )

                # Single-number rules (2,3,4,7,8)
                if rule_name in DEFAULT_RULE_POINTS and c2 is not None:
                    with c2:
                        rule_points[rule_name] = int(
                            st.number_input(
                                "Pts",
                                min_value=2,
                                max_value=200,
                                value=int(DEFAULT_RULE_POINTS[rule_name]),
                                step=1,
                                key=f"rule_points_{rule_name}",
                            )
                        )

                # Window+threshold rules (5,6)
                if rule_name in DEFAULT_RULE_WINDOW_THRESHOLD and c2 is not None and c3 is not None:
                    defaults = DEFAULT_RULE_WINDOW_THRESHOLD[rule_name]
                    with c2:
                        window_val = int(
                            st.number_input(
                                "Window",
                                min_value=2,
                                max_value=200,
                                value=int(defaults["window"]),
                                step=1,
                                key=f"rule_window_{rule_name}",
                            )
                        )
                    with c3:
                        # Ensure threshold cannot exceed the chosen window
                        threshold_default = min(int(defaults["threshold"]), window_val)
                        threshold_val = int(
                            st.number_input(
                                "Threshold",
                                min_value=1,
                                max_value=window_val,
                                value=threshold_default,
                                step=1,
                                key=f"rule_threshold_{rule_name}",
                            )
                        )
                    rule_window_threshold[rule_name] = {"window": window_val, "threshold": threshold_val}

                with c4:
                    desc = get_rule_description_dynamic(rule_name, rule_points, rule_window_threshold)
                    st.markdown(f"**{rule_name}**: {desc}")

            st.markdown("On the **secondary chart**, the app checks:")
            enable_secondary = st.checkbox(
                "On",
                value=DEFAULT_RULE_ENABLED.get("Secondary chart: point beyond control limit", True),
                key="rule_enabled_secondary_limit_breach",
            )
            st.markdown(
                f"- **Secondary chart: point beyond control limit**: "
                f"{RULE_DISPLAY_TEXT['Secondary chart: point beyond control limit']}"
            )
    else:
        # No UI controls shown: use defaults and return them
        enabled_rules = {
            rule_name: DEFAULT_RULE_ENABLED.get(rule_name, True)
            for rule_name in primary_rules
        }

        # Defaults for single-number rules (2,3,4,7,8)
        rule_points = {
            rule_name: int(DEFAULT_RULE_POINTS[rule_name])
            for rule_name in primary_rules
            if rule_name in DEFAULT_RULE_POINTS
        }

        # Defaults for window+threshold rules (5,6)
        rule_window_threshold = {}
        for rule_name in primary_rules:
            if rule_name in DEFAULT_RULE_WINDOW_THRESHOLD:
                defaults = DEFAULT_RULE_WINDOW_THRESHOLD[rule_name]
                window_val = int(defaults["window"])
                threshold_val = int(defaults["threshold"])

                # Safety: threshold should never exceed window
                threshold_val = min(threshold_val, window_val)

                rule_window_threshold[rule_name] = {
                    "window": window_val,
                    "threshold": threshold_val,
                }

        # Secondary chart rule default
        enable_secondary = DEFAULT_RULE_ENABLED.get(
            "Secondary chart: point beyond control limit",
            True
        )

    return enabled_rules, rule_points, rule_window_threshold, enable_secondary


def render_show_additional_information() -> bool:
    """Render option for showing additional information in the app."""
    return st.checkbox(
        "Show additional information",
        value=False,
        help=(
            "Selecting this option will result in additional information regarding the app's logic and data being shown."
        ),
    )


def render_header() -> tuple[dict[str, bool], dict[str, int], dict[str, dict[str, int]], bool]:
    """Render the App header"""
    st.title(APP_TITLE)
    st.markdown(APP_SUBTITLE)
    return None


def render_sidebar_file_upload() -> Any:
    """Render file upload control in the sidebar."""
    st.header("Upload your Data")
    return st.file_uploader(
        "Upload a CSV or Excel file",
        type=SUPPORTED_UPLOAD_TYPES,
    )


def render_data_preview(df: pd.DataFrame) -> None:
    """Render dataset preview only."""
    with st.expander("Preview (first 10 rows)", expanded=False):
        st.dataframe(df.head(10), width='stretch')

def render_selected_columns_missing_notice(
    df: pd.DataFrame,
    measurement_col: str | None,
    date_col: str | None,
    subgroup_col: str | None,
) -> None:
    """Show a short notice if selected processing columns contain missing values."""
    selected_cols = [col for col in [measurement_col, date_col, subgroup_col] if col is not None]
    if not selected_cols:
        return

    missing_count = int(df[selected_cols].isna().sum().sum())
    if missing_count > 0:
        st.info(
            f"There {'is' if missing_count == 1 else 'are'} {missing_count:,} missing "
            f"cell{'s' if missing_count != 1 else ''} in the selected processing column"
            f"{'s' if len(selected_cols) != 1 else ''}."
        )


def render_column_mapping(
    df: pd.DataFrame,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Render column mapping controls in the sidebar."""
    st.subheader("Column Mapping")
    all_cols = list(df.columns)

    # Variable charts (I-MR, Xbar-R, Xbar-S)
    measurement_col = st.selectbox(
        "Measurement (numeric; required for I‑MR / Xbar charts)",
        options=["—"] + all_cols,
        index=0,
    )
    date_col = st.selectbox(
        "Date (optional)",
        options=["(none)"] + all_cols,
        index=0,
    )
    subgroup_col = st.selectbox(
        "Subgroup (optional, categorical; required for Xbar charts)",
        options=["(none)"] + all_cols,
        index=0,
    )

    st.markdown("---")
    st.caption("P‑chart (attributes): provide inspected count nᵢ and nonconforming count Dᵢ")

    inspected_col = st.selectbox(
        "Inspected (numeric; required for P chart)",
        options=["(none)"] + all_cols,
        index=0,
    )
    defects_col = st.selectbox(
        "Nonconforming / Defects (numeric; required for P chart)",
        options=["(none)"] + all_cols,
        index=0,
    )

    m_col = None if measurement_col == "—" else measurement_col
    d_col = None if date_col == "(none)" else date_col
    g_col = None if subgroup_col == "(none)" else subgroup_col
    n_col = None if inspected_col == "(none)" else inspected_col
    dft_col = None if defects_col == "(none)" else defects_col

    return m_col, d_col, g_col, n_col, dft_col


def render_null_treatment_option(show_info) -> str:
    """Render null/empty measurement treatment options in the sidebar."""
    if show_info:
        st.sidebar.subheader("Null / Empty observation handling")
        selected_label = st.sidebar.radio(
            "How should empty/null measurement observations be treated?",
            options=list(NULL_TREATMENT_OPTIONS.keys()),
            index=0,
        )
        return NULL_TREATMENT_OPTIONS[selected_label]
    else:
        return "discard"

def render_structural_break_option(show_info) -> bool:
    """Render sidebar option for automatic structural break detection."""
    return st.checkbox(
        "Automatically detect structural breaks and re-baseline chart limits",
        value=False,
        help=(
            "When enabled, the app detects sequential changes in the process mean and/or "
            "standard deviation using only past and current observations. "
            "When a break is confirmed, new center lines, control limits, and sigma lines "
            "are calculated from that break onward."
        ),
    )


def render_missing_date_zero_option(show_info) -> bool:
    "Render sidebar option to insert missing dates with zero values (requires date column)."
    if show_info:
        st.sidebar.subheader("Missing date handling")
        return st.sidebar.checkbox(
            "Insert missing dates and set their measurement to 0",
            value=True,
            help=(
                "If a date column is mapped, this will ensure every day between the "
                "minimum and maximum date exists. Missing days are inserted with measurement=0 "
                "to capture events like shutdowns/maintenance as special-cause variation."
            ),
        )
    else:
        return True


def render_histogram_segment_option(enable_structural_break_detection: bool) -> bool:
    """
    Render sidebar option for segment-based histogram and limit-summary behavior
    per SPC chart.

    This section is only shown when automatic structural break detection is enabled.
    """
    if not enable_structural_break_detection:
        return False

    st.sidebar.subheader("Histogram and limit summary options")
    return st.sidebar.checkbox(
        "Create separate histograms and limit summaries for structural-break segments",
        value=True,
        help=(
            "If unticked, each SPC chart shows one histogram and one limit summary "
            "for the full chart. If ticked, separate histograms and limit summaries "
            "are shown for each structural segment identified in that chart."
        ),
    )


def render_histogram_scaling_option(split_histograms_by_structure: bool) -> bool:
    """
    Render sidebar option for scaling segmented histograms within each chart.

    This option only appears when segmented histograms are enabled.
    """
    if not split_histograms_by_structure:
        return False

    return st.sidebar.checkbox(
        "Scale segmented histograms within each chart",
        value=True,
        help=(
            "If ticked, segmented histograms for a single SPC chart will use the same "
            "y-axis scale so they can be compared visually. This applies separately "
            "to the main chart and to each additional chart."
        ),
    )


def render_sequence_rule_marker_option(show_info) -> bool:
    "Render sidebar option controlling how sequence-based rule points are marked/count."
    if show_info:
        st.sidebar.subheader("Sequence-based rule marking")
        return st.sidebar.checkbox(
            "Mark all points in a sequence-based rule break",
            value=False,
            help=(
                "If unticked, only the final point that triggers a sequence-based rule break is "
                "counted/marked as a rule break. Earlier contributing points are shown as "
                "white squares with a red outline and do not count as rule breaks."
            ),
        )
    else:
        return False

def render_validity_messages(evaluation: ChartEvaluation) -> None:
    """Render chart validity assessment results."""
    st.subheader("Valid SPC chart types for this data")
    for message in evaluation.messages:
        st.markdown(message)

def get_histogram_bin_count(n: int) -> int:
    """Choose a reasonable histogram bin count based on sample size."""
    return max(15, min(80, int(2 * np.sqrt(max(n, 1)))))


def compute_scaled_histogram_settings(
    series_list: list[pd.Series],
) -> dict[str, Any] | None:
    """
    Compute common histogram settings for a group of segmented histograms
    belonging to one SPC chart.

    Returns a dict containing:
    - xbins: Plotly histogram bin settings
    - yaxis_range: common y-axis range
    - yaxis_dtick: common y-axis tick interval

    If there is not enough valid data, returns None.
    """
    cleaned_series = [
        pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        for series in series_list
    ]
    cleaned_series = [arr for arr in cleaned_series if len(arr) > 0]

    if not cleaned_series:
        return None

    combined = np.concatenate(cleaned_series)
    if len(combined) == 0:
        return None

    x_min = float(np.min(combined))
    x_max = float(np.max(combined))

    # Handle degenerate case where all values are identical
    if np.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5

    nbins = get_histogram_bin_count(len(combined))
    bin_edges = np.linspace(x_min, x_max, nbins + 1)
    bin_size = float(bin_edges[1] - bin_edges[0])

    max_count = 0
    for arr in cleaned_series:
        counts, _ = np.histogram(arr, bins=bin_edges)
        if len(counts) > 0:
            max_count = max(max_count, int(np.max(counts)))

    if max_count <= 0:
        max_count = 1

    # Small headroom above tallest bar
    y_max = int(np.ceil(max_count * 1.08))

    # Reasonable tick interval
    y_dtick = max(1, int(np.ceil(y_max / 5)))

    return {
        "xbins": {
            "start": x_min,
            "end": x_max,
            "size": bin_size,
        },
        "yaxis_range": [0, y_max],
        "yaxis_dtick": y_dtick,
    }


def build_histogram_figure(
    series: pd.Series,
    title: str,
    x_axis_title: str,
    xbins: dict[str, float] | None = None,
    yaxis_range: list[float] | None = None,
    yaxis_dtick: float | None = None,
):
    """Create a histogram figure for the provided series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()

    mean = clean.mean()
    std = clean.std(ddof=1)
    p75 = np.percentile(clean, 75)
    p90 = np.percentile(clean, 90)

    minus_3s = mean - 3 * std
    plus_3s  = mean + 3 * std

    histogram_kwargs = {
        "x": clean,
        "marker": dict(color="#5B5B5B"),
        "opacity": 0.9,
        "name": x_axis_title,
    }

    if xbins is None:
        histogram_kwargs["nbinsx"] = get_histogram_bin_count(len(clean))
    else:
        histogram_kwargs["xbins"] = xbins

    fig = go.Figure()
    fig.add_trace(go.Histogram(**histogram_kwargs))

    def add_vline(x, label, color):
        fig.add_vline(
            x=x,
            line_width=2,
            line_dash="dot",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
            annotation_font_size=11,
            opacity=0.9,
        )

    # Mean
    add_vline(mean, "Mean", "#d62728")

    # Percentiles
    add_vline(p75, "p75", "#2ca02c")
    add_vline(p90, "p90", "#ff7f0e")

    # ±3 Sigma
    add_vline(minus_3s, "−3σ", "#686868")
    add_vline(plus_3s, "+3σ", "#686868")

    values = [mean, p75, p90, minus_3s, plus_3s]
    
    x_kde, y_kde = compute_kde(clean.to_numpy())

    if x_kde is not None:
        # Determine bin width (must match histogram)
        if xbins is not None and "size" in xbins:
            bin_width = xbins["size"]
        else:
            # fallback: approximate from data and bin count
            bin_count = get_histogram_bin_count(len(clean))
            bin_width = (clean.max() - clean.min()) / bin_count

        # Scale density to histogram counts
        y_kde_scaled = y_kde * len(clean) * bin_width

        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde_scaled,
                mode="lines",
                name="Kernel Density",
                line=dict(
                            width=2,
                            color="crimson",
                        ),
            )
        )


    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title=x_axis_title,
        yaxis_title="Count",
        bargap=0.08,
        height=420,
        margin=dict(l=40, r=20, t=70, b=40),
        showlegend=False,
    )

    fig.update_layout(
        bargap=0.05,
        legend=dict(orientation="h"),
    )


    if yaxis_range is not None:
        fig.update_yaxes(range=yaxis_range)

    if yaxis_dtick is not None:
        fig.update_yaxes(dtick=yaxis_dtick)

    return fig, values



def format_segment_label(
    seg_df: pd.DataFrame,
    segment_number: int,
    start_obs: int,
    end_obs: int,
    use_date_labels: bool,
) -> str:
    """
    Build the label for a histogram segment.

    If use_date_labels is True and a real date column is available in the chart_df,
    label the segment using its date range. Otherwise, fall back to observation numbers.
    """
    if use_date_labels and "date" in seg_df.columns:
        valid_dates = pd.to_datetime(seg_df["date"], errors="coerce").dropna()

        if not valid_dates.empty:
            start_dt = valid_dates.min()
            end_dt = valid_dates.max()

            # Date-only formatting if timestamps are normalized, otherwise include time
            if (
                start_dt == pd.Timestamp(start_dt).normalize()
                and end_dt == pd.Timestamp(end_dt).normalize()
            ):
                start_label = start_dt.strftime("%Y-%m-%d")
                end_label = end_dt.strftime("%Y-%m-%d")
            else:
                start_label = start_dt.strftime("%Y-%m-%d %H:%M")
                end_label = end_dt.strftime("%Y-%m-%d %H:%M")

            if start_label == end_label:
                return f"Segment {segment_number} · {start_label}"

            return f"Segment {segment_number} · {start_label} to {end_label}"

    return f"Segment {segment_number} · Obs {start_obs}–{end_obs}"


def render_histograms_section(
    chart_df: pd.DataFrame,
    limits: dict[str, dict[str, Any]],
    chart_title: str,
    split_by_structure: bool,
    use_date_labels: bool = False,
    scale_segmented_histograms: bool = False,
) -> None:
    """
    Render histogram(s) for the primary series shown in the SPC chart.

    Behavior:
    - If split_by_structure is False: show one histogram for the full primary series
      displayed in the chart.
    - If split_by_structure is True and multiple structural segments exist:
      show one histogram per segment inside tabs within an expander.
    - If split_by_structure is True but there is only one segment:
      fall back to a single histogram.
    """
    primary = limits["primary"]
    y_col = primary["y_col"]
    y_label = primary["label"]

    segment_ranges = limits.get("segment_ranges", [(0, len(chart_df))])
    segment_summaries = limits.get("segment_summaries", [])

    # Fallback to one histogram if segmentation is not requested or no real split exists
    if (not split_by_structure) or len(segment_ranges) <= 1:
        with st.expander(f"Histogram — {chart_title}", expanded=False):
            clean = pd.to_numeric(chart_df[y_col], errors="coerce").dropna()
            if clean.empty:
                st.info("No valid values are available to plot a histogram for this chart.")
            else:
                if use_date_labels and "date" in chart_df.columns:
                    valid_dates = pd.to_datetime(chart_df["date"], errors="coerce").dropna()
                    if not valid_dates.empty:
                        start_dt = valid_dates.min()
                        end_dt = valid_dates.max()

                        if (
                            start_dt == pd.Timestamp(start_dt).normalize()
                            and end_dt == pd.Timestamp(end_dt).normalize()
                        ):
                            start_label = start_dt.strftime("%Y-%m-%d")
                            end_label = end_dt.strftime("%Y-%m-%d")
                        else:
                            start_label = start_dt.strftime("%Y-%m-%d %H:%M")
                            end_label = end_dt.strftime("%Y-%m-%d %H:%M")

                        if start_label == end_label:
                            st.caption(f"Date range: {start_label}")
                        else:
                            st.caption(f"Date range: {start_label} to {end_label}")

                fig, values = build_histogram_figure(
                    chart_df[y_col],
                    title=f"{chart_title} — {y_label} Histogram",
                    x_axis_title=y_label,
                )
                st.plotly_chart(fig, width='stretch')

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(f"**Mean:** {format_metric_value(values[0])}")
                c2.markdown(f"**75th Percentile:** {format_metric_value(values[1])}")
                c3.markdown(f"**90th Percentile:** {format_metric_value(values[2])}")
                c4.markdown(f"**-3σ:** {format_metric_value(values[3])}")
                c5.markdown(f"**+3σ:** {format_metric_value(values[4])}")
        return

    # Segmented histograms
    with st.expander(f"Histograms by Structural Segment — {chart_title}", expanded=False):
        tab_labels = []
        segment_series_list: list[pd.Series] = []

        for idx, (start, end) in enumerate(segment_ranges, start=1):
            seg_df = chart_df.iloc[start:end].copy()
            segment_series_list.append(seg_df[y_col])

            if idx <= len(segment_summaries):
                seg_summary = segment_summaries[idx - 1]
                start_obs = int(seg_summary["start_obs"])
                end_obs = int(seg_summary["end_obs"])
            else:
                start_obs = start + 1
                end_obs = end

            tab_labels.append(
                format_segment_label(
                    seg_df=seg_df,
                    segment_number=idx,
                    start_obs=start_obs,
                    end_obs=end_obs,
                    use_date_labels=use_date_labels,
                )
            )

        scale_settings = None
        if scale_segmented_histograms and len(segment_ranges) > 1:
            scale_settings = compute_scaled_histogram_settings(segment_series_list)

        if scale_settings is not None:
            st.caption("Segmented histograms in this chart use a common y-axis scale.")

        tabs = st.tabs(tab_labels)

        for idx, ((start, end), tab) in enumerate(zip(segment_ranges, tabs), start=1):
            with tab:
                seg_df = chart_df.iloc[start:end].copy()
                clean = pd.to_numeric(seg_df[y_col], errors="coerce").dropna()

                if clean.empty:
                    st.info("No valid values are available for this structural segment.")
                    continue

                fig, values = build_histogram_figure(
                    seg_df[y_col],
                    title=f"{chart_title} — {y_label} Histogram (Segment {idx})",
                    x_axis_title=y_label,
                    xbins=scale_settings["xbins"] if scale_settings is not None else None,
                    yaxis_range=scale_settings["yaxis_range"] if scale_settings is not None else None,
                    yaxis_dtick=scale_settings["yaxis_dtick"] if scale_settings is not None else None,
                )
                st.plotly_chart(fig, width='stretch')

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(f"**Mean:** {format_metric_value(values[0])}")
                c2.markdown(f"**75th Percentile:** {format_metric_value(values[1])}")
                c3.markdown(f"**90th Percentile:** {format_metric_value(values[2])}")
                c4.markdown(f"**-3σ:** {format_metric_value(values[3])}")
                c5.markdown(f"**+3σ:** {format_metric_value(values[4])}")

def render_limit_summary_metrics_for_segment(
    segment_summary: dict[str, Any],
    primary_label: str,
    secondary_label: str | None,
) -> None:
    """Render the limit metrics for one segment."""
    p1, p2, p3 = st.columns(3)
    p1.metric(f"{primary_label} CL", format_metric_value(segment_summary.get("primary_CL")))
    p2.metric(f"{primary_label} UCL", format_metric_value(segment_summary.get("primary_UCL")))
    p3.metric(f"{primary_label} LCL", format_metric_value(segment_summary.get("primary_LCL")))

    if secondary_label:
        s1, s2, s3 = st.columns(3)
        s1.metric(f"{secondary_label} CL", format_metric_value(segment_summary.get("secondary_CL")))
        s2.metric(f"{secondary_label} UCL", format_metric_value(segment_summary.get("secondary_UCL")))
        s3.metric(f"{secondary_label} LCL", format_metric_value(segment_summary.get("secondary_LCL")))


def render_limit_summary(
    chart_df: pd.DataFrame,
    limits: dict[str, dict[str, Any]],
    split_by_structure: bool,
    use_date_labels: bool = False,
) -> None:
    """Render limit summary, optionally split into tabs by structural segment."""
    primary = limits["primary"]
    secondary = limits["secondary"]
    segment_summaries = limits.get("segment_summaries", [])
    segment_ranges = limits.get("segment_ranges", [(0, len(chart_df))])

    with st.expander("Limit Summary", expanded=False):
        # Default single-summary behavior
        if (not split_by_structure) or len(segment_ranges) <= 1 or len(segment_summaries) <= 1:
            if len(segment_summaries) > 1:
                st.caption(
                    f"{len(segment_summaries)} structural segments detected. "
                    f"The metrics below reflect the most recent segment."
                )

            # Use the chart-level scalar values (current behavior)
            p1, p2, p3 = st.columns(3)
            try:
                flag = len(primary["CL"]) == 1
            except:
                flag = True
            p1.metric(f"{primary['label']} CL", format_metric_value(primary["CL"]) if flag else "Varies")
            try:
                flag = len(primary["UCL"]) == 1
            except:
                flag = True
            p2.metric(f"{primary['label']} UCL", format_metric_value(primary["UCL"]) if flag else "Varies")
            try:
                flag = len(primary["LCL"]) == 1
            except:
                flag = True
            p3.metric(f"{primary['label']} LCL", format_metric_value(primary["LCL"]) if flag else "Varies")

            if secondary:
                s1, s2, s3 = st.columns(3)
                s1.metric(f"{secondary['label']} CL", format_metric_value(secondary["CL"]))
                s2.metric(f"{secondary['label']} UCL", format_metric_value(secondary["UCL"]))
                s3.metric(f"{secondary['label']} LCL", format_metric_value(secondary["LCL"]))

            return

        # Segmented-tab behavior
        st.caption("Select a structural segment tab to view that segment’s limit summary.")

        tab_labels = []
        for idx, (start, end) in enumerate(segment_ranges, start=1):
            seg_df = chart_df.iloc[start:end].copy()

            if idx <= len(segment_summaries):
                seg_summary = segment_summaries[idx - 1]
                start_obs = int(seg_summary["start_obs"])
                end_obs = int(seg_summary["end_obs"])
            else:
                start_obs = start + 1
                end_obs = end

            tab_labels.append(
                format_segment_label(
                    seg_df=seg_df,
                    segment_number=idx,
                    start_obs=start_obs,
                    end_obs=end_obs,
                    use_date_labels=use_date_labels,
                )
            )

        tabs = st.tabs(tab_labels)

        for idx, tab in enumerate(tabs, start=1):
            with tab:
                if idx <= len(segment_summaries):
                    seg_summary = segment_summaries[idx - 1]
                    render_limit_summary_metrics_for_segment(
                        segment_summary=seg_summary,
                        primary_label=primary["label"],
                        secondary_label=secondary["label"] if secondary else None,
                    )
                else:
                    st.info("No segment limit summary is available for this tab.")


def build_rule_break_counts_df(violations_df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table showing number of occurrences per rule break."""
    if violations_df.empty:
        return pd.DataFrame(columns=["rule", "occurrences", "rule_description"])
    
    df_count = violations_df
    if "count_as_break" in df_count.columns:
        df_count = df_count[df_count["count_as_break"] == True]

    summary = (
        df_count.groupby("rule", as_index=False)
         .size()
         .rename(columns={"size": "occurrences"})
    )

    summary["rule_description"] = summary["rule"].map(RULE_DISPLAY_TEXT)
    summary["sort_order"] = summary["rule"].map(lambda x: RULE_SORT_ORDER.get(x, 999))

    return (
        summary.sort_values(["sort_order", "rule"])
        .drop(columns=["sort_order"])
        .reset_index(drop=True)
    )


def render_violations_section(
    primary_violations: pd.DataFrame,
    secondary_violations: pd.DataFrame,
) -> None:
    """Render rule-break count summaries inside an expander with tabs."""
    primary_summary = build_rule_break_counts_df(primary_violations)
    secondary_summary = build_rule_break_counts_df(secondary_violations)

    with st.expander("Rule-break Summary", expanded=False):
        tab1, tab2 = st.tabs(["Primary Chart", "Secondary Chart"])

        with tab1:
            if primary_summary.empty:
                st.success("No primary-chart SPC rule breaks detected.")
            else:
                st.dataframe(primary_summary, width='stretch')

        with tab2:
            if secondary_summary.empty:
                st.success("No secondary-chart rule breaks detected.")
            else:
                st.dataframe(secondary_summary, width='stretch')


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

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start date",
            value=default_start.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key="imr_main_start_date",
        )

    with col2:
        end_date = st.date_input(
            "End date",
            value=default_end.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key="imr_main_end_date",
        )

    # Ensure the range is valid (swap if user picks them “backwards”)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    selected_range = (start_date, end_date)


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
def run_imr_flow(
    df_work: pd.DataFrame,
    measurement_col: str,
    date_col: str | None,
    enable_structural_break_detection: bool,
    split_histograms_by_structure: bool,
    scale_segmented_histograms: bool,
    enabled_rules: dict[str, bool],
    rule_points: dict[str, int],
    rule_window_threshold: dict[str, dict[str, int]],
    enable_secondary: bool,
    mark_all_sequence_points: bool = True,
    fill_missing_dates_zero: bool = False,
    show_info: bool = False,
) -> None:
    """Execute I-MR build, limits, violations, and rendering."""
    df_for_main_chart = df_work.copy()

    # If date_col is provided and user enabled the option, insert missing dates with zeros
    if date_col and fill_missing_dates_zero:
        df_for_main_chart, missing_dates = fill_missing_dates_with_zero(
            df_for_main_chart, date_col=date_col, measurement_col=measurement_col, freq="D", agg="sum"
        )
        # Optional: show count
        if show_info:
            st.info(f"Inserted {len(missing_dates)} missing date(s) with measurement=0.")

    df_work = df_for_main_chart
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
    else:
        df_for_main_chart = df_for_main_chart.tail(365).copy()

        if len(df_for_main_chart) < 3:
            st.warning("Fewer than 3 valid observations are available for the main I‑MR chart.")
            st.stop()

    chart_df = build_imr_chart_df(df_for_main_chart, measurement_col=measurement_col, date_col=date_col)
    chart_df, limits = get_limits_with_optional_structural_breaks(
        chart_df=chart_df,
        chart_type="I-MR",
        enable_structural_break_detection=enable_structural_break_detection,
    )

    x_axis_mode = "Time" if date_col else "Index"
    primary_violations, secondary_violations = detect_violations_for_chart(
        chart_df,
        limits,
        enabled_rules=enabled_rules,
        rule_points=rule_points,
        rule_window_threshold=rule_window_threshold,
        enable_secondary=enable_secondary,
        mark_all_sequence_points=mark_all_sequence_points,
    )

    render_limit_summary(
        chart_df=chart_df,
        limits=limits,
        split_by_structure=split_histograms_by_structure,
        use_date_labels=bool(date_col),
    )
    
    fig = plot_spc_chart(
        chart_df=chart_df,
        limits=limits,
        title="I-MR Chart",
        primary_violations=primary_violations,
        secondary_violations=secondary_violations,
        x_axis_mode=x_axis_mode,
    )

    apply_uniform_date_format(fig)
    st.plotly_chart(fig, width='stretch')
    
    render_histograms_section(
        chart_df=chart_df,
        limits=limits,
        chart_title="I-MR Chart",
        split_by_structure=split_histograms_by_structure,
        use_date_labels=bool(date_col),
        scale_segmented_histograms=scale_segmented_histograms,
    )
    
    if show_info:
        render_violations_section(primary_violations, secondary_violations)

    if date_col:
        period_requests = render_imr_periodic_options(df_work=df_work, date_col=date_col)
        render_imr_periodic_charts(
            df_work=df_work,
            measurement_col=measurement_col,
            date_col=date_col,
            period_requests=period_requests,
            enable_structural_break_detection=enable_structural_break_detection,
            split_histograms_by_structure=split_histograms_by_structure,
            scale_segmented_histograms=scale_segmented_histograms,
            enabled_rules=enabled_rules,
            rule_points=rule_points,
            rule_window_threshold=rule_window_threshold,
            enable_secondary=enable_secondary,
            show_info=show_info,
        )


def run_xbar_r_flow(
    df_work: pd.DataFrame,
    measurement_col: str,
    subgroup_col: str,
    enable_structural_break_detection: bool,
    split_histograms_by_structure: bool,
    scale_segmented_histograms: bool,
    enabled_rules: dict[str, bool],
    rule_points: dict[str, int],
    rule_window_threshold: dict[str, dict[str, int]],
    enable_secondary: bool,
    mark_all_sequence_points: bool = True,
    show_info: bool = False,
) -> None:
    """Execute Xbar-R build, limits, violations, and rendering."""
    unsupported_sizes = check_unsupported_group_sizes(df_work, measurement_col, subgroup_col)
    if unsupported_sizes:
        st.error(
            f"Some subgroup sizes are unsupported for constants (n in {unsupported_sizes}). "
            f"Supported n is 2–25."
        )
        st.stop()

    chart_df = build_xbar_r_chart_df(df_work, measurement_col=measurement_col, subgroup_col=subgroup_col)
    chart_df, limits = get_limits_with_optional_structural_breaks(
        chart_df=chart_df,
        chart_type="Xbar-R",
        enable_structural_break_detection=enable_structural_break_detection,
    )

    primary_violations, secondary_violations = detect_violations_for_chart(
        chart_df,
        limits,
        enabled_rules=enabled_rules,
        rule_points=rule_points,
        rule_window_threshold=rule_window_threshold,
        enable_secondary=enable_secondary,
        mark_all_sequence_points=mark_all_sequence_points,
    )

    render_limit_summary(
        chart_df=chart_df,
        limits=limits,
        split_by_structure=split_histograms_by_structure,
        use_date_labels=False,
    )

    fig = plot_spc_chart(
        chart_df=chart_df,
        limits=limits,
        title="Xbar–R Chart",
        primary_violations=primary_violations,
        secondary_violations=secondary_violations,
        x_axis_mode="Subgroup",
    )

    apply_uniform_date_format(fig)
    st.plotly_chart(fig, width='stretch')

    render_histograms_section(
        chart_df=chart_df,
        limits=limits,
        chart_title="Xbar–R Chart",
        split_by_structure=split_histograms_by_structure,
        use_date_labels=False,
        scale_segmented_histograms=scale_segmented_histograms,
    )

    if show_info:
        render_violations_section(primary_violations, secondary_violations)


def run_xbar_s_flow(
    df_work: pd.DataFrame,
    measurement_col: str,
    subgroup_col: str,
    enable_structural_break_detection: bool,
    split_histograms_by_structure: bool,
    scale_segmented_histograms: bool,
    enabled_rules: dict[str, bool],
    rule_points: dict[str, int],
    rule_window_threshold: dict[str, dict[str, int]],
    enable_secondary: bool,
    mark_all_sequence_points: bool = True,
    show_info: bool = False,
) -> None:
    """Execute Xbar-S build, limits, violations, and rendering."""
    unsupported_sizes = check_unsupported_group_sizes(df_work, measurement_col, subgroup_col)
    if unsupported_sizes:
        st.error(
            f"Some subgroup sizes are unsupported for constants (n in {unsupported_sizes}). "
            f"Supported n is 2–25."
        )
        st.stop()

    chart_df = build_xbar_s_chart_df(df_work, measurement_col=measurement_col, subgroup_col=subgroup_col)
    chart_df, limits = get_limits_with_optional_structural_breaks(
        chart_df=chart_df,
        chart_type="Xbar-S",
        enable_structural_break_detection=enable_structural_break_detection,
    )

    primary_violations, secondary_violations = detect_violations_for_chart(
        chart_df,
        limits,
        enabled_rules=enabled_rules,
        rule_points=rule_points,
        rule_window_threshold=rule_window_threshold,
        enable_secondary=enable_secondary,
        mark_all_sequence_points=mark_all_sequence_points,
    )

    render_limit_summary(
        chart_df=chart_df,
        limits=limits,
        split_by_structure=split_histograms_by_structure,
        use_date_labels=False,
    )

    fig = plot_spc_chart(
        chart_df=chart_df,
        limits=limits,
        title="Xbar–S Chart",
        primary_violations=primary_violations,
        secondary_violations=secondary_violations,
        x_axis_mode="Subgroup",
    )

    apply_uniform_date_format(fig)
    st.plotly_chart(fig, width='stretch')

    render_histograms_section(
        chart_df=chart_df,
        limits=limits,
        chart_title="Xbar–S Chart",
        split_by_structure=split_histograms_by_structure,
        use_date_labels=False,
        scale_segmented_histograms=scale_segmented_histograms,
    )

    if show_info:
        render_violations_section(primary_violations, secondary_violations)


def run_p_flow(
    df_work: pd.DataFrame,
    inspected_col: str,
    defects_col: str,
    date_col: str | None,
    enable_structural_break_detection: bool,
    split_histograms_by_structure: bool,
    scale_segmented_histograms: bool,
    enabled_rules: dict[str, bool],
    rule_points: dict[str, int],
    rule_window_threshold: dict[str, dict[str, int]],
    mark_all_sequence_points: bool = True,
    show_info: bool = False,
) -> None:
    """Execute P chart build, limits, violations, and rendering."""
    chart_df = build_p_chart_df(
        df=df_work,
        inspected_col=inspected_col,
        defects_col=defects_col,
        date_col=date_col,
    )
    
    chart_df, limits = get_limits_with_optional_structural_breaks(
        chart_df=chart_df,
        chart_type="P",
        enable_structural_break_detection=enable_structural_break_detection,
    )
    
    # No secondary for P
    primary_violations, secondary_violations = detect_violations_for_chart(
        chart_df=chart_df,
        limits=limits,
        enabled_rules=enabled_rules,
        rule_points=rule_points,
        rule_window_threshold=rule_window_threshold,
        enable_secondary=False,
        mark_all_sequence_points=mark_all_sequence_points,
    )

    render_limit_summary(
        chart_df=chart_df,
        limits=limits,
        split_by_structure=split_histograms_by_structure,
        use_date_labels=(date_col is not None),
    )

    title = "P Chart"
    fig = plot_spc_chart(
        chart_df=chart_df,
        limits=limits,
        title=title,
        primary_violations=primary_violations,
        secondary_violations=secondary_violations,
        x_axis_mode="Subgroup" if date_col is None else "Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    render_histograms_section(
        chart_df=chart_df,
        limits=limits,
        chart_title=title,
        split_by_structure=split_histograms_by_structure,
        use_date_labels=(date_col is not None),
        scale_segmented_histograms=scale_segmented_histograms,
    )

    render_violations_section(primary_violations, secondary_violations)

# ============================================================
# Main App
# ============================================================
def main() -> None:
    """Run the Streamlit SPC application."""
    render_header()
    show_info = render_show_additional_information()
    enabled_rules, rule_points, rule_window_threshold, enable_secondary = render_spc_explainer(show_info)

    with st.expander("Setup", expanded=True):
        uploaded_file = render_sidebar_file_upload()
        df = load_uploaded_file(uploaded_file)

        if df is None:
            st.info("Upload a CSV or Excel file to get started.")
            return

        if show_info:
            st.success("Dataset loaded successfully.")
            render_data_preview(df)

        measurement_col, date_col, subgroup_col, inspected_col, defects_col = render_column_mapping(df)
        render_selected_columns_missing_notice(df, measurement_col, date_col, subgroup_col)
        null_treatment = render_null_treatment_option(show_info)
        fill_missing_dates_zero = render_missing_date_zero_option(show_info)
        
        if (measurement_col is None) and ((inspected_col is None) or (defects_col is None)):
            st.info("Please select a Measurement column to continue.")
            st.stop()

        df_work = clean_working_data(df, measurement_col, date_col, subgroup_col, inspected_col, defects_col, null_treatment)

        if df_work.empty:
            st.warning("No valid numeric measurement data remains after cleaning.")
            st.stop()

        evaluation = evaluate_chart_validity(df_work, measurement_col, subgroup_col, inspected_col, defects_col)
        if show_info:
            render_validity_messages(evaluation)

        if not evaluation.valid_options:
            st.warning("No valid SPC chart options based on the current selection. Adjust your column mapping or data.")
            st.stop()

        chosen_chart = st.radio(
            "Select one of the valid options:",
            options=evaluation.valid_options,
            index=0,
            horizontal=True,
        )

    # ---- Holiday calendar creation (only if date column provided)
    holiday_calendar_df = None
    if date_col and (chosen_chart == "I-MR"):
        min_dt, max_dt = get_valid_date_bounds(df_work, date_col)
        if min_dt is not None and max_dt is not None:
            holiday_calendar_df = build_holiday_calendar(
                start_dt=min_dt,
                end_dt=max_dt,
                country_code="ZA",   # You can make this a sidebar selectbox later
                subdiv=None,
                observed=True,
                easter_week_mode="iso_week",  # or "goodfri_to_mon"
            )
            save_holiday_calendar_to_session(holiday_calendar_df)

            # Optional: allow download from sidebar or main page
            if show_info:
                with st.sidebar.expander("Holiday calendar", expanded=False):
                    if pyholidays is None:
                        st.warning("Python package 'holidays' not installed. Public holiday detection is disabled.")
                    st.caption("Saved holiday flags for this dataset's date range.")
                    st.download_button(
                        "Download holiday calendar (CSV)",
                        data=holiday_calendar_df.to_csv(index=False).encode("utf-8"),
                        file_name="holiday_calendar.csv",
                        mime="text/csv",
                    )

    enable_structural_break_detection = render_structural_break_option(show_info)

    if show_info:
        split_histograms_by_structure = render_histogram_segment_option(
            enable_structural_break_detection=enable_structural_break_detection
        )

        scale_segmented_histograms = render_histogram_scaling_option(
            split_histograms_by_structure=split_histograms_by_structure
        )
    else:
        split_histograms_by_structure = True
        scale_segmented_histograms = True

    mark_all_sequence_points = render_sequence_rule_marker_option(show_info)

    if chosen_chart == "I-MR":
        run_imr_flow(
            df_work=df_work,
            measurement_col=measurement_col,
            date_col=date_col,
            enable_structural_break_detection=enable_structural_break_detection,
            split_histograms_by_structure=split_histograms_by_structure,
            scale_segmented_histograms=scale_segmented_histograms,
            enabled_rules=enabled_rules,
            rule_points=rule_points,
            rule_window_threshold=rule_window_threshold,
            enable_secondary=enable_secondary,
            mark_all_sequence_points=mark_all_sequence_points,
            fill_missing_dates_zero=fill_missing_dates_zero,
            show_info=show_info,
        )

    elif chosen_chart == "Xbar-R":
        if subgroup_col is None:
            st.error("Subgroup column is required for Xbar-R.")
            st.stop()
        run_xbar_r_flow(
            df_work=df_work,
            measurement_col=measurement_col,
            subgroup_col=subgroup_col,
            enable_structural_break_detection=enable_structural_break_detection,
            split_histograms_by_structure=split_histograms_by_structure,
            scale_segmented_histograms=scale_segmented_histograms,
            enabled_rules=enabled_rules,
            rule_points=rule_points,
            rule_window_threshold=rule_window_threshold,
            enable_secondary=enable_secondary,
            mark_all_sequence_points=mark_all_sequence_points,
            show_info=show_info,
        )

    elif chosen_chart == "Xbar-S":
        if subgroup_col is None:
            st.error("Subgroup column is required for Xbar-S.")
            st.stop()
        run_xbar_s_flow(
            df_work=df_work,
            measurement_col=measurement_col,
            subgroup_col=subgroup_col,
            enable_structural_break_detection=enable_structural_break_detection,
            split_histograms_by_structure=split_histograms_by_structure,
            scale_segmented_histograms=scale_segmented_histograms,
            enabled_rules=enabled_rules,
            rule_points=rule_points,
            rule_window_threshold=rule_window_threshold,
            enable_secondary=enable_secondary,
            mark_all_sequence_points=mark_all_sequence_points,
            show_info=show_info,
        )
    elif chosen_chart == "P":
        if inspected_col is None or defects_col is None:
            st.error("P chart requires 'Inspected' and 'Defects' columns.")
            st.stop()
        run_p_flow(
            df_work=df_work,
            inspected_col=inspected_col,
            defects_col=defects_col,
            date_col=date_col,  # optional; works if your p data has real dates
            enable_structural_break_detection=enable_structural_break_detection,
            split_histograms_by_structure=split_histograms_by_structure,
            scale_segmented_histograms=scale_segmented_histograms,
            enabled_rules=enabled_rules,
            rule_points=rule_points,
            rule_window_threshold=rule_window_threshold,
            mark_all_sequence_points=mark_all_sequence_points,
            show_info=show_info,
        )
    else:
        st.error(f"Unsupported chart type: {chart_type}")



if __name__ == "__main__":
    main()
