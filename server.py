"""
Rujwasco Water Distribution MCP Server

Exposes the notebook's Pandas analyses as read-only MCP tools over stdio.
The server reads the supplied CSV; it never modifies the source dataset.

IMPORTANT: with the stdio transport, stdout is the protocol channel.
Never use print() here - log to stderr only.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | rujwasco-mcp | %(message)s",
    stream=sys.stderr,
    force=True,
)
logger = logging.getLogger("rujwasco-mcp")

mcp = FastMCP("rujwasco-distribution")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(
    os.getenv("RUJWASCO_DATA_PATH", BASE_DIR / "data" / "water_distribution.csv")
)

REQUIRED_COLUMNS = {
    "date",
    "zone",
    "water_input_m3",
    "water_billed_m3",
    "average_pressure_bar",
    "supply_hours",
    "burst_incidents",
    "repair_hours",
    "active_connections",
}

NUMERIC_COLUMNS = sorted(REQUIRED_COLUMNS - {"date", "zone"})

# Metrics that are rates/averages should be averaged, not summed, across rows per date.
TREND_METRICS = {
    "water_input_m3": "sum",
    "water_billed_m3": "sum",
    "average_pressure_bar": "mean",
    "supply_hours": "mean",
    "burst_incidents": "sum",
    "repair_hours": "sum",
}


# =========================================================
# DATA ACCESS
# =========================================================

_cache: dict[str, Any] = {"mtime": None, "frame": None}


def _read_dataset() -> pd.DataFrame:
    """Read and validate the CSV. Re-reads only when the file changes on disk."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Set RUJWASCO_DATA_PATH to the CSV location."
        )

    mtime = DATA_PATH.stat().st_mtime
    if _cache["frame"] is not None and _cache["mtime"] == mtime:
        return _cache["frame"]

    frame = pd.read_csv(DATA_PATH)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("The dataset contains invalid dates.")

    frame["zone"] = frame["zone"].astype("string").str.strip()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    _cache.update(mtime=mtime, frame=frame)
    logger.info("Loaded dataset: %s rows from %s", len(frame), DATA_PATH)
    return frame


def _parse_date(value: str | None, label: str) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.to_datetime(value, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {label} '{value}'. Use YYYY-MM-DD.") from exc


def load_data(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Return the dataset, optionally filtered by inclusive date boundaries."""
    data = _read_dataset()

    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must not be after end_date.")

    if start is not None:
        data = data[data["date"] >= start]
    if end is not None:
        data = data[data["date"] <= end]

    if data.empty:
        raise ValueError("No records found for the requested date range.")

    return data


def filter_zone(data: pd.DataFrame, zone: str | None) -> pd.DataFrame:
    if zone is None:
        return data
    selected = data[data["zone"].str.casefold() == zone.strip().casefold()]
    if selected.empty:
        available = sorted(data["zone"].dropna().unique().tolist())
        raise ValueError(f"Unknown zone '{zone}'. Available zones: {available}")
    return selected


def period_of(data: pd.DataFrame) -> dict[str, str]:
    return {
        "start": data["date"].min().date().isoformat(),
        "end": data["date"].max().date().isoformat(),
    }


def safe_pct(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator * 100, 2) if denominator else None


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records."""
    output = frame.to_dict(orient="records")
    for row in output:
        for key, value in list(row.items()):
            if value is None or value is pd.NA:
                row[key] = None
            elif hasattr(value, "isoformat"):
                row[key] = value.isoformat()
            elif hasattr(value, "item"):
                py_value = value.item()
                row[key] = None if pd.isna(py_value) else py_value
            elif isinstance(value, float) and pd.isna(value):
                row[key] = None
    return output


# =========================================================
# TOOLS
# =========================================================

@mcp.tool()
def get_dataset_overview() -> dict:
    """Return row count, date coverage, zones, columns and data-quality checks."""
    data = load_data()
    return {
        "rows": int(len(data)),
        "columns": list(data.columns),
        "period": period_of(data),
        "zones": sorted(data["zone"].dropna().unique().tolist()),
        "missing_values": {k: int(v) for k, v in data.isna().sum().items()},
        "duplicate_rows": int(data.duplicated().sum()),
    }


@mcp.tool()
def get_water_input_by_zone(start_date: str | None = None, end_date: str | None = None) -> dict:
    """Sum water entering each zone and calculate the daily mean for the selected period."""
    data = load_data(start_date, end_date)
    result = (
        data.groupby("zone")
        .agg(
            total_input_m3=("water_input_m3", "sum"),
            average_daily_input_m3=("water_input_m3", "mean"),
            records=("water_input_m3", "size"),
        )
        .sort_values("total_input_m3", ascending=False)
        .reset_index()
    )
    return {
        "period": period_of(data),
        "total_system_input_m3": round(float(data["water_input_m3"].sum()), 2),
        "by_zone": records(result.round(2)),
    }


@mcp.tool()
def get_supply_hours(
    zone: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Compare daily supply hours by zone, including mean, minimum, maximum and standard deviation."""
    data = filter_zone(load_data(start_date, end_date), zone)
    result = (
        data.groupby("zone")["supply_hours"]
        .agg(average_hours="mean", minimum_hours="min", maximum_hours="max", std_hours="std")
        .reset_index()
        .round(3)
    )
    return {"period": period_of(data), "by_zone": records(result)}


@mcp.tool()
def get_pipe_burst_analysis(
    zone: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Summarize burst counts, total repair hours and mean repair hours by zone."""
    data = filter_zone(load_data(start_date, end_date), zone)
    result = (
        data.groupby("zone")
        .agg(
            total_bursts=("burst_incidents", "sum"),
            total_repair_hours=("repair_hours", "sum"),
            average_daily_repair_hours=("repair_hours", "mean"),
        )
        .sort_values("total_bursts", ascending=False)
        .reset_index()
        .round(2)
    )
    return {
        "period": period_of(data),
        "system_total_bursts": int(data["burst_incidents"].sum()),
        "system_total_repair_hours": round(float(data["repair_hours"].sum()), 2),
        "by_zone": records(result),
    }


@mcp.tool()
def get_pressure_analysis(
    zone: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Return average, minimum, maximum and standard deviation of pressure by zone."""
    data = filter_zone(load_data(start_date, end_date), zone)
    result = (
        data.groupby("zone")["average_pressure_bar"]
        .agg(
            average_pressure_bar="mean",
            minimum_pressure_bar="min",
            maximum_pressure_bar="max",
            std_pressure_bar="std",
        )
        .reset_index()
        .round(3)
    )
    return {"period": period_of(data), "by_zone": records(result)}


@mcp.tool()
def get_pressure_burst_correlation(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Calculate Pearson correlation between daily average pressure and daily burst incidents."""
    data = load_data(start_date, end_date)
    correlation = data["average_pressure_bar"].corr(data["burst_incidents"])
    return {
        "period": period_of(data),
        "pearson_correlation": None if pd.isna(correlation) else round(float(correlation), 4),
        "observations": int(len(data)),
        "interpretation_warning": (
            "Correlation does not establish causation. Pipe condition, operating practices, "
            "pressure transients and other factors may influence bursts."
        ),
    }


@mcp.tool()
def get_water_loss_analysis(
    zone: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Calculate the notebook's apparent unaccounted-for volume and percentage.

    UAV = water input - billed volume.
    UAV% = total UAV / total input * 100.
    This is not a verified NRW water balance.
    """
    data = filter_zone(load_data(start_date, end_date), zone)

    grouped = data.groupby("zone").agg(
        total_input_m3=("water_input_m3", "sum"),
        total_billed_m3=("water_billed_m3", "sum"),
    )
    grouped["unaccounted_volume_m3"] = grouped["total_input_m3"] - grouped["total_billed_m3"]
    grouped["unaccounted_percentage"] = (
        grouped["unaccounted_volume_m3"]
        / grouped["total_input_m3"].where(grouped["total_input_m3"] != 0)
        * 100
    )
    grouped = grouped.reset_index().round(2)

    total_input = float(data["water_input_m3"].sum())
    total_billed = float(data["water_billed_m3"].sum())
    total_unaccounted = total_input - total_billed

    return {
        "period": period_of(data),
        "system": {
            "total_input_m3": round(total_input, 2),
            "total_billed_m3": round(total_billed, 2),
            "unaccounted_volume_m3": round(total_unaccounted, 2),
            "unaccounted_percentage": safe_pct(total_unaccounted, total_input),
        },
        "by_zone": records(grouped),
        "methodology_note": (
            "This is the dataset's apparent unaccounted-for volume indicator, not a verified "
            "Non-Revenue Water calculation. A proper NRW water balance must account for "
            "authorised unbilled consumption and aligned measurement periods."
        ),
    }


@mcp.tool()
def compare_zones(
    zones: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Compare selected zones across input, supply, bursts, pressure and apparent water loss."""
    requested = {z.strip().casefold(): z.strip() for z in zones if z and z.strip()}
    if len(requested) < 2:
        raise ValueError("Provide at least two distinct zone names.")

    data = load_data(start_date, end_date)
    available = {z.casefold(): z for z in data["zone"].dropna().unique()}

    unknown = [original for key, original in requested.items() if key not in available]
    if unknown:
        raise ValueError(
            f"Zones not found: {unknown}. Available zones: {sorted(available.values())}"
        )

    selected = data[data["zone"].str.casefold().isin(requested.keys())]

    grouped = selected.groupby("zone").agg(
        water_input_m3=("water_input_m3", "sum"),
        billed_m3=("water_billed_m3", "sum"),
        average_supply_hours=("supply_hours", "mean"),
        total_bursts=("burst_incidents", "sum"),
        repair_hours=("repair_hours", "sum"),
        average_pressure_bar=("average_pressure_bar", "mean"),
    )
    grouped["unaccounted_volume_m3"] = grouped["water_input_m3"] - grouped["billed_m3"]
    grouped["unaccounted_percentage"] = (
        grouped["unaccounted_volume_m3"]
        / grouped["water_input_m3"].where(grouped["water_input_m3"] != 0)
        * 100
    )
    return {
        "period": period_of(selected),
        "zones": records(grouped.round(2).reset_index()),
    }


@mcp.tool()
def get_daily_trends(
    zone: str,
    metric: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Return daily values for one zone. Metric must be water_input_m3, water_billed_m3,
    average_pressure_bar, supply_hours, burst_incidents or repair_hours."""
    if metric not in TREND_METRICS:
        raise ValueError(f"Unsupported metric. Choose one of: {sorted(TREND_METRICS)}")

    data = filter_zone(load_data(start_date, end_date), zone)
    result = (
        data.groupby("date")[metric]
        .agg(TREND_METRICS[metric])
        .reset_index()
        .sort_values("date")
    )
    return {
        "zone": data["zone"].iloc[0],
        "metric": metric,
        "aggregation": TREND_METRICS[metric],
        "period": period_of(data),
        "daily_values": records(result.round(3)),
    }


if __name__ == "__main__":
    # MCP stdio transport: launched as a subprocess by main.py (or Claude Desktop).
    mcp.run()