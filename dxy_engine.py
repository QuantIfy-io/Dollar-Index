"""ICE dollar-index reconstruction, basket attribution, and the post-CPI read."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

# ICE US Dollar Index published weights and scale.
# EUR and GBP enter as dollars-per-unit (negative exponents).
# JPY, CAD, SEK, CHF enter as foreign-units-per-dollar (positive exponents).
DXY_SCALE = 50.14348112
ICE_WEIGHTS: dict[str, float] = {
    "eurusd": -0.576,
    "usdjpy": 0.136,
    "gbpusd": -0.119,
    "usdcad": 0.091,
    "usdsek": 0.042,
    "usdchf": 0.036,
}
ICE_LABELS: dict[str, str] = {
    "eurusd": "EUR",
    "usdjpy": "JPY",
    "gbpusd": "GBP",
    "usdcad": "CAD",
    "usdsek": "SEK",
    "usdchf": "CHF",
}
CPI_PRINT = pd.Timestamp("2026-10-14")
PPI_PRINT = pd.Timestamp("2026-10-15")


def _as_timestamp(value: date | datetime | pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def slice_panel(
    panel: pd.DataFrame,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
) -> pd.DataFrame:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if start is not None:
        frame = frame.loc[frame["date"] >= _as_timestamp(start)]
    if end is not None:
        frame = frame.loc[frame["date"] <= _as_timestamp(end)]
    if frame.empty:
        raise ValueError("No DXY sessions in the selected window.")
    return frame.sort_values("date").reset_index(drop=True)


def reconstruct_dxy(panel: pd.DataFrame) -> pd.Series:
    """Rebuild the ICE index from the six legs to check tracking."""
    log_sum = sum(weight * np.log(panel[col].astype(float)) for col, weight in ICE_WEIGHTS.items())
    return DXY_SCALE * np.exp(log_sum)


def add_derived(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    frame["dxy_hat"] = reconstruct_dxy(frame)
    frame["dxy_return"] = frame["dxy"].pct_change()
    if "dgs2" in frame.columns:
        frame["dgs2_change"] = frame["dgs2"].diff()
    return frame


def _pct(start: float, end: float) -> float:
    if start == 0 or not np.isfinite(start):
        return float("nan")
    return 100.0 * (end / start - 1.0)


def summary_metrics(panel: pd.DataFrame) -> dict[str, float | str]:
    frame = add_derived(panel).dropna(subset=["dxy"])
    last = frame.iloc[-1]
    first = frame.iloc[0]
    lookback_18m = last["date"] - pd.Timedelta(days=548)
    trail = frame.loc[frame["date"] >= lookback_18m]
    high_col = "dxy_high" if "dxy_high" in trail.columns and trail["dxy_high"].notna().any() else "dxy"
    high_idx = trail[high_col].idxmax()
    high_row = trail.loc[high_idx]

    ytd = frame.loc[frame["date"] >= pd.Timestamp(last["date"].year, 1, 1)]
    year_ago = last["date"] - pd.Timedelta(days=365)
    yoy = frame.loc[frame["date"] >= year_ago]

    cpi_rows = frame.loc[frame["date"] == CPI_PRINT]
    pre_cpi = frame.loc[frame["date"] < CPI_PRINT]
    after_cpi = frame.loc[frame["date"] > CPI_PRINT]

    dgs2 = float(last["dgs2"]) if "dgs2" in last.index and pd.notna(last.get("dgs2")) else float("nan")
    dgs10 = float(last["dgs10"]) if "dgs10" in last.index and pd.notna(last.get("dgs10")) else float("nan")

    tracking = float((frame["dxy"] / frame["dxy_hat"] - 1.0).abs().median() * 100.0)

    corr = float("nan")
    sub = frame.dropna(subset=["dxy_return", "dgs2_change"]) if "dgs2_change" in frame.columns else frame
    if "dgs2_change" in sub.columns and len(sub) > 40:
        corr = float(sub["dxy_return"].corr(sub["dgs2_change"]))

    return {
        "start": pd.Timestamp(first["date"]).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(last["date"]).strftime("%Y-%m-%d"),
        "dxy_end": float(last["dxy"]),
        "dxy_start": float(first["dxy"]),
        "dxy_change_pct": _pct(float(first["dxy"]), float(last["dxy"])),
        "dxy_ytd_pct": _pct(float(ytd["dxy"].iloc[0]), float(last["dxy"])) if len(ytd) else float("nan"),
        "dxy_yoy_pct": _pct(float(yoy["dxy"].iloc[0]), float(last["dxy"])) if len(yoy) else float("nan"),
        "high_18m": float(high_row[high_col]),
        "high_18m_date": pd.Timestamp(high_row["date"]).strftime("%Y-%m-%d"),
        "gap_to_high_pct": _pct(float(high_row[high_col]), float(last["dxy"])),
        "eurusd_end": float(last["eurusd"]),
        "usdjpy_end": float(last["usdjpy"]),
        "dgs2_end": dgs2,
        "dgs10_end": dgs10,
        "dff_end": float(last["dff"]) if "dff" in last.index and pd.notna(last.get("dff")) else float("nan"),
        "cpi_dxy": float(cpi_rows["dxy"].iloc[0]) if len(cpi_rows) else float("nan"),
        "cpi_dxy_high": float(cpi_rows["dxy_high"].iloc[0])
        if len(cpi_rows) and "dxy_high" in cpi_rows.columns
        else float("nan"),
        "pre_cpi_dxy": float(pre_cpi["dxy"].iloc[-1]) if len(pre_cpi) else float("nan"),
        "post_cpi_dxy": float(after_cpi["dxy"].iloc[-1]) if len(after_cpi) else float("nan"),
        "tracking_error_pct": tracking,
        "corr_dxy_dgs2": corr,
        "n_days": int(len(frame)),
    }


def basket_attribution(panel: pd.DataFrame) -> pd.DataFrame:
    """Approximate DXY point contribution of each ICE leg over the window."""
    frame = panel.dropna(subset=list(ICE_WEIGHTS)).sort_values("date")
    if len(frame) < 2:
        raise ValueError("Need at least two sessions to attribute the basket.")
    start = frame.iloc[0]
    end = frame.iloc[-1]
    dxy0 = float(start["dxy"])
    rows = []
    total = 0.0
    for col, weight in ICE_WEIGHTS.items():
        log_move = float(np.log(end[col] / start[col]))
        points = dxy0 * weight * log_move
        total += points
        rows.append(
            {
                "leg": ICE_LABELS[col],
                "column": col,
                "ice_weight": weight,
                "start": float(start[col]),
                "end": float(end[col]),
                "move_pct": _pct(float(start[col]), float(end[col])),
                "dxy_points": points,
            }
        )
    actual = float(end["dxy"] - start["dxy"])
    rows.append(
        {
            "leg": "Residual",
            "column": "residual",
            "ice_weight": 0.0,
            "start": float("nan"),
            "end": float("nan"),
            "move_pct": float("nan"),
            "dxy_points": actual - total,
        }
    )
    return pd.DataFrame(rows)


def event_window(panel: pd.DataFrame, *, pre: int = 7, post: int = 5) -> pd.DataFrame:
    frame = add_derived(panel)
    loc = frame.index[frame["date"] == CPI_PRINT]
    if len(loc) == 0:
        # nearest session on or after the print
        later = frame.loc[frame["date"] >= CPI_PRINT]
        if later.empty:
            raise ValueError("CPI print date is after the end of the panel.")
        center = later.index[0]
    else:
        center = loc[0]
    lo = max(0, int(center) - pre)
    hi = min(len(frame) - 1, int(center) + post)
    window = frame.iloc[lo : hi + 1].copy()
    base = float(window["dxy"].iloc[0])
    window["dxy_rebased"] = 100.0 * window["dxy"] / base
    if "dgs2" in window.columns:
        window["dgs2_from_start_bp"] = 100.0 * (window["dgs2"] - window["dgs2"].iloc[0])
    return window.reset_index(drop=True)


def rolling_rate_link(panel: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    frame = add_derived(panel).dropna(subset=["dxy_return"])
    if "dgs2_change" not in frame.columns:
        return pd.DataFrame(columns=["date", "corr", "beta"])
    work = frame.dropna(subset=["dgs2_change"]).copy()
    work["corr"] = work["dxy_return"].rolling(window).corr(work["dgs2_change"])
    cov = work["dxy_return"].rolling(window).cov(work["dgs2_change"])
    var = work["dgs2_change"].rolling(window).var()
    work["beta"] = cov / var.replace(0, np.nan)
    return work[["date", "corr", "beta"]].dropna().reset_index(drop=True)


def study_verdict(metrics: dict[str, float | str]) -> tuple[str, str]:
    ytd = metrics.get("dxy_ytd_pct")
    gap = metrics.get("gap_to_high_pct")
    dgs2 = metrics.get("dgs2_end")
    high_date = metrics.get("high_18m_date")
    ytd_txt = f"{ytd:+.1f}%" if isinstance(ytd, float) and np.isfinite(ytd) else "n/a"
    gap_txt = f"{gap:+.2f}%" if isinstance(gap, float) and np.isfinite(gap) else "n/a"
    dgs2_txt = f"{dgs2:.2f}%" if isinstance(dgs2, float) and np.isfinite(dgs2) else "n/a"

    if isinstance(ytd, float) and ytd > 0 and isinstance(gap, float) and gap > -2.0:
        title = "The dollar is not in a decline."
        body = (
            f"DXY is {ytd_txt} year to date and {gap_txt} from its 18-month high "
            f"({high_date}). The US 2-year still yields {dgs2_txt}. A faded CPI spike "
            "is digestion at the highs, not the start of a dollar bear market."
        )
        return title, body
    title = "Watch the path, not the one-day spike."
    body = (
        f"DXY is {ytd_txt} in the selected window, {gap_txt} from the 18-month high. "
        f"US 2-year yield {dgs2_txt}. The index follows relative rate expectations — "
        "especially versus the euro, 57.6% of ICE DXY."
    )
    return title, body
