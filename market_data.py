"""Live Yahoo Finance chart + FRED series, with a bundled parquet fallback."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
PANEL_PATH = DATA_DIR / "dxy_panel.parquet"
METADATA_PATH = DATA_DIR / "warehouse_metadata.json"

YAHOO_UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) QuantIfy/1.0"
}

# ICE US Dollar Index and the six published legs.
YAHOO_SERIES: dict[str, str] = {
    "dxy": "DX-Y.NYB",
    "eurusd": "EURUSD=X",
    "usdjpy": "USDJPY=X",
    "gbpusd": "GBPUSD=X",
    "usdcad": "USDCAD=X",
    "usdchf": "USDCHF=X",
    "usdsek": "USDSEK=X",
    "tnx": "^TNX",
}

FRED_SERIES: dict[str, str] = {
    "dgs2": "DGS2",
    "dgs10": "DGS10",
    "dff": "DFF",
}

LOOKBACK_YEARS = 3
CPI_PRINT = "2026-10-14"
PPI_PRINT = "2026-10-15"


def default_window_end() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def default_window_start(end: str | None = None, *, years: int = LOOKBACK_YEARS) -> str:
    end_dt = datetime.fromisoformat(end or default_window_end()).date()
    return (end_dt - timedelta(days=int(365.25 * years))).isoformat()


def _yahoo_chart_url(symbol: str, start: str, end: str) -> str:
    start_ts = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(
        (datetime.fromisoformat(end) + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp()
    )
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(symbol, safe='')}?period1={start_ts}&period2={end_ts}&interval=1d"
    )


def fetch_yahoo_ohlc(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLC from Yahoo's public chart API (no yfinance client)."""
    url = _yahoo_chart_url(symbol, start, end)
    resp = requests.get(url, headers=YAHOO_UA, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        error = (payload.get("chart") or {}).get("error") or payload
        raise RuntimeError(f"Yahoo chart empty for {symbol}: {error}")
    block = results[0]
    stamps = block.get("timestamp") or []
    quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(stamps, unit="s", utc=True).tz_convert(None).normalize(),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
        }
    )
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"])
    frame = frame[frame["close"] > 0]
    if frame.empty:
        raise RuntimeError(f"Yahoo chart had no closes for {symbol}.")
    return frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_fred_series(series_id: str, start: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw = pd.read_csv(StringIO(resp.text))
    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = raw.columns[0]
    value_col = raw.columns[1]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]),
            "value": pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["value"])
    if frame.empty:
        raise RuntimeError(f"FRED {series_id} returned no numeric rows.")
    return frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_live_panel(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    end = end or default_window_end()
    start = start or default_window_start(end)
    parts: dict[str, pd.Series] = {}
    errors: list[str] = []

    for name, symbol in YAHOO_SERIES.items():
        try:
            ohlc = fetch_yahoo_ohlc(symbol, start, end)
            parts[name] = ohlc.set_index("date")["close"].rename(name)
            if name == "dxy":
                parts["dxy_high"] = ohlc.set_index("date")["high"].rename("dxy_high")
                parts["dxy_low"] = ohlc.set_index("date")["low"].rename("dxy_low")
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    for name, series_id in FRED_SERIES.items():
        try:
            fred = fetch_fred_series(series_id, start)
            parts[name] = fred.set_index("date")["value"].rename(name)
        except Exception as exc:
            errors.append(f"FRED {series_id}: {exc}")

    required = {"dxy", "eurusd", "usdjpy", "gbpusd", "usdcad", "usdchf", "usdsek"}
    missing = sorted(required - set(parts))
    if missing:
        detail = "; ".join(errors) or "no series returned"
        raise RuntimeError(f"Could not load required FX legs {missing}. {detail}")

    panel = pd.concat(parts.values(), axis=1).sort_index()
    panel.index.name = "date"
    panel = panel.loc[panel["dxy"].notna()].copy()
    fill_cols = [c for c in panel.columns if c != "dxy"]
    panel[fill_cols] = panel[fill_cols].ffill()
    need = [c for c in ("eurusd", "usdjpy", "gbpusd", "usdcad", "usdchf", "usdsek") if c in panel.columns]
    panel = panel.dropna(subset=need)
    panel = panel.reset_index()
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None).dt.normalize()
    return panel


def write_warehouse(panel: pd.DataFrame, *, source: str, start: str, end: str) -> dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(PANEL_PATH, index=False)
    meta: dict[str, object] = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "window_start": start,
        "window_end": end,
        "n_days": int(len(out)),
        "panel_start": pd.Timestamp(out["date"].iloc[0]).strftime("%Y-%m-%d"),
        "panel_end": pd.Timestamp(out["date"].iloc[-1]).strftime("%Y-%m-%d"),
        "last_dxy": float(out["dxy"].iloc[-1]),
        "cpi_print": CPI_PRINT,
        "ppi_print": PPI_PRINT,
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_warehouse_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"No bundled warehouse at {PANEL_PATH}. Run python3 download_data.py first."
        )
    frame = pd.read_parquet(PANEL_PATH)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    return frame.sort_values("date").reset_index(drop=True)


def load_warehouse_metadata() -> dict[str, object]:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_panel(*, refresh: bool = True) -> tuple[pd.DataFrame, dict[str, object]]:
    """Prefer a live pull; fall back to the bundled parquet if the network fails."""
    bundled_meta = load_warehouse_metadata()
    if refresh:
        try:
            live = fetch_live_panel()
            meta = {
                **bundled_meta,
                "source": "live Yahoo chart + FRED",
                "panel_start": pd.Timestamp(live["date"].iloc[0]).strftime("%Y-%m-%d"),
                "panel_end": pd.Timestamp(live["date"].iloc[-1]).strftime("%Y-%m-%d"),
                "n_days": int(len(live)),
                "last_dxy": float(live["dxy"].iloc[-1]),
                "live": True,
            }
            return live, meta
        except Exception as exc:
            if PANEL_PATH.exists():
                bundled_meta = {
                    **bundled_meta,
                    "live": False,
                    "live_error": str(exc),
                }
                return load_warehouse_panel(), bundled_meta
            raise
    return load_warehouse_panel(), {**bundled_meta, "live": False}
