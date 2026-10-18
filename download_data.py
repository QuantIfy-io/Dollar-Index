"""Seed the bundled DXY / FX / yield warehouse."""
from __future__ import annotations

from market_data import (
    default_window_end,
    default_window_start,
    fetch_live_panel,
    write_warehouse,
)


def main() -> None:
    end = default_window_end()
    start = default_window_start(end)
    panel = fetch_live_panel(start, end)
    meta = write_warehouse(panel, source="Yahoo chart + FRED", start=start, end=end)
    print(
        f"Wrote {meta['n_days']} sessions {meta['panel_start']} → {meta['panel_end']} "
        f"last DXY {meta['last_dxy']:.2f}"
    )


if __name__ == "__main__":
    main()
