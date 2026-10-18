"""Streamlit: why the ICE dollar index is not in a decline after the 14 Oct CPI print."""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dxy_engine import (
    CPI_PRINT,
    PPI_PRINT,
    add_derived,
    basket_attribution,
    event_window,
    rolling_rate_link,
    slice_panel,
    study_verdict,
    summary_metrics,
)
from market_data import load_panel, load_warehouse_metadata

st.set_page_config(
    page_title="Why isn't the dollar declining?",
    page_icon=":material/currency_exchange:",
    layout="wide",
)

AMBER = "#e07a3d"
BLUE = "#4c8bf5"
TEAL = "#38a8a4"
RED = "#d45d79"
GOLD = "#c9a227"
MUTED = "#8a93a3"
LEG_COLORS = {
    "EUR": AMBER,
    "JPY": BLUE,
    "GBP": TEAL,
    "CAD": GOLD,
    "SEK": "#9b7ed9",
    "CHF": RED,
    "Residual": MUTED,
}

PLOTLY_LEGEND = dict(
    orientation="h",
    yanchor="top",
    y=-0.32,
    xanchor="center",
    x=0.5,
    bgcolor="rgba(0,0,0,0)",
)


def _style(
    fig: go.Figure,
    *,
    title: str,
    height: int,
    ytitle: str,
    xtitle: str = "Date",
    showlegend: bool = True,
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15)),
        showlegend=showlegend,
        legend=PLOTLY_LEGEND if showlegend else dict(),
        margin=dict(l=56, r=16, t=48, b=96 if showlegend else 48),
        height=height,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        yaxis_title=ytitle,
        xaxis_title=xtitle,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def _vline(fig: go.Figure, ts: pd.Timestamp, label: str) -> None:
    fig.add_vline(
        x=ts.to_pydatetime(),
        line_width=1,
        line_dash="dot",
        line_color=RED,
        annotation_text=label,
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color=RED,
    )


def _formulas() -> None:
    with st.expander("ICE dollar index formula", expanded=False):
        st.markdown(
            "The ICE US Dollar Index is a geometric basket. Euro is **57.6%** of the weight. "
            "That is why a dollar 'mystery' is often just EUR/USD plus US front-end yields."
        )
        st.latex(r"DXY = 50.14348112 \times EURUSD^{-0.576} \times USDJPY^{0.136}")
        st.latex(
            r"\phantom{DXY ={}} \times GBPUSD^{-0.119} \times USDCAD^{0.091} "
            r"\times USDSEK^{0.042} \times USDCHF^{0.036}"
        )
        st.markdown(
            r"""
EUR/USD and GBP/USD enter as dollars per unit (negative exponents): a weaker euro
raises $DXY$. JPY, CAD, SEK, and CHF enter as foreign units per dollar.

Log-point attribution over a window, starting from $DXY_0$:

$$
\Delta DXY \approx DXY_0 \sum_i w_i \,\Delta \log S_i
$$

$w_i$ is the ICE exponent. Residual is published $DXY$ minus the reconstructed move
(fixing, timing, and the constant scale).
            """
        )


@st.cache_data(ttl=3600, show_spinner="Loading dollar, FX, and Treasury yields…")
def _load(refresh: bool) -> tuple[pd.DataFrame, dict]:
    return load_panel(refresh=refresh)


st.title("Why isn't the dollar index declining?")
st.markdown(
    "Schwab's 15 Oct 2026 open asked what hot **September CPI** would do to the Fed path. "
    "The ICE dollar index tagged an **18-month high** on the print, gave the spike back, "
    "and is still **up on the year**. This app asks whether that is a dollar decline — "
    "or a rate-differential index sitting near the highs."
)
_formulas()

refresh = st.toggle("Refresh live Yahoo + FRED", value=False)
try:
    panel, meta = _load(refresh)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if meta.get("live_error"):
    st.caption(
        "Live Yahoo/FRED refresh was blocked. Using the bundled warehouse "
        f"through **{meta.get('panel_end', 'the last saved session')}**."
    )

d_min = pd.Timestamp(panel["date"].min()).date()
d_max = pd.Timestamp(panel["date"].max()).date()
default_from = date(d_max.year - 1, d_max.month, d_max.day)
if default_from < d_min:
    default_from = d_min

c_from, c_to = st.columns(2)
date_from = c_from.date_input("Date from", value=default_from, min_value=d_min, max_value=d_max)
date_to = c_to.date_input("Date to", value=d_max, min_value=d_min, max_value=d_max)
if date_from > date_to:
    st.error("Date from must be on or before date to.")
    st.stop()

try:
    view = slice_panel(panel, date_from, date_to)
    view = add_derived(view)
    metrics = summary_metrics(view)
    attrib = basket_attribution(view)
    events = event_window(panel)
    link = rolling_rate_link(view)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

verdict_title, verdict_body = study_verdict(metrics)
source = meta.get("source", "bundled")
st.caption(
    f"{source} · **{metrics['start']}** → **{metrics['end']}** · "
    f"{metrics['n_days']} sessions · reconstructed DXY median gap "
    f"{metrics['tracking_error_pct']:.2f}%"
)

st.info(f"**{verdict_title}** {verdict_body}")

row = st.columns(6)
row[0].metric("DXY", f"{metrics['dxy_end']:.2f}", f"{metrics['dxy_change_pct']:+.1f}% window")
row[1].metric("Year to date", f"{metrics['dxy_ytd_pct']:+.1f}%")
row[2].metric(
    "18-month high",
    f"{metrics['high_18m']:.2f}",
    f"{metrics['gap_to_high_pct']:+.2f}% vs last · {metrics['high_18m_date']}",
)
row[3].metric(
    "US 2-year yield",
    f"{metrics['dgs2_end']:.2f}%" if pd.notna(metrics["dgs2_end"]) else "n/a",
)
row[4].metric(
    "US 10-year yield",
    f"{metrics['dgs10_end']:.2f}%" if pd.notna(metrics["dgs10_end"]) else "n/a",
)
row[5].metric("EUR/USD", f"{metrics['eurusd_end']:.4f}")

st.markdown(
    f"**14 Oct CPI session:** DXY high **{metrics['cpi_dxy_high']:.2f}** "
    f"(18-month extreme), close **{metrics['cpi_dxy']:.2f}**. "
    f"Prior close **{metrics['pre_cpi_dxy']:.2f}**. Last after the print: "
    f"**{metrics['post_cpi_dxy']:.2f}**."
)

fig_dxy = go.Figure()
fig_dxy.add_trace(
    go.Scatter(
        x=view["date"],
        y=view["dxy"],
        name="ICE DXY",
        line=dict(color=AMBER, width=2.4),
        hovertemplate="%{y:.2f}<extra>DXY</extra>",
    )
)
fig_dxy.add_trace(
    go.Scatter(
        x=view["date"],
        y=view["dxy_hat"],
        name="Reconstructed from 6 legs",
        line=dict(color=BLUE, width=1.4, dash="dot"),
        hovertemplate="%{y:.2f}<extra>DXY hat</extra>",
    )
)
_vline(fig_dxy, CPI_PRINT, "CPI")
if PPI_PRINT >= pd.Timestamp(date_from) and PPI_PRINT <= pd.Timestamp(date_to):
    _vline(fig_dxy, PPI_PRINT, "PPI")
st.plotly_chart(
    _style(fig_dxy, title="ICE dollar index vs reconstructed basket", height=420, ytitle="Index"),
    width="stretch",
)

left, right = st.columns(2)

fig_y = go.Figure()
if "dgs2" in view.columns:
    fig_y.add_trace(
        go.Scatter(
            x=view["date"],
            y=view["dgs2"],
            name="US 2-year",
            line=dict(color=TEAL, width=2.2),
            hovertemplate="%{y:.2f}%<extra>2y</extra>",
        )
    )
if "dgs10" in view.columns:
    fig_y.add_trace(
        go.Scatter(
            x=view["date"],
            y=view["dgs10"],
            name="US 10-year",
            line=dict(color=GOLD, width=2.0),
            hovertemplate="%{y:.2f}%<extra>10y</extra>",
        )
    )
_vline(fig_y, CPI_PRINT, "CPI")
left.plotly_chart(
    _style(fig_y, title="Treasury yields — the dollar's usual bid", height=380, ytitle="Percent"),
    width="stretch",
)

fig_fx = go.Figure()
fig_fx.add_trace(
    go.Scatter(
        x=view["date"],
        y=view["eurusd"],
        name="EUR/USD",
        line=dict(color=BLUE, width=2.2),
        hovertemplate="%{y:.4f}<extra>EURUSD</extra>",
    )
)
_vline(fig_fx, CPI_PRINT, "CPI")
right.plotly_chart(
    _style(
        fig_fx,
        title="Euro — 57.6% of ICE DXY",
        height=380,
        ytitle="Dollars per euro",
        showlegend=False,
    ),
    width="stretch",
)

st.subheader("What moved DXY in this window")
st.caption(
    "Log-point attribution from the first session to the last. A weaker euro is a positive "
    "DXY contribution. Residual is published DXY minus the six-leg reconstruction."
)
ordered = attrib.sort_values("dxy_points")
fig_bar = go.Figure()
fig_bar.add_trace(
    go.Bar(
        y=ordered["leg"],
        x=ordered["dxy_points"],
        orientation="h",
        marker_color=[LEG_COLORS.get(leg, MUTED) for leg in ordered["leg"]],
        name="DXY points",
        hovertemplate="%{y}: %{x:+.2f} pts<extra></extra>",
    )
)
fig_bar.add_vline(x=0, line_width=1, line_color="rgba(128,128,128,0.5)")
st.plotly_chart(
    _style(
        fig_bar,
        title="Basket contribution to the DXY change (points)",
        height=360,
        ytitle="",
        xtitle="DXY points",
        showlegend=False,
    ),
    width="stretch",
)

st.subheader("CPI week — spike, then fade")
st.caption(
    "Schwab's 15 Oct note was written after the CPI spike and before PPI. "
    "Hike odds jumped, then Fed speakers and the AI-credit bid in Treasuries "
    "took the dollar off the 18-month high. The *level* stayed elevated."
)
fig_ev = go.Figure()
fig_ev.add_trace(
    go.Scatter(
        x=events["date"],
        y=events["dxy"],
        name="DXY",
        line=dict(color=AMBER, width=2.4),
        hovertemplate="%{y:.2f}<extra>DXY</extra>",
    )
)
_vline(fig_ev, CPI_PRINT, "CPI 14 Oct")
_vline(fig_ev, PPI_PRINT, "PPI 15 Oct")
st.plotly_chart(
    _style(fig_ev, title="DXY around the September CPI / PPI prints", height=360, ytitle="Index"),
    width="stretch",
)

if not link.empty:
    fig_c = go.Figure()
    fig_c.add_trace(
        go.Scatter(
            x=link["date"],
            y=link["corr"],
            name="60-session correlation",
            line=dict(color=TEAL, width=2.0),
            hovertemplate="%{y:.2f}<extra>corr</extra>",
        )
    )
    fig_c.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.4)")
    st.plotly_chart(
        _style(
            fig_c,
            title="DXY daily return vs US 2-year yield change",
            height=340,
            ytitle="Correlation",
        ),
        width="stretch",
    )
    last_corr = metrics["corr_dxy_dgs2"]
    if pd.notna(last_corr):
        st.caption(
            f"Full-window correlation of DXY returns with 2-year yield changes: **{last_corr:+.2f}**. "
            "Positive: the dollar still trades as a rates instrument, not a risk-off residual."
        )

st.subheader("Schwab 15 Oct context")
st.markdown(
    """
September **core CPI +0.3%** for a second month; three-month annualized **3.4%** versus a **2%** goal.
Headline **3.7%** year over year (vs 3.6% expected). Core **2.6%** (vs 2.5%).
October hike odds jumped from about **20%** to **35%** on CME FedWatch in Schwab's late-Wednesday read
(other desks printed closer to **50%** into the close).

PPI the next morning was hotter still: final demand **+1.3%** month over month and **+6.3%** year over year —
the largest annual rise since late 2022. The dollar did not hold the CPI high. US 2-year closed **4.77%**
on 15 Oct after an intraday spike toward **4.89%**. FOMC **27–28 Oct**. PCE **29 Oct**.

The question is not "did CPI print hot." It is whether the **relative** US policy path still pays you
to hold dollars against a basket that is mostly euros.
    """
)

st.download_button(
    "Download window CSV",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="dxy-window.csv",
    mime="text/csv",
)

bundled = load_warehouse_metadata()
if bundled:
    st.caption(
        f"Bundled snapshot {bundled.get('panel_start')} → {bundled.get('panel_end')} "
        f"({bundled.get('source')}). Toggle live refresh to pull Yahoo chart + FRED again."
    )
