from __future__ import annotations
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List

from .helpers import (
    fmt_currency,
    round_price,
    round_ppm2,
)

# === Palette (from PropMind logo) ===
COLOR_PRIMARY = "#0a628d"  # PropMind Blue
COLOR_ACCENT  = "#23b4c8"  # PropMind Teal
COLOR_LIGHT   = "#c0dbe1"  # Light tint for bands/fills
COLOR_GRAY    = "#6b7280"  # Neutral for secondary/negative
PRICE_BAND_COLOR = "#b1fcd5"

CHART_HEIGHT     = 350   # default height for most charts (consistent)
CHART_HEIGHT_SM  = 160   # compact height (e.g., price interval)
CHART_XPAD_RATIO = 0.15  # widen the default view a bit (was ~0.08)
CHART_WIDTH = 960  # pick what you want

PLOTLY_FONT = {
    "family": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
    "size": 13,
}

def _rgba(hex_color: str, a: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def _lighten(hex_color: str, amt: float = 0.35) -> str:
    hex_color = hex_color.lstrip("#")
    r,g,b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    lr = int(r + (255 - r)*amt); lg = int(g + (255 - g)*amt); lb = int(b + (255 - b)*amt)
    return f"#{lr:02x}{lg:02x}{lb:02x}"

# Use the SAME band styling everywhere
FILL_BAND = _rgba(PRICE_BAND_COLOR, 0.1)   # fill for P25–P75 and likelihood area
BAND_LINE = _rgba(COLOR_ACCENT, 0.45)   # faint outline for band edges


def _fig_to_div(fig: go.Figure, static: bool = False) -> str:
    # Merge margins: use chart's margins if they exist, otherwise defaults
    default_margin = dict(l=30, r=20, t=30, b=30)
    existing = (fig.layout.margin.to_plotly_json()
                if getattr(fig.layout, "margin", None) else {})
    merged_margin = {**default_margin, **{k: v for k, v in existing.items() if v is not None}}

    fig.update_layout(
        template="plotly_white",
        margin=merged_margin,
        font=PLOTLY_FONT,
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[COLOR_PRIMARY, COLOR_ACCENT, COLOR_GRAY],
    )
    config = dict(responsive=True)
    if static:
        config.update(staticPlot=True, displayModeBar=False)
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config=config)



def chart_price_interval(pred: float, low: float, high: float) -> str:
    pred = round_price(pred); low = round_price(low); high = round_price(high)

    fig = go.Figure()
    # band
    fig.add_trace(go.Scatter(
        x=[low, high], y=[0, 0], mode="lines",
        line=dict(width=10, color=COLOR_LIGHT),
        showlegend=False, hoverinfo="skip"
    ))
    # markers
    fig.add_trace(go.Scatter(
        x=[low, pred, high], y=[0, 0, 0], mode="markers+text",
        marker=dict(
            size=[18, 12, 18],
            color=[COLOR_ACCENT, COLOR_PRIMARY, COLOR_ACCENT],
            symbol=["line-ns-open", "circle", "line-ns-open"]
        ),
        text=[f"Low<br>{fmt_currency(low)}", f"Predicted<br>{fmt_currency(pred)}", f"High<br>{fmt_currency(high)}"],
        textposition=["top center", "bottom center", "top center"],
        hoverinfo="skip", showlegend=False
    ))
    span = max(1.0, high - low)
    pad = 0.25 * span
    fig.update_xaxes(showgrid=False, showline=False, zeroline=False, showticklabels=False, range=[low - pad, high + pad])
    fig.update_yaxes(visible=False, range=[-0.2, 0.2])
    fig.update_layout(height=CHART_HEIGHT_SM)
    return _fig_to_div(fig, static=True)



def chart_shap(shap_values: Dict[str, Dict], top_n: int = 6):
    nice = {
        "location": "Location",
        "energy efficiency": "Energy efficiency (EPC)",
        "property style": "Property Style",
        "distance to amenities": "Amenities",
        "space layout": "Space layout",
        "seasonality": "Seasonality",
        "residual_factors": "Other residual factors",
    }

    rows = []
    for k, v in shap_values.items():
        if v.get("negligible", False):
            continue
        impact = float(v.get("impact", 0.0))
        features = v.get("features", [])
        rows.append({
            "key": k,
            "feature": nice.get(k, k.title()),
            "impact": impact,
            "features": features,
        })

    df = pd.DataFrame(rows)
    # If nothing left
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No material group impacts",
            x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper"
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return _fig_to_div(fig, static=True), pd.DataFrame([])

    # If only residual_factors remains → don’t plot a lone bar
    if len(df) == 1 and df.iloc[0]["key"] == "residual_factors":
        fig = go.Figure()
        fig.add_annotation(
            text="No material group impacts (only residual factors present)",
            x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper"
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return _fig_to_div(fig, static=True), pd.DataFrame([])

    # Sort descending by absolute impact
    df = df.sort_values("impact", key=lambda s: s.abs(), ascending=False)
    if len(df) > top_n:
        df = df.head(top_n)

    # Colors
    bar_colors = [COLOR_LIGHT if x >= 0 else _lighten(COLOR_GRAY, 0.45) for x in df["impact"]]
    line_colors = [COLOR_PRIMARY if x >= 0 else COLOR_GRAY for x in df["impact"]]

    # Hover text with included features
    hovertext = [
        f"{row['feature']}<br>Includes: {', '.join(row['features'])}"
        if row["features"] else row["feature"]
        for _, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["impact"],
        y=df["feature"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color=line_colors, width=1)),
        hovertext=hovertext,
        hoverinfo="text",
        name="Impact",
    ))

    # Centerline at 0
    fig.add_shape(
        type="line", x0=0, x1=0, y0=-0.5, y1=len(df) - 0.5,
        line=dict(width=1, color="#999")
    )

    fig.update_xaxes(title_text="Impact on price (£)", tickformat=",")
    fig.update_layout(bargap=0.25, height=CHART_HEIGHT, width=CHART_WIDTH)

    return _fig_to_div(fig, static=True), df


def chart_trend_toggle(trend: list, property_ppm2: float, property_price: float, title: str) -> str:
    df = pd.DataFrame(trend).copy()
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(text="No trend data", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        fig.update_layout(height=CHART_HEIGHT+100, width=CHART_WIDTH)
        return _fig_to_div(fig)

    # --- prep & x-range (a bit "unzoomed") ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    dt = df["date"].dropna()
    x_range = None
    if len(dt) >= 2:
        span = dt.max() - dt.min()
        xpad = span * (CHART_XPAD_RATIO if 'CHART_XPAD_RATIO' in globals() else 0.15)
        x_range = [dt.min() - xpad, dt.max() + xpad]

    def pick(*cols):
        for c in cols:
            if c in df.columns: return c
        return None

    p25_ppm2 = pick("p25_ppm2_3m", "p25_ppm2_6m")
    p75_ppm2 = pick("p75_ppm2_3m", "p75_ppm2_6m")
    med_ppm2 = pick("median_ppm2_3m", "median_ppm2_6m")
    p25_price = pick("p25_price_3m", "p25_price_6m")
    p75_price = pick("p75_price_3m", "p75_price_6m")
    med_price = pick("median_price_3m", "median_price_6m")

    has_ppm2  = all(c is not None for c in (p25_ppm2, p75_ppm2, med_ppm2))
    has_price = all(c is not None for c in (p25_price, p75_price, med_price))

    FT2_PER_M2 = 10.763910416709722

    traces_cnt = []
    # ---------------- £/m² ----------------
    if has_ppm2:
        lo_m2 = df[[p25_ppm2, p75_ppm2]].min(axis=1)
        hi_m2 = df[[p25_ppm2, p75_ppm2]].max(axis=1)
        mask  = df["date"].notna() & lo_m2.notna() & hi_m2.notna()
        x     = df.loc[mask, "date"]
        med_m = df["date"].notna() & df[med_ppm2].notna()

        prop_m2 = round_ppm2(property_ppm2)

        # lower edge (faint)
        fig.add_trace(go.Scatter(x=x, y=lo_m2[mask], mode="lines",
                                 line=dict(width=1, color=BAND_LINE),
                                 hoverinfo="skip", showlegend=False, visible=True))
        # upper edge + fill
        fig.add_trace(go.Scatter(x=x, y=hi_m2[mask], mode="lines", fill="tonexty",
                                 line=dict(width=1, color=BAND_LINE),
                                 name="P25–P75 (£/m²)", fillcolor=FILL_BAND,
                                 hovertemplate="Date: %{x|%Y-%m-%d}<br>£/m²: %{y:,.0f}<extra></extra>",
                                 visible=True))
        # median
        fig.add_trace(go.Scatter(x=df.loc[med_m, "date"], y=df.loc[med_m, med_ppm2], mode="lines",
                                 name="Median (£/m²)", line=dict(color=COLOR_PRIMARY, width=2),
                                 hovertemplate="Date: %{x|%Y-%m-%d}<br>£/m²: %{y:,.0f}<extra></extra>",
                                 visible=True))
        # property
        x_min = (x.min() if len(x) else dt.min()); x_max = (x.max() if len(x) else dt.max())
        fig.add_trace(go.Scatter(x=[x_min, x_max], y=[prop_m2, prop_m2], mode="lines",
                                 name="Property £/m²",
                                 line=dict(color=COLOR_ACCENT, width=2, dash="dash"),
                                 visible=True))

        y0 = min(lo_m2.min(), prop_m2); y1 = max(hi_m2.max(), prop_m2)
        pad = 0.45 * (y1 - y0 if y1 > y0 else 1.0)
        y_range_m2 = [y0 - pad, y1 + pad]
        traces_cnt.append(4)
    else:
        y_range_m2 = None; traces_cnt.append(0)

    # ---------------- £/ft² ----------------
    if has_ppm2:
        # reuse same mask/x to keep band aligned
        lo_ft = (lo_m2 / FT2_PER_M2); hi_ft = (hi_m2 / FT2_PER_M2)
        med_m = df["date"].notna() & df[med_ppm2].notna()
        prop_ft = (round_ppm2(property_ppm2) / FT2_PER_M2)

        fig.add_trace(go.Scatter(x=x, y=lo_ft[mask], mode="lines",
                                 line=dict(width=1, color=BAND_LINE),
                                 hoverinfo="skip", showlegend=False, visible=False))
        fig.add_trace(go.Scatter(x=x, y=hi_ft[mask], mode="lines", fill="tonexty",
                                 line=dict(width=1, color=BAND_LINE),
                                 name="P25–P75 (£/ft²)", fillcolor=FILL_BAND,
                                 hovertemplate="Date: %{x|%Y-%m-%d}<br>£/ft²: %{y:,.0f}<extra></extra>",
                                 visible=False))
        fig.add_trace(go.Scatter(x=df.loc[med_m, "date"], y=(df.loc[med_m, med_ppm2] / FT2_PER_M2),
                                 mode="lines", name="Median (£/ft²)",
                                 line=dict(color=COLOR_PRIMARY, width=2),
                                 hovertemplate="Date: %{x|%Y-%m-%d}<br>£/ft²: %{y:,.0f}<extra></extra>",
                                 visible=False))
        fig.add_trace(go.Scatter(x=[x_min, x_max], y=[prop_ft, prop_ft], mode="lines",
                                 name="Property £/ft²",
                                 line=dict(color=COLOR_ACCENT, width=2, dash="dash"),
                                 visible=False))

        y0f = min(lo_ft[mask].min(), prop_ft); y1f = max(hi_ft[mask].max(), prop_ft)
        padf = 0.45 * (y1f - y0f if y1f > y0f else 1.0)
        y_range_ft = [y0f - padf, y1f + padf]
        traces_cnt.append(4)
    else:
        y_range_ft = None; traces_cnt.append(0)

    # ---------------- £ (price) ----------------
    if has_price:
        lo_p = df[[p25_price, p75_price]].min(axis=1)
        hi_p = df[[p25_price, p75_price]].max(axis=1)
        maskp = df["date"].notna() & lo_p.notna() & hi_p.notna()
        xp = df.loc[maskp, "date"]
        med_p = df["date"].notna() & df[med_price].notna()
        prop_p = round_price(property_price)

        fig.add_trace(go.Scatter(x=xp, y=lo_p[maskp], mode="lines",
                                 line=dict(width=1, color=BAND_LINE),
                                 hoverinfo="skip", showlegend=False, visible=False))
        fig.add_trace(go.Scatter(x=xp, y=hi_p[maskp], mode="lines", fill="tonexty",
                                 line=dict(width=1, color=BAND_LINE),
                                 name="P25–P75 (£)", fillcolor=FILL_BAND,
                                 hovertemplate="Date: %{x|%Y-%m-%d}<br>£: %{y:,.0f}<extra></extra>",
                                 visible=False))
        fig.add_trace(go.Scatter(x=df.loc[med_p, "date"], y=df.loc[med_p, med_price], mode="lines",
                                 name="Median (£)", line=dict(color=COLOR_PRIMARY, width=2),
                                 hovertemplate="Date: %{x|%Y-%m-%d}<br>£: %{y:,.0f}<extra></extra>",
                                 visible=False))
        x_min_p = (xp.min() if len(xp) else dt.min()); x_max_p = (xp.max() if len(xp) else dt.max())
        fig.add_trace(go.Scatter(x=[x_min_p, x_max_p], y=[prop_p, prop_p], mode="lines",
                                 name="Property £",
                                 line=dict(color=COLOR_ACCENT, width=2, dash="dash"),
                                 visible=False))

        y0p = min(lo_p.min(), prop_p); y1p = max(hi_p.max(), prop_p)
        padp = 0.45 * (y1p - y0p if y1p > y0p else 1.0)
        y_range_p = [y0p - padp, y1p + padp]
        traces_cnt.append(4)
    else:
        y_range_p = None; traces_cnt.append(0)

    # --- x/y axes & title ---
    if x_range: fig.update_xaxes(range=x_range, title_text=title)
    else:       fig.update_xaxes(title_text=title)

    # --- updatemenus (toggle) ---
    m2, ft2, pr = traces_cnt  # counts per group
    vis_m2  = [True]*m2 + [False]*(ft2 + pr)
    vis_ft2 = [False]*m2 + [True]*ft2 + [False]*pr
    vis_pr  = [False]*(m2 + ft2) + [True]*pr

    buttons = []
    if m2:
        buttons.append(dict(
            label="£/m²", method="update",
            args=[{"visible": vis_m2},
                  {"yaxis": {"title": "£/m²", "range": y_range_m2, "tickformat": ","}}]
        ))
    if ft2:
        buttons.append(dict(
            label="£/ft²", method="update",
            args=[{"visible": vis_ft2},
                  {"yaxis": {"title": "£/ft²", "range": y_range_ft, "tickformat": ","}}]
        ))
    if pr:
        buttons.append(dict(
            label="£", method="update",
            args=[{"visible": vis_pr},
                  {"yaxis": {"title": "£", "range": y_range_p, "tickformat": ","}}]
        ))

    # after you build `buttons`, add:
    if buttons:
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=0.0, y=1.12,  # place above the plot
                xanchor="left", yanchor="bottom",
                showactive=True,
                bgcolor="#ffffff",  # visible on white template
                bordercolor="#d1d5db", borderwidth=1,
                pad={"l": 4, "r": 4, "t": 4, "b": 4},
                buttons=buttons
            )],
            margin=dict(t=80)  # <- ensure there's space (preserved by _fig_to_div)
        )

    # default y-axis (first visible set)
    if m2:   fig.update_yaxes(title_text="£/m²", range=y_range_m2, tickformat=",")
    elif ft2:fig.update_yaxes(title_text="£/ft²", range=y_range_ft, tickformat=",")
    elif pr: fig.update_yaxes(title_text="£", range=y_range_p, tickformat=",")

    fig.update_layout(height=CHART_HEIGHT + 100, width=CHART_WIDTH)
    return _fig_to_div(fig)



def chart_borrow_rates(borrow_rate) -> str:
    """
    Accepts:
      - a list of dicts [{date, rate_2y, rate_5y}, ...], or
      - a dict of lists {"date": [...], "rate_2y":[...], "rate_5y":[...]}, or
      - a single dict {"date": ..., "rate_2y": ..., "rate_5y": ...}

    Returns: Plotly HTML <div> as string.
    """
    # ---------- DF COERCION (unchanged structure) ----------
    if borrow_rate is None:
        borrow_rate = []

    if isinstance(borrow_rate, dict):
        if any(isinstance(v, (list, tuple, np.ndarray)) for v in borrow_rate.values()):
            df = pd.DataFrame(borrow_rate)
        else:
            df = pd.DataFrame([borrow_rate])
    else:
        df = pd.DataFrame(borrow_rate)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No borrow rate data", x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False
        )
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return _fig_to_div(fig, static=True)

    expected = {"date", "rate_2y", "rate_5y"}
    missing = expected - set(df.columns)
    if missing:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing columns: {', '.join(sorted(missing))}",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return _fig_to_div(fig, static=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    for col in ["rate_2y", "rate_5y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------- PLOTTING (PropMind palette + unified size) ----------
    fig = go.Figure()

    mask2 = df["date"].notna() & df["rate_2y"].notna()
    fig.add_trace(go.Scatter(
        x=df.loc[mask2, "date"],
        y=df.loc[mask2, "rate_2y"],
        mode="lines",
        name="2y",
        line=dict(color=COLOR_PRIMARY, width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>2y: %{y:.2f}%<extra></extra>",
    ))

    mask5 = df["date"].notna() & df["rate_5y"].notna()
    fig.add_trace(go.Scatter(
        x=df.loc[mask5, "date"],
        y=df.loc[mask5, "rate_5y"],
        mode="lines",
        name="5y",
        line=dict(color=COLOR_ACCENT, width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>5y: %{y:.2f}%<extra></extra>",
    ))

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Borrow rate (%)", tickformat=".2f")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=CHART_HEIGHT, width=CHART_WIDTH
    )

    return _fig_to_div(fig)



def _pmt_monthly(rate_pct: float, months: int, principal: float) -> float:
    """Standard repayment PMT with monthly compounding."""
    r = (rate_pct / 100.0) / 12.0
    if r == 0:
        return principal / months if months else 0.0
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)

def compute_payment_grid(
    rate_pct: float,
    property_price: float | None = None,
    loan_amount: float | None = None,
    maturities_years: list[int] = None,
    upfront_percents: list[float] = None,
) -> pd.DataFrame:
    """Return a DataFrame of £/month indexed by upfront% with columns as terms."""
    if maturities_years is None:
        maturities_years = [10, 15, 20, 25, 30]
    if upfront_percents is None:
        upfront_percents = [0, 5, 10, 20, 40]

    rows = []
    index = []
    for up in upfront_percents:
        index.append(f"{up:.0f}%")
        row_vals = []
        for yrs in maturities_years:
            if loan_amount is not None:
                L = float(loan_amount)
            else:
                L = (float(property_price) * (1 - up / 100.0)) if property_price is not None else 0.0
            pmt = _pmt_monthly(rate_pct, yrs * 12, L) if L > 0 else 0.0
            row_vals.append(pmt)
        rows.append(row_vals)

    df = pd.DataFrame(rows, index=index, columns=[f"{y}y" for y in maturities_years])
    return df

def chart_sale_likelihood(curve: List[Dict], pred_price: float) -> str:
    """
    curve: [{'price': <number>, 'likelihood_pct': <number>}, ...]
    """
    if not curve:
        fig = go.Figure()
        fig.add_annotation(text="No sale-likelihood data", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return _fig_to_div(fig, static=True)

    x = [float(p["price"]) for p in curve]
    y = [float(p["likelihood_pct"]) for p in curve]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=COLOR_PRIMARY, width=2),
        fill="tozeroy", fillcolor=FILL_BAND,
        hovertemplate="£%{x:,.0f}<br>Likelihood: %{y:.1f}%<extra></extra>",
        name="Sales likelihood"
    ))

    # vertical line at predicted price
    fig.add_vline(x=pred_price, line_width=2, line_dash="dash",
                  line_color=COLOR_ACCENT, annotation_text="Predicted price",
                  annotation_position="top")

    fig.update_xaxes(title_text="Asking price (£)", tickformat=",")
    fig.update_yaxes(title_text="Likelihood (%)", range=[0, 100], ticksuffix="%")
    fig.update_layout(height=CHART_HEIGHT, width=CHART_WIDTH)
    return _fig_to_div(fig)
