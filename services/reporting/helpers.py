from __future__ import annotations
import html as htmlmod
import re
from typing import Any, Optional
import pandas as pd

# ---------- constants ----------
FT2_PER_M2: float = 10.763910416709722
MI_PER_KM: float = 0.621371
ROUND_PRICE_NEAREST = 500
ROUND_PPM2_NEAREST = 10

# ---------- primitives ----------

def round_to_nearest(x: float, base: int) -> float:
    try:
        return int(round(float(x) / base)) * base
    except Exception:
        return x

def to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def is_na(x: Any) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x))

# ---------- rounding & formatting ----------

def round_price(x: float) -> float:
    return round_to_nearest(x, ROUND_PRICE_NEAREST)

def round_ppm2(x: float) -> float:
    return round_to_nearest(x, ROUND_PPM2_NEAREST)

def fmt_currency(x: float) -> str:
    return f"£{float(x):,.0f}"


def fmt_int(x, default: str = "-") -> str:
    if is_na(x):
        return default
    try:
        return str(int(round(float(x))))
    except Exception:
        return default


def fmt_pct(val: float, decimals: int = 0) -> str:
    """
    Format a float 0–100 as a percentage string for display.

    Examples
    --------
    fmt_pct(87.456)   -> "87%"
    fmt_pct(87.456,1) -> "87.5%"
    """
    if val is None:
        return "-"
    try:
        return f"{round(float(val), decimals)}%"
    except Exception:
        return "-"


def fmt_m2_ft2(x) -> str:
    try:
        m2 = int(round(float(x)))
        ft2 = int(round(m2 * FT2_PER_M2))
        return f"{m2} m² ({ft2} ft²)"
    except Exception:
        return "-"


def fmt_km_mi(x) -> str:
    try:
        km = float(x)
        mi = km * MI_PER_KM
        return f"{km:.2f} km ({mi:.2f} mi)"
    except Exception:
        return "-"


def fmt_ppm2_with_ft2(ppm2) -> str:
    try:
        ppm2_rounded = round_ppm2(float(ppm2))
        per_ft2 = round(float(ppm2_rounded) / FT2_PER_M2)
        return f"{ppm2_rounded:,.0f} ({per_ft2:,.0f})"
    except Exception:
        return "-"


def fmt_km2(x) -> str:
    if is_na(x):
        return "-"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def pretty_date(x) -> str:
    if is_na(x):
        return "-"
    try:
        ts = pd.to_datetime(x)
    except Exception:
        return str(x)
    return ts.strftime("%B %Y")

# ---------- HTML helpers ----------

def esc(text: str) -> str:
    """HTML-escape + <br>, and support **bold**."""
    if text is None:
        return ""
    # Replace **...** before escaping
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(text))
    # Now escape everything except the <b> tags
    text = htmlmod.escape(text).replace("\n", "<br>")
    # Unescape the <b> tags
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    return text


def dual_line(base: str, alt: str) -> str:
    return (
        "<div class='dual'>"
        f"<span class='base'>{htmlmod.escape(base)}</span>"
        f"<span class='alt'>{htmlmod.escape(alt)}</span>"
        "</div>"
    )


def fmt_ppm2_with_ft2_html(ppm2) -> str:
    try:
        ppm2_rounded = round_ppm2(float(ppm2))
        per_ft2 = round(float(ppm2_rounded) / FT2_PER_M2)
        return dual_line(f"{ppm2_rounded:,.0f}", f"({per_ft2:,.0f})")
    except Exception:
        return dual_line("-", "(-)")


def fmt_m2_ft2_html(x) -> str:
    try:
        m2 = int(round(float(x)))
        ft2 = int(round(m2 * FT2_PER_M2))
        return dual_line(f"{m2} m²", f"({ft2} ft²)")
    except Exception:
        return dual_line("-", "(-)")


def fmt_km_mi_html(x) -> str:
    try:
        km = float(x)
        mi = km * MI_PER_KM
        return dual_line(f"{km:.2f} km", f"({mi:.2f} mi)")
    except Exception:
        return dual_line("-", "(-)")


EPC_BAND_RANGES = {
    "A": (92, 120),   # cap high at 120 for practicality
    "B": (81, 91),
    "C": (69, 80),
    "D": (55, 68),
    "E": (39, 54),
    "F": (21, 38),
    "G": (0, 20),
}

def epc_letter_from_score(score: float) -> str:
    if score is None:
        return "-"
    s = float(score)
    for band, (lo, hi) in EPC_BAND_RANGES.items():
        if lo <= s <= hi:
            return band
    return "A" if s > 120 else "G"  # safety fallback