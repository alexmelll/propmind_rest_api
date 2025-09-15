"""
Property metadata parsing and cleaning utilities.
Production-hardened: defensive checks, logging, normalization.
"""

import logging
import re
from typing import Any, Dict, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ======================
# Age band parsing
# ======================
def parse_age_band(value: Union[str, float, None]) -> Dict[str, Any]:
    """
    Parse a string like '1950-1966', 'before 1900', 'after 2012', or '1999'
    into a structured dict with start_year, end_year, exact_year, and label.
    """
    if pd.isna(value) or value is None:
        return {"type": "na", "start_year": None, "end_year": None, "exact_year": None, "label": "N/A"}

    val = str(value).strip()

    # Match a range like '1950-1966'
    range_match = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", val)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        return {"type": "range", "start_year": start, "end_year": end, "exact_year": int(0.5*(start+end)), "label": f"{start}–{end}"}

    # Match "before 1900"
    before_match = re.fullmatch(r"before\s+(\d{4})", val, re.IGNORECASE)
    if before_match:
        year = int(before_match.group(1))
        return {"type": "before", "start_year": None, "end_year": year, "exact_year": year, "label": f"before {year}"}

    # Match "after" or "onwards"
    after_match = re.fullmatch(r"(after|onwards)\s*(\d{4})", val, re.IGNORECASE)
    if after_match:
        year = int(after_match.group(2))
        return {"type": "after", "start_year": year, "end_year": None, "exact_year": year, "label": f"after {year}"}

    # Match exact year
    exact_match = re.fullmatch(r"\d{4}", val)
    if exact_match:
        year = int(val)
        return {"type": "exact", "start_year": None, "end_year": None, "exact_year": year, "label": str(year)}

    # Known invalid values
    if val.upper() in {"NO DATA!", "INVALID!"}:
        return {"type": "invalid", "start_year": None, "end_year": None, "exact_year": None, "label": "Invalid"}

    logger.warning("Unrecognized age band format: %r", val)
    return {"type": "unknown", "start_year": None, "end_year": None, "exact_year": None, "label": "Unknown"}


# ======================
# Tenure inference
# ======================
def infer_likely_tenure(property_type: str, age_info: Dict[str, Any]) -> str:
    """
    Infer tenure (Freehold/Leasehold/Unknown) from property type and build year.
    - Flats/Maisonettes → Leasehold
    - Houses/Bungalows → Freehold unless very recent build (≥2010 → Leasehold)
    """
    prop = (property_type or "").lower().strip()

    # Rule 1: Flats & maisonettes → leasehold
    if "flat" in prop or "maisonette" in prop:
        return "Leasehold"

    # Build year extraction
    build_year: Optional[int]
    if age_info.get("type") == "exact":
        build_year = age_info.get("exact_year")
    elif age_info.get("type") == "range":
        build_year = age_info.get("start_year")
    else:
        build_year = None

    # Rule 2: Houses/Bungalows → usually freehold
    if any(house_type in prop for house_type in ["house", "detached", "semi-detached", "terrace", "bungalow"]):
        if build_year and build_year >= 2010:
            return "Leasehold"
        return "Freehold"

    return "Unknown"


# ======================
# Cleaning helpers
# ======================
def clean_nans(row: pd.Series) -> Dict[str, Any]:
    """
    Convert NaNs in a pandas row to None for serialization.
    """
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def compute_building_age(col: pd.Series) -> pd.Series:
    """
    Map age bands or year-like strings to a representative year.
    Returns a Series of integers or NaNs.
    """
    age_band_mapping: Dict[str, int] = {
        "1996-2002": 1999,
        "1950-1966": 1958,
        "1983-1990": 1987,
        "1976-1982": 1979,
        "1930-1949": 1940,
        "before 1900": 1850,
        "1900-1929": 1915,
        "1967-1975": 1971,
        "after 2007": 2015,
        "2003-2006": 2005,
        "2007-2011": 2009,
        "1991-1995": 1992,
        "after 2012": 2018,
        "2023": 2023,
        "2019": 2019,
        "2022": 2022,
        "2024": 2024,
        "2017": 2017,
        "2020": 2020,
        "2018": 2018,
        "2016": 2016,
        "2012": 2012,
        "1920": 1920,
        "2015": 2015,
        "1900": 1900,
        "2014": 2014,
        "2013": 2013,
        "2021": 2021,
        "1929": 1929,
    }
    return col.map(age_band_mapping)


# ======================
# Postcode extraction
# ======================
def get_postcode_from_address(address: str) -> Optional[str]:
    """
    Extract a UK postcode from a free-text address using regex.
    Always returns postcode in uppercase with a single space before the final 3 characters.
    Returns None if not found.
    """
    if not isinstance(address, str):
        return None

    postcode_regex = re.compile(r"\b([A-Z]{1,2}[0-9][0-9A-Z]?)\s*([0-9][A-Z]{2})\b", re.IGNORECASE)
    match = postcode_regex.search(address)
    if match:
        return f"{match.group(1).upper()} {match.group(2).upper()}"
    return None

# =========================
# Postcode helpers
# =========================
def sector(pc: str) -> Optional[str]:
    m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d)", str(pc).upper())
    return f"{m.group(1)} {m.group(2)}" if m else None


def prefix(pc: str) -> Optional[str]:
    m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)", str(pc).upper())
    return m.group(1) if m else None