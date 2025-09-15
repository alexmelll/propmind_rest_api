"""
Amenities enrichment utilities for property data.
Production-hardened: validation, logging, robust API calls.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from rest_api.utils.features import (
    parse_age_band,
    infer_likely_tenure,
    clean_nans,
    get_postcode_from_address,
)
from rest_api.db.accessors.epc_reduced_accessors import get_epc_data_from_db
from rest_api.utils.s3_utils import load_csv

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
DATA_DIR = os.getenv("DATA_DIR", "../data")

# =====================================================================
# Distance utilities
# =====================================================================
def add_nearest_distance(
    df: pd.DataFrame, ref_df: pd.DataFrame, output_col: str = "dist_to_nearest_km"
) -> pd.DataFrame:
    """
    Adds nearest distance (in km) from each row in df to the closest point in ref_df.
    Assumes 'lat' and 'lon' columns exist in both.

    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame with 'lat' and 'lon' columns.
    ref_df : pd.DataFrame
        Reference DataFrame with 'lat' and 'lon' columns.
    output_col : str
        Name of the output distance column.

    Returns
    -------
    pd.DataFrame
        df with an extra column for nearest distance (km).
    """
    if df.empty or ref_df.empty:
        logger.warning("Empty DataFrame(s) passed to add_nearest_distance.")
        df[output_col] = np.nan
        return df

    df_clean = df.dropna(subset=["lat", "lon"]).copy()
    ref_clean = ref_df.dropna(subset=["lat", "lon"]).copy()

    df_clean["lat"] = pd.to_numeric(df_clean["lat"], errors="coerce")
    df_clean["lon"] = pd.to_numeric(df_clean["lon"], errors="coerce")
    ref_clean["lat"] = pd.to_numeric(ref_clean["lat"], errors="coerce")
    ref_clean["lon"] = pd.to_numeric(ref_clean["lon"], errors="coerce")

    df_clean = df_clean[np.isfinite(df_clean["lat"]) & np.isfinite(df_clean["lon"])]
    ref_clean = ref_clean[np.isfinite(ref_clean["lat"]) & np.isfinite(ref_clean["lon"])]

    if ref_clean.empty:
        logger.warning("No valid reference coordinates in add_nearest_distance.")
        df[output_col] = np.nan
        return df

    # Convert to radians for haversine approximation
    points = np.radians(df_clean[["lat", "lon"]].to_numpy())
    ref_points = np.radians(ref_clean[["lat", "lon"]].to_numpy())

    tree = cKDTree(ref_points)
    dist_radians, _ = tree.query(points, k=1)

    dist_km = dist_radians * EARTH_RADIUS_KM
    df.loc[df_clean.index, output_col] = dist_km

    return df


# =====================================================================
# Coordinates from API
# =====================================================================
def get_coords_batch(postcodes: List[str]) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """
    Query postcodes.io for coordinates of a batch of UK postcodes.

    Returns list of (postcode, lat, lon). If lookup fails, lat/lon = None.
    """
    url = "https://api.postcodes.io/postcodes"
    try:
        response = requests.post(url, json={"postcodes": postcodes}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error("Postcodes.io request failed: %s", e)
        return [(pc, None, None) for pc in postcodes]

    results = response.json().get("result", [])
    coords: List[Tuple[str, Optional[float], Optional[float]]] = []
    for res in results:
        pc = res.get("query")
        if res.get("result"):
            coords.append(
                (pc, res["result"].get("latitude"), res["result"].get("longitude"))
            )
        else:
            coords.append((pc, None, None))
    return coords


def add_coords_to_df(
    df: pd.DataFrame, postcode_col: str = "postcode", batch_size: int = 100
) -> pd.DataFrame:
    """
    Add 'lat' and 'lon' columns to DataFrame by resolving postcodes via API.

    Parameters
    ----------
    df : pd.DataFrame
    postcode_col : str
        Column name containing postcodes.
    batch_size : int
        Batch size for API requests.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["lat"] = np.nan
    df["lon"] = np.nan

    unique_postcodes = (
        df[postcode_col].dropna().astype(str).str.upper().unique().tolist()
    )

    for i in range(0, len(unique_postcodes), batch_size):
        batch = unique_postcodes[i : i + batch_size]
        coords = get_coords_batch(batch)
        for postcode, lat, lon in coords:
            mask = df[postcode_col].astype(str).str.upper() == postcode.upper()
            if lat is not None and lon is not None:
                df.loc[mask, "lat"] = float(lat)
                df.loc[mask, "lon"] = float(lon)

    return df


# =====================================================================
# Static datasets
# =====================================================================
def load_static_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load static amenity datasets from CSV files.

    Returns
    -------
    dict
        {"parks": df, "schools": df, "tubes": df}
    """
    try:
        parks_df = load_csv("parks.csv")
        schools_df = load_csv("schools.csv")
        tubes_df = load_csv("tubes.csv")
    except Exception as e:
        logger.exception("Failed to load static datasets.")
        raise

    return {"parks": parks_df, "schools": schools_df, "tubes": tubes_df}


# =====================================================================
# Enrichment functions
# =====================================================================
def enrich_dist_to_amenities(
    df: pd.DataFrame, static_data: Dict[str, pd.DataFrame], postcode_col: str = "postcode"
) -> pd.DataFrame:
    """
    Add distance-to-amenities columns (park, school, tube) to df.

    Returns enriched DataFrame.
    """
    if df.empty:
        return df

    df = add_coords_to_df(df, postcode_col=postcode_col)
    df = add_nearest_distance(df, static_data["parks"], "dist_to_park")
    df = add_nearest_distance(df, static_data["schools"], "dist_to_school")
    df = add_nearest_distance(df, static_data["tubes"], "dist_to_tube")

    return df


def enrich_property_data(address: str, static_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Enrich property data from EPC DB and static datasets.

    Steps:
    - Try EPC DB lookup
    - If missing, create minimal record (address + postcode)
    - Parse property age & infer tenure
    - Add distances to amenities

    Returns
    -------
    dict
        Cleaned and enriched property data.
    """
    if not isinstance(address, str) or not address.strip():
        raise ValueError("Invalid address provided to enrich_property_data.")

    address_norm = address.strip().upper()

    try:
        raw = get_epc_data_from_db(address_norm)
    except Exception as e:
        logger.error("EPC DB lookup failed for %s: %s", address_norm, e)
        raw = None

    if not raw or (isinstance(raw, dict) and len(raw.keys()) <= 1):
        match_df = pd.DataFrame(
            [{"full_address": address_norm, "postcode": get_postcode_from_address(address_norm)}]
        )
    else:
        match_df = pd.DataFrame([raw])
        match_df["built_date"] = match_df["property_age"].map(parse_age_band)
        match_df["tenure"] = match_df.apply(
            lambda row: infer_likely_tenure(
                property_type=row.get("property_type"), age_info=row.get("built_date")
            ),
            axis=1,
        )

    match_df = enrich_dist_to_amenities(match_df, static_data)
    return clean_nans(match_df.iloc[0])
