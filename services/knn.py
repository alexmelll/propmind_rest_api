"""
KNN-based comparable property search and valuation.
Production-hardened: validation, logging, safe fallbacks.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .amenities import add_coords_to_df
from rest_api.utils.features import prefix

logger = logging.getLogger(__name__)

# =========================
# Config
# =========================
RARITY_FALLBACKS: Dict[str, list[list[str]]] = {
    "Maisonette": [["Maisonette"], ["Maisonette", "Flat"]],
    "Bungalow": [["Bungalow"], ["Bungalow", "Detached"]],
    "*": [["ExactType"], ["ExactType", "AnyType"]],
}
COMP_FEATURES = [
    "property_type","tenure", "built_form",
    "floor_area", "floor_level", "num_rooms", "energy_eff",
]

# =========================
# Utilities
# =========================
def haversine(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance in km between one point and arrays."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

# =========================
# Filtering & radius search
# =========================
def filter_comps(
    df: pd.DataFrame,
    q: Dict[str, Any],
    months_lookback: int,
    fa_tol: float = 0.35,
    fallback_stage: int = 0,
) -> pd.DataFrame:
    """Filter comps by property type fallback, date, and floor area."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    cutoff = datetime.today() - timedelta(days=30 * months_lookback)
    fa = float(q.get("floor_area", np.nan))

    fallbacks = RARITY_FALLBACKS.get(q.get("property_type"), RARITY_FALLBACKS["*"])
    allowed = fallbacks[fallback_stage] if fallback_stage < len(fallbacks) else ["AnyType"]

    if "AnyType" not in allowed:
        if "ExactType" in allowed:
            df = df[df["property_type"] == q.get("property_type")]
        else:
            df = df[df["property_type"].isin(allowed)]

    df = df[
        (df["date"] >= cutoff) &
        (df["floor_area"].between(fa * (1 - fa_tol), fa * (1 + fa_tol)))
    ].dropna()

    # Remove identical property (same postcode, area, etc.)
    same = (
        (df.get("postcode", "") == q.get("postcode", "")) &
        (df["floor_area"] == fa) &
        (df.get("energy_eff", np.nan) == q.get("energy_eff", np.nan)) &
        (df["property_type"] == q.get("property_type")) &
        (df.get("built_form", "") == q.get("built_form", ""))
    )
    return df[~same]


def expand_radius(
    df: pd.DataFrame,
    qlat: float,
    qlon: float,
    init_km: float,
    max_km: float,
    step_km: float,
    prefix: str | None = None,
    min_needed: int = 15,
) -> Tuple[pd.DataFrame, float]:
    """Expand radius until min comps found. Prefer same prefix if given."""
    best, radius = pd.DataFrame(), init_km
    while radius <= max_km:
        sub = df[df["prefix"] == prefix] if prefix else df
        if not sub.empty:
            dkm = haversine(qlat, qlon, sub["lat"].to_numpy(), sub["lon"].to_numpy())
            sub = sub.loc[dkm <= radius].copy()
            sub["geo_dist_km"] = dkm[dkm <= radius]
            if len(sub) > len(best):
                best = sub
            if len(sub) >= min_needed:
                return sub, radius
        radius += step_km
    return best, min(radius, max_km)


def compute_similarity_pct(raw_w: np.ndarray, mode: str = "calibrated") -> np.ndarray:
    """
    Turn raw kernel weights into user-facing similarity percentages.

    Parameters
    ----------
    raw_w : np.ndarray
        Raw kernel weights (before normalization/capping).
    mode : str
        "relative"   → top comp = 100%, others scaled down
        "absolute"   → direct %, usually low values (truthful but compressed)
        "calibrated" → softened absolute (default), uses gamma to spread values

    Returns
    -------
    np.ndarray of similarity percentages (0–100, rounded to 1dp).
    """
    if len(raw_w) == 0:
        return np.array([])

    if mode == "relative":
        w_max = float(raw_w.max())
        score = (raw_w / (w_max or 1.0)) * 100.0

    elif mode == "absolute":
        score = raw_w * 100.0

    elif mode == "calibrated":
        gamma = 0.1  # <1 spreads values upward
        score = (raw_w ** gamma) * 100.0
        score = np.clip(score, 5, 98)  # cosmetic floor/ceiling

    else:
        raise ValueError(f"Unknown mode {mode}")

    return np.round(score, 1)


# =========================
# KNN with weights
# =========================
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def knn_weighted_price(
    df: pd.DataFrame,
    q: Dict[str, Any],
    preprocessor: Any,
    k: int,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Compute weighted KNN price per m² using:
      - hybrid kernel of feature & geo distances
      - dynamic cap for very close comps
      - feature-similarity floor to prevent bad matches dominating
    """

    def cap_and_redistribute_dynamic(
        w: np.ndarray,
        geo_dist: np.ndarray,
        base_cap: float = 0.25,
        near_cap: float = 0.5,
        near_thresh: float = 0.05,
        iters: int = 3,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Ensure no comp exceeds cap (dynamic: looser for very near comps)."""
        w = np.clip(w, 0, None)
        w = w / (w.sum() + eps)
        cap = np.where(geo_dist <= near_thresh, near_cap, base_cap)
        for _ in range(iters):
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap[over]).sum()
            w[over] = cap[over]
            rest = ~over
            share = w[rest].sum()
            if share < eps:
                w[rest] += excess / max(rest.sum(), 1)
            else:
                w[rest] += excess * (w[rest] / share)
            w = np.clip(w, 0, 1)
            w = w / (w.sum() + eps)
        return w

    fa = float(q["floor_area"])
    df = df.copy()
    df["price_per_m2"] = df["price"] / df["floor_area"]
    df = df.dropna(subset=["price_per_m2", "lat", "lon"])

    if df.empty:
        raise ValueError("No valid comps with price_per_m2 and coordinates")

    # Encode features
    preprocessor.fit(df[COMP_FEATURES])
    Xq = preprocessor.transform(pd.DataFrame([q])[COMP_FEATURES])
    Xc = preprocessor.transform(df[COMP_FEATURES])

    # KNN in feature space
    n_k = min(k, len(df))
    nn = NearestNeighbors(n_neighbors=n_k).fit(Xc)
    feat_dist, idx = nn.kneighbors(Xq, n_neighbors=n_k)

    comps = df.iloc[idx[0]].copy()
    comps["feature_distance"] = feat_dist[0]

    # Geo distances
    if "geo_dist_km" not in comps.columns:
        comps["geo_dist_km"] = haversine(
            q["lat"], q["lon"], comps["lat"].to_numpy(), comps["lon"].to_numpy()
        )

    # Scaling
    eps = 1e-6
    f_scale = np.percentile(comps["feature_distance"], 75) + eps
    g_scale = np.percentile(comps["geo_dist_km"], 75) + eps
    lam_f, lam_g = 1.0 / f_scale, 2.0 / g_scale

    # Kernels
    w_feat = np.exp(-lam_f * comps["feature_distance"] ** 2)
    w_geo = np.exp(-lam_g * comps["geo_dist_km"] ** 2)

    # Hybrid weighting
    alpha = 0.6  # tune between feature (α→1) vs geo (α→0)
    w = (w_feat ** alpha) * (w_geo ** (1 - alpha))

    # Boost very close comps
    w *= np.where(comps["geo_dist_km"] <= 0.05, 2, 1.0)

    # Feature-similarity floor: downweight very dissimilar comps
    far_feat = comps["feature_distance"] > np.percentile(comps["feature_distance"], 90)
    w[far_feat] *= 0.5

    # Normalize & apply dynamic cap
    w = cap_and_redistribute_dynamic(
        w / (w.sum() + eps), comps["geo_dist_km"].to_numpy()
    )
    comps["weight"] = w

    # Final price per m²
    ppm2 = float((w * comps["price_per_m2"]).sum())
    comps = comps.sort_values("weight", ascending=False).reset_index(drop=True)

    return comps, ppm2, float(ppm2 * fa)


# =========================
# Main API
# =========================
def get_knn_comps(
    q: Dict[str, Any],
    df: pd.DataFrame,
    preprocessor: Any,
    k_neighbours: int = 15,
    min_same_prefix: int = 10,
    show_top: int = 10,
    months_lookback: int = 12,
    initial_radius_km: float = 0.4,
    max_radius_km: float = 1.0,
    radius_step_km: float = 0.2,
) -> Dict[str, Any]:
    """Main function to get comps with fallbacks and radius expansion."""
    best_comps, best_radius, stage_used = pd.DataFrame(), np.nan, -1
    qlat, qlon = float(q["lat"]), float(q["lon"])
    qprefix = q.get("prefix") or prefix(q["postcode"])

    for stage in range(3):  # strict → relaxed → any type
        filtered = filter_comps(df, q, months_lookback, fallback_stage=stage)

        local_same, r_same = expand_radius(
            filtered, qlat, qlon, initial_radius_km, max_radius_km,
            radius_step_km, prefix=qprefix, min_needed=k_neighbours
        )
        if len(local_same) >= min_same_prefix:
            best_comps, best_radius, stage_used = local_same, r_same, stage
            break

        local_mixed, r_mixed = expand_radius(
            filtered, qlat, qlon, initial_radius_km, max_radius_km,
            radius_step_km, prefix=None, min_needed=k_neighbours
        )
        local, radius = (local_mixed, r_mixed) if len(local_mixed) > len(local_same) else (local_same, r_same)

        if len(local) > len(best_comps):
            best_comps, best_radius, stage_used = local, radius, stage

    if best_comps.empty:
        raise ValueError(f"No comps found within {max_radius_km} km.")

    comps_sorted, ppm2, price = knn_weighted_price(best_comps, q, preprocessor, k_neighbours)
    quality = {
        "num_comps": int(len(comps_sorted)),
        "radius_used_km": float(best_radius),
        "avg_geo_km": float(comps_sorted["geo_dist_km"].mean()),
        "max_geo_km": float(comps_sorted["geo_dist_km"].max()),
        "avg_feat_dist": float(comps_sorted["feature_distance"].mean()),
        "pp_m2_std": float(comps_sorted["price_per_m2"].std(ddof=0)),
        "prefix_share": float((comps_sorted.get("prefix") == qprefix).mean()),
        "fallback_stage_used": stage_used,
    }
    display_cols = ['full_address', 'city', 'date', 'price', 'price_per_m2', 'property_type', 'built_form', 'energy_eff', 'num_rooms', 'floor_area', 'floor_level', 'geo_dist_km', 'lat', 'lon']
    return {
        "estimated_price": round(price, 2),
        "estimated_price_per_m2": round(ppm2, 2),
        "quality": quality,
        "comps": comps_sorted.to_dict(orient="records"),
        "display_comps": comps_sorted[display_cols].head(show_top).to_dict(orient="records"),
    }

# =========================
# Entry point
# =========================
def find_similar_properties(
    query_dict: Dict[str, Any],
    preprocessor: Any,
    training_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Public API: enrich query with coords and find comps."""
    try:
        enriched_query = add_coords_to_df(pd.DataFrame([query_dict]), postcode_col="postcode")
        if pd.isna(enriched_query.iloc[0].get("lat")) or pd.isna(enriched_query.iloc[0].get("lon")):
            raise ValueError("Could not enrich address with coordinates.")
        return get_knn_comps(enriched_query.iloc[0].to_dict(), training_df, preprocessor)
    except Exception:
        logger.exception("Failed to find similar properties")
        raise
