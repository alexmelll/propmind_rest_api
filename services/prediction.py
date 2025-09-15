"""
Prediction and SHAP utilities for property price models.
Production-hardened: input validation, logging, safe fallbacks.
"""

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from rest_api.services.amenities import add_coords_to_df
from rest_api.utils.features import parse_age_band, sector, prefix
from rest_api.utils.encoder import encode_data
from rest_api.db.accessors.training_data_accessors import get_training_data

logger = logging.getLogger(__name__)

# =========================
# Enrichment with sector priors
# =========================
def enrich_with_sector_prior_ppm2(
    df: pd.DataFrame,
    postcode_col: str = "postcode",
    date_col: str = "date",
    min_count_12m: int = 15,
    max_months: int = 36,
) -> pd.DataFrame:
    """
    Enrich a dataframe with sector/prefix price-per-m2 priors.
    Uses rolling medians over 12m and up to max_months horizon.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["sector"] = df[postcode_col].map(sector)
    df["prefix"] = df[postcode_col].map(prefix)

    # Load training data for prefixes in df
    hist = get_training_data(field="prefix", value=df["prefix"].dropna().unique().tolist())
    if hist.empty:
        df["sector_ppm2_prior"] = np.nan
        return df

    hist[date_col] = pd.to_datetime(hist[date_col], errors="coerce")
    hist["sector"] = hist[postcode_col].map(sector)
    hist["prefix"] = hist[postcode_col].map(prefix)
    hist["price_per_m2"] = hist["price"] / hist["floor_area"]

    days_max = int(round(max_months * 30.44))

    def _roll_stats(hist: pd.DataFrame, key: str) -> pd.DataFrame:
        g = (
            hist.dropna(subset=[key, date_col, "price_per_m2"])
            .sort_values([key, date_col])
            .reset_index(drop=True)
        )

        def add_roll(d: pd.DataFrame) -> pd.DataFrame:
            d = d.copy()
            d["med12"] = d.rolling("365D", on=date_col)["price_per_m2"].median()
            d["cnt12"] = d.rolling("365D", on=date_col)["price_per_m2"].count()
            d["medmax"] = d.rolling(f"{days_max}D", on=date_col)["price_per_m2"].median()
            d["exp"] = d["price_per_m2"].expanding().median()
            return d[[key, date_col, "med12", "cnt12", "medmax", "exp"]]

        out = g.groupby(key, group_keys=False).apply(add_roll)
        pref = key[:3]  # "sec" or "pre"
        return out.rename(
            columns={
                "med12": f"{pref}_med12",
                "cnt12": f"{pref}_cnt12",
                "medmax": f"{pref}_medmax",
                "exp": f"{pref}_exp",
            }
        )

    sec_ts = _roll_stats(hist, "sector")
    pre_ts = _roll_stats(hist, "prefix")

    df1 = pd.merge_asof(
        df.sort_values(date_col),
        sec_ts.sort_values([date_col, "sector"]),
        by="sector",
        on=date_col,
        direction="backward",
    )
    df2 = pd.merge_asof(
        df1.sort_values(date_col),
        pre_ts.sort_values([date_col, "prefix"]),
        by="prefix",
        on=date_col,
        direction="backward",
    ).sort_index()

    use_s12 = (df2["sec_cnt12"] >= min_count_12m) & df2["sec_med12"].notna()
    use_p12 = (df2["pre_cnt12"] >= min_count_12m) & df2["pre_med12"].notna()

    prior = np.where(
        use_s12,
        df2["sec_med12"],
        np.where(
            df2["sec_medmax"].notna(),
            df2["sec_medmax"],
            np.where(
                use_p12,
                df2["pre_med12"],
                np.where(df2["pre_medmax"].notna(), df2["pre_medmax"], df2["pre_exp"]),
            ),
        ),
    )
    df["sector_ppm2_prior"] = pd.to_numeric(prior, errors="coerce")
    return df


# =========================
# Preprocessing
# =========================
def preprocess_input(df: pd.DataFrame, feature_names: List[str], to_enrich: bool = True) -> pd.DataFrame:
    """Transform raw input into model-ready features."""
    X = encode_data(df)

    if to_enrich:
        X = add_coords_to_df(X, postcode_col="postcode")
        if "built_date" in X.columns:
            X["built_date"] = X["built_date"].astype(str).str.replace("–", "-", regex=False)
            build_years = X["built_date"].map(parse_age_band)
            X["property_age"] = build_years.apply(lambda x: x['exact_year'])
        today = date.today()
        X["date"] = today
        X["year"] = today.year
        X["month"] = today.month
        if "num_rooms" in X and "floor_area" in X:
            X["room_density"] = X["floor_area"] / X["num_rooms"]
        X = enrich_with_sector_prior_ppm2(X)

    X = X.reindex(columns=feature_names, fill_value=np.nan)
    return X.apply(pd.to_numeric, errors="coerce")


# =========================
# SHAP adjustment
# =========================
def compute_adj_shap_values(
    real_avg_price: float,
    pred_price: float,
    shap_values: np.ndarray,
    lam: float = 1e6,
    make_exact: bool = True,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Adjust SHAP values so they sum to pred - real.
    Ridge-stabilized scaling with cancellation-safe fallback.
    """
    s = shap_values.astype(float)
    delta = float(pred_price - real_avg_price)
    S = float(s.sum())
    L1 = float(np.abs(s).sum())

    if L1 < eps:
        return np.full_like(s, delta / max(len(s), 1)) if make_exact else np.zeros_like(s)

    tau = 0.08
    if abs(S) <= tau * L1:
        if not make_exact:
            return s
        resid = delta - S
        mask = s > 0 if resid >= 0 else s < 0
        if np.any(mask):
            L1_mask = float(np.abs(s[mask]).sum())
            w = np.zeros_like(s)
            if L1_mask > eps:
                w[mask] = np.abs(s[mask]) / L1_mask
            else:
                w[:] = 1.0 / len(s)
        else:
            w = np.abs(s) / (L1 + eps)
        return s + resid * w

    factor = (S * delta + lam) / (S * S + lam + eps)
    adj = factor * s
    if make_exact:
        resid = delta - float(adj.sum())
        w = np.abs(adj)
        wsum = float(w.sum())
        if wsum < eps:
            adj += resid / len(adj)
        else:
            adj += resid * (w / wsum)
    return adj


# =========================
# Prediction
# =========================
def adjust_shap(feature_name, value, median, shap_value, constraints_map):
    """
    Adjust SHAP values based on monotonic constraints and comparison to median comps.
    - If SHAP contradicts the expected monotonic direction, return a label instead of raw value.
    - Otherwise, return the numeric SHAP contribution.
    """
    if feature_name not in constraints_map:
        return {"type": "value", "impact": float(shap_value)}

    direction = constraints_map[feature_name]  # +1 = higher is better, -1 = lower is better

    # Define expected direction relative to comps median
    if direction == +1:   # higher is better (e.g. EPC, size)
        expected_positive = value >= median
    elif direction == -1: # lower is better (e.g. distance)
        expected_positive = value <= median
    else:
        return {"type": "value", "impact": float(shap_value)}

    # If SHAP contradicts expectation, override with label
    if expected_positive and shap_value < 0:
        return {"type": "label", "impact": "positive"}
    if not expected_positive and shap_value > 0:
        return {"type": "label", "impact": "negative"}

    return {"type": "value", "impact": float(shap_value)}


def run_prediction(
    X: pd.DataFrame,
    clf: Any,
    regressors: Dict[str, Any],
    shap_background: Optional[Dict[str, Any]] = None,
    return_shap: bool = False,
) -> Dict[str, Any]:
    """
    Blend classifier band probabilities with per-band regressors
    to produce final price predictions.
    Optionally compute SHAP values.
    """
    band_probs = clf.predict_proba(X)
    n_bands = len(regressors.get("point", {}))
    n = len(X)
    floor_area = X["floor_area"].values

    blended_mean = np.zeros(n)
    blended_lo = np.zeros(n)
    blended_hi = np.zeros(n)

    available_q = sorted(regressors.get("quantiles", {}).keys())
    if not available_q:
        raise ValueError("No quantile regressors available")
    q_low, q_high = available_q[0], available_q[-1]

    shap_value_blend = np.zeros(X.shape[1]) if return_shap else None
    base_value_blend = 0.0

    for i in range(n_bands):
        point_model = regressors["point"].get(i)
        lo_model = regressors["quantiles"][q_low].get(i)
        hi_model = regressors["quantiles"][q_high].get(i)
        if not all([point_model, lo_model, hi_model]):
            logger.warning("Skipping band %s due to missing models", i)
            continue

        ppm2_mean = point_model.predict(X)
        ppm2_lo = lo_model.predict(X)
        ppm2_hi = hi_model.predict(X)

        blended_mean += band_probs[:, i] * ppm2_mean * floor_area
        blended_lo += band_probs[:, i] * ppm2_lo * floor_area
        blended_hi += band_probs[:, i] * ppm2_hi * floor_area

        if return_shap:
            if shap_background is None:
                raise ValueError("SHAP background data required when return_shap=True")

            explainer = shap.TreeExplainer(
                point_model,
                data=shap_background["comps"][point_model.feature_names_in_],
            )
            shap_expl = explainer(X[point_model.feature_names_in_], check_additivity=False)

            shap_vals = np.asarray(shap_expl.values, dtype=float).reshape(-1)
            base_vals = np.asarray(explainer.expected_value, dtype=float)

            shap_value_blend += shap_vals * float(floor_area[0]) * float(band_probs[0, i])
            base_value_blend += base_vals * float(floor_area[0]) * float(band_probs[0, i])

    if not (np.all(blended_lo <= blended_mean) and np.all(blended_mean <= blended_hi)):
        logger.warning("Prediction interval check failed: lo ≤ mean ≤ hi not satisfied")

    if return_shap:
        real_price_avg = shap_background["estimated_price"]
        adj_shap_vals = compute_adj_shap_values(real_price_avg, blended_mean[0], shap_value_blend)
        residual = real_price_avg + sum(adj_shap_vals) - blended_mean

        constraints_map = {
            # 'sector_ppm2_prior': +1,
            'dist_to_tube': -1,
            'dist_to_school': -1,
            'dist_to_park': -1,
            'energy_eff': +1
        }

        comps = shap_background["comps"]
        safe_explanations = {}
        for feat, shap_val in zip(point_model.feature_names_in_, adj_shap_vals):
            value = X.iloc[0][feat]
            median_val = np.median(comps[feat])
            safe_explanations[feat] = adjust_shap(feat, value, median_val, shap_val, constraints_map)

        return {
            "pred_price": float(blended_mean[0]),
            "pred_ppm2": float(blended_mean[0]) / float(floor_area[0]),
            "price_low": float(blended_lo[0]),
            "price_high": float(blended_hi[0]),
            "base_value": float(real_price_avg),
            "residual_value": float(residual),
            "shap_values": safe_explanations,
        }

    return {
        "pred_price": float(blended_mean[0]),
        "pred_ppm2": float(blended_mean[0]) / float(floor_area[0]),
        "price_low": float(blended_lo[0]),
        "price_high": float(blended_hi[0]),
    }

