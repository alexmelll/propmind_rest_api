import pandas as pd
import numpy as np
import math
from datetime import date, timedelta
from typing import Dict, Any, Iterable, List
from rest_api.db.accessors.raw_transaction_stats_accessors import get_raw_transaction_stats
from rest_api.utils.features import prefix
from rest_api.services.api import ModelDeps
from rest_api.services.knn import find_similar_properties
from rest_api.services.prediction import preprocess_input, run_prediction
from rest_api.schemas.output_data_schemas import ExplainResponse
from rest_api.db.accessors.borrow_rates_accessors import get_borrow_rates
from rest_api.db.accessors.rent_stats_accessors import get_rent_stats
from rest_api.services.llm.llm_helper import explain_price_factors

def get_mortgage_data(rate: pd.Series, price: float) -> Dict[str, Any]:
    return {
        "rate_pct": {'2y': rate['rate_2y'], '5y': rate['rate_5y']},  # the rate to test (%)
        "property_price": price,
        "maturities_years": [20, 25, 30, 35],
        "upfront_percents": [10, 15, 20, 25, 30]
    }

def classify_confidence(fallback_stage_used: int, pp_m2_std: float) -> str:
    if fallback_stage_used == 0 and pp_m2_std < 1000:
        return "high"
    if fallback_stage_used in {0, 1} and 1000 <= pp_m2_std <= 1500:
        return "medium"
    return "low"

def aggregate_shap_for_chart_and_llm(
    shap_values: Dict[str, Any],
    pred_price: float,
    base_value: float,
    min_abs: float = 1000.0,
    detail_groups: Iterable[str] = (),
    with_details: bool = False,
) -> Dict[str, Any]:
    """
    Aggregate SHAP values into:
      - numeric_groups: grouped impacts (excl. negligible) + residual so totals reconcile
        Each group has a short human description instead of raw feature names.
      - qualitative_factors: list of non-numeric features with labels
    """

    groups: Dict[str, set] = {
        "location": {"lat", "lon", "sector_ppm2_prior"},
        "energy efficiency": {"energy_eff", "epc", "energy_rating"},
        "property style": {"tenure", "property_type", "built_form", "property_age", "floor_level"},
        "distance to amenities": {"dist_to_tube", "dist_to_park", "dist_to_school"},
        "space layout": {"floor_area", "num_rooms", "room_density"},
        "seasonality": {"year", "month"},
    }

    group_descriptions: Dict[str, str] = {
        "location": "Driven by how desirable the postcode and local area are",
        "energy efficiency": "Reflects the property’s EPC rating and energy costs",
        "property style": "Captures the property’s type, build style, age, and floor position",
        "distance to amenities": "Proximity to public transport, parks, and schools",
        "space layout": "How the size and room layout affect usability and value",
        "seasonality": "Influence of market timing and seasonal conditions",
        "residual_factors": "Other smaller influences",
    }

    def norm(s: str) -> str:
        return s.strip().lower()

    feat_to_group: Dict[str, str] = {
        norm(feat): g for g, feats in groups.items() for feat in feats
    }

    totals: Dict[str, float] = {g: 0.0 for g in groups.keys()}
    qualitative_factors: List[Dict[str, str]] = []
    label_features: List[str] = []

    for feat_name, val in shap_values.items():
        if isinstance(val, dict):
            if val.get("type") == "value":
                v = float(val["impact"])
            else:
                label_features.append(str(feat_name))
                qualitative_factors.append(
                    {"feature": str(feat_name), "label": val.get("impact", "neutral")}
                )
                continue
        else:
            try:
                v = float(val)
            except Exception:
                label_features.append(str(feat_name))
                qualitative_factors.append({"feature": str(feat_name), "label": "neutral"})
                continue

        g = feat_to_group.get(norm(str(feat_name)))
        if not g:
            continue

        if norm(str(feat_name)) == "property_age":
            # keep numeric contribution but override qualitative label
            totals[g] += v
            try:
                age_val = float(val)
                year_built = 2025 - age_val
                if year_built < 1919 or year_built >= 2016:
                    qualitative_factors.append({"feature": "property_age", "label": "positive"})
                else:
                    qualitative_factors.append(
                        {"feature": "property_age", "label": "positive" if v >= 0 else "negative"}
                    )
            except Exception:
                qualitative_factors.append({"feature": "property_age", "label": "neutral"})
            continue

        totals[g] += v

    numeric_groups: Dict[str, Any] = {}
    residual_val = 0.0

    for g, total in totals.items():
        if abs(total) < min_abs:
            residual_val += total
            continue

        numeric_groups[g] = {
            "impact": total,
            "description": group_descriptions.get(g, g),
        }

    # --- Add residual so totals reconcile ---
    grouped_sum = sum(v["impact"] for v in numeric_groups.values())
    total_needed = pred_price - base_value
    residual_val += total_needed - grouped_sum

    numeric_groups["residual_factors"] = {
        "impact": residual_val,
        "description": group_descriptions["residual_factors"],
    }

    return {
        "numeric_groups": numeric_groups,
        "qualitative_factors": qualitative_factors,
    }




def get_yield_stats(num_rooms: float, prefix: str, pred_price: float):
    # Get rent stats (monthly, e.g. p25_pcm, p75_pcm)
    rent_stats = get_rent_stats(
        date.today() - timedelta(days=180),
        num_rooms,
        prefix,
        columns=["p25_pcm", "p75_pcm"]
    )

    # Copy to avoid overwriting original pcm values
    df = rent_stats.copy()

    # Add yields (annualised rent / predicted price)
    df["p25_yield"] = df["p25_pcm"] * 12 / pred_price
    df["p75_yield"] = df["p75_pcm"] * 12 / pred_price

    return df.to_dict(orient="records")


def get_geo_stats(data: pd.DataFrame, geo: str, min_date: date, floor_area: float):
    median_cols =  ['date', 'median_ppm2_6m', "p25_ppm2_6m", "p75_ppm2_6m"]

    prefix_trend = get_raw_transaction_stats(geography=geo, min_date=min_date, name='6m')[median_cols]
    prefix_trend[['median_price_6m', 'p25_price_6m', 'p75_price_6m']] =  prefix_trend[['median_ppm2_6m', 'p25_ppm2_6m', 'p75_ppm2_6m']] * floor_area

    city_trend = get_raw_transaction_stats(geography='LONDON', min_date=min_date, name='6m')[median_cols]
    city_trend[['median_price_6m', 'p25_price_6m', 'p75_price_6m']] =  city_trend[['median_ppm2_6m', 'p25_ppm2_6m', 'p75_ppm2_6m']] * floor_area


    data['prefix_trend'] = prefix_trend.to_dict('records')
    data['city_trend'] = city_trend.to_dict('records')

    return data

def sale_likelihood_pct(pred_price: float,
                        pct_shift: float,
                        floor: float = 0.10,
                        ceil: float = 0.90,
                        k_up: float = 7.9,
                        k_down: float = 14.7) -> float:
    delta = float(pct_shift) / 100.0
    x = -delta
    k = float(k_down if delta < 0 else k_up)
    p = floor + (ceil - floor) / (1.0 + math.exp(-k * x))
    return round(100.0 * max(floor, min(ceil, p)), 1)

def sale_likelihood_curve(
    pred_price: float,
    pct_min: float = -15,
    pct_max: float = 15,
    step: float = 5.0,
    **kwargs
) -> List[Dict[str, float]]:
    """
    Generate (price, pct_shift, likelihood) points for plotting a sale likelihood curve.

    Parameters
    ----------
    pred_price : float
        Predicted fair price of the property.
    pct_min, pct_max : float
        Range of percentage shifts to explore (e.g. -20 .. +20).
    step : float
        Step size in percent between points.
    kwargs : passed through to sale_likelihood_pct (e.g. custom k_up/k_down).

    Returns
    -------
    List[Dict[str, float]] with keys:
        'pct_shift'       -> % change from predicted price
        'price'           -> actual price (£)
        'likelihood_pct'  -> sale likelihood %
    """
    pts = []
    pct = pct_min
    while pct <= pct_max + 1e-9:
        price = round(pred_price * (1 + pct / 100.0), 0)
        likelihood = sale_likelihood_pct(pred_price, pct, **kwargs)
        pts.append({
            "pct_shift": round(pct, 1),
            "price": price,
            "likelihood_pct": likelihood,
        })
        pct += step
    return pts

def get_epc_scenario_res(X: pd.DataFrame, clf: Any, regressors: Dict[str, Any]) -> Dict[str, Any]:
    res = {}
    X_bump = X.copy()
    X_bump['energy_eff'] = np.where(X_bump['energy_eff'] < 75, X_bump['energy_eff'] + 20 , X_bump['energy_eff'] + 15)
    res['epc_bump'] = X_bump['energy_eff'].iloc[0]
    pred = run_prediction(X_bump, clf, regressors)
    res['price_bump'] = pred['pred_price']

    return res


def get_report_data(request_body: dict, training_df: pd.DataFrame, deps: ModelDeps) -> ExplainResponse:
    similar = find_similar_properties(
        request_body,
        training_df=training_df,
        preprocessor=deps.preprocessor,
    )
    # Preprocess request & background with same preprocessor/features
    feature_names = getattr(deps.clf, "feature_names_in_", None)
    X = preprocess_input(pd.DataFrame([request_body]), feature_names)
    request_body['lat'] = X['lat'].iloc[0]
    request_body['lon'] = X['lon'].iloc[0]
    shap_background = preprocess_input(pd.DataFrame(similar["comps"]), feature_names, to_enrich=False)
    shap_background = {"comps": shap_background, "estimated_price": similar["estimated_price"]}

    one_year = date.today() - timedelta(days=365)
    pred = run_prediction(X, deps.clf, deps.regressors, shap_background, return_shap=True)
    pred['epc_scenario'] = get_epc_scenario_res(X, deps.clf, deps.regressors)
    pred['shap_values'] = aggregate_shap_for_chart_and_llm(pred['shap_values'], pred['pred_price'], pred['base_value'])
    pred['property_info'] = request_body
    pred['display_comps'] = similar['display_comps']
    pred['total_num_comps'] = similar['quality']['num_comps']
    borrow_rates = get_borrow_rates(min_date=one_year)
    pred['sale_likelihood'] = sale_likelihood_curve(pred['pred_price'])
    pred['borrow_rates'] = borrow_rates.to_dict('records')
    pred["mortgage_planner"] = get_mortgage_data(borrow_rates.iloc[-1], pred['pred_price'])
    pred['comps_confidence'] = classify_confidence(similar['quality']['fallback_stage_used'], similar['quality']['pp_m2_std'])
    pref = prefix(request_body['postcode'])
    pred['rent_stats'] = get_yield_stats(request_body['num_rooms'], pref, pred['pred_price'])
    pred['prefix_trend'] = \
        get_raw_transaction_stats(geography=pref,
                                  min_date=one_year, name='6m')[
            ['date', 'median_ppm2_6m']].to_dict('records')
    # pred['city_trend'] = \
    #     get_raw_transaction_stats(geography='LONDON', min_date=date.today() - timedelta(days=365))[
    #         ['date', 'median_ppm2_3m']].to_dict('records')    pred = make_numbers_nice(pred)
    pred['nlp_analysis'] = explain_price_factors(pred)
    floor_area = pred['property_info']['floor_area']
    pred = get_geo_stats(data=pred, geo=pref, min_date=date.today() - timedelta(days=365), floor_area=floor_area)

    return pred