import copy
import math

def make_numbers_nice(payload: dict, inplace: bool = False) -> dict:
    """
    Normalize numeric fields in a nested dict/list structure.

    Rules:
      - Prices: round to nearest £1,000
      - Price per m²: round to nearest £10  (keys with 'ppm2' or 'price_per_m2')
      - SHAP impacts: round to nearest £10  (payload['shap_values'][*]['impact'])
      - Distances: round to 2 decimals     (keys containing 'dist' or 'distance')

    Args:
        payload: your JSON-like dict
        inplace: if True, mutate payload; else return a deep-copied cleaned dict

    Returns:
        dict: cleaned structure
    """
    obj = payload if inplace else copy.deepcopy(payload)

    def _is_num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _round_base(x, base):
        try:
            return int(round(float(x) / base)) * base
        except Exception:
            return x

    def _round_price(x):      # £1,000
        return _round_base(x, 1000)

    def _round_ppm2(x):       # £10
        return _round_base(x, 10)

    def _round_shap(x):       # £10
        return _round_base(x, 10)

    def _round_dist(x):       # 2 decimals
        try:
            return float(f"{float(x):.2f}")
        except Exception:
            return x

    # Heuristics for field types
    PRICE_KEYS_EXACT = {
        "pred_price", "price_low", "price_high", "base_value", "price",
        "median_price", "p25_price", "p75_price",
        "median_price_3m", "p25_price_3m", "p75_price_3m",
    }
    PPM2_KEYS_EXACT = {"price_per_m2"}
    # anything containing these substrings is considered ppm2 €/£ per m²
    PPM2_SUBSTRINGS = ("ppm2",)  # matches median_ppm2_3m, p25_ppm2_3m, etc.

    def walk(node, path=()):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                out[k] = transform(k, v, path + (k,))
            return out
        elif isinstance(node, list):
            return [walk(v, path) for v in node]
        else:
            return node

    def transform(key, val, path):
        # SHAP impacts (payload['shap_values'][*]['impact'])
        if len(path) >= 2 and path[0] == "shap_values" and key == "impact" and _is_num(val):
            return _round_shap(val)

        # Recurse into containers
        if isinstance(val, dict) or isinstance(val, list):
            return walk(val, path)

        if not _is_num(val):
            return val

        k = key.lower()

        # Distances: any key with 'dist' or 'distance'
        if "dist" in k or "distance" in k:
            return _round_dist(val)

        # Price-per-m²
        if k in PPM2_KEYS_EXACT or any(sub in k for sub in PPM2_SUBSTRINGS):
            return _round_ppm2(val)

        # Prices
        if key in PRICE_KEYS_EXACT or key.endswith("_price_3m"):
            return _round_price(val)

        # default: leave unchanged
        return val

    return walk(obj)
