from pydantic import BaseModel
from typing import Optional, Union, Dict, List, Any

class BuiltDate(BaseModel):
    type: str
    label: str
    start_year: Optional[int]
    end_year: Optional[int]
    exact_year: Optional[int]

class EnrichedResponse(BaseModel):
    matched: bool
    address: str
    postcode: str
    built_date: Optional[BuiltDate] = None
    energy_eff: Optional[float] = None
    tenure: Optional[str] = None
    property_type: Optional[str] = None
    built_form: Optional[str] = None
    floor_area: Optional[Union[float, int]] = None
    floor_level: Optional[Union[float, int]] = None
    num_rooms: Optional[float] = None
    dist_to_park: float
    dist_to_tube: float
    dist_to_school: float

class PredictResponse(BaseModel):
    pred_price: float
    pred_ppm2: float
    price_low: float
    price_high: float

class SimilarPropertiesResponse(BaseModel):
    estimated_price: float
    estimated_price_per_m2: float
    comps: List[Dict[str, Any]]
    comps_quality: Dict[str, Any]
    display_comps: List[Dict[str, Any]]

class ExplainResponse(PredictResponse):
    report_url: str
    base_value: float
    residuals: float
    shap_values: Dict[str, Any]
    display_comps: List[Dict[str, Any]]
    comps_confidence: str
    prefix_trend: List[Dict[Any, Any]]
    city_trend: List[Dict[Any, Any]]
    nlp_analysis: str
    property_info: Dict[str, Any]