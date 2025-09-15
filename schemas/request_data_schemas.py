from pydantic import BaseModel

class AddressInput(BaseModel):
    address: str

class PredictRequest(BaseModel):
    full_address: str
    city: str
    postcode: str
    built_date: str
    energy_eff: int
    tenure: str
    property_type: str
    built_form: str
    floor_area: float
    floor_level: float
    num_rooms: int
    dist_to_park: float
    dist_to_tube: float
    dist_to_school: float

class KnnRequest(BaseModel):
    postcode: str
    built_form: str
    energy_eff: int
    tenure: str
    property_type: str
    built_form: str
    floor_area: float
    floor_level: float
    num_rooms: int
