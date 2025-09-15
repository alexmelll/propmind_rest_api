import pandas as pd
from rest_api.db.accessors.features_mapping_accessors import get_feature_mapping_from_db

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    tenure_map = get_feature_mapping_from_db('tenure')
    property_type_map = get_feature_mapping_from_db('property_type')
    built_form_map = get_feature_mapping_from_db('built_form')

    df['tenure'] = df['tenure'].map(tenure_map)
    df['property_type'] = df['property_type'].map(property_type_map)
    df['built_form'] = df['built_form'].map(built_form_map)

    return df