import pandas as pd
from sqlalchemy import text, select
from rest_api.db.db_session import session_scope
from rest_api.db.schemas.features_mapping_schema import FeaturesMapping

def get_feature_mapping_from_db(feature: str) -> FeaturesMapping | None:
    with session_scope() as s:
        q = select(FeaturesMapping).where(FeaturesMapping.feature == feature)
        res =  s.execute(q)
        objects = res.scalars().all()  # ORM objects only
    df = pd.DataFrame([
        {col: getattr(o, col) for col in o.__table__.columns.keys()}
        for o in objects
    ])
    return dict(zip(df['key'], df['value']))