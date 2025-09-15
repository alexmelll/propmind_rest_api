from sqlalchemy import text, select
import pandas as pd
from rest_api.db.db_session import session_scope
from rest_api.db.db_saver import save_dataframe_to_table
from rest_api.db.schemas.epc_reduced_schema import EPCReduced

def save_reduced_epc_to_db(df: pd.DataFrame) -> None:
    save_dataframe_to_table(df, EPCReduced)

def get_epc_data_from_db(full_address: str) -> dict | None:
    with session_scope() as s:
        q = select(EPCReduced).where(EPCReduced.full_address == full_address)
        obj = s.execute(q).scalars().first()
        if obj is None:
            return None
        return {col: getattr(obj, col) for col in obj.__table__.columns.keys()}
