from typing import Optional, Union, Sequence
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from datetime import date, datetime

from rest_api.db.db_session import session_scope
from rest_api.db.schemas.training_data_schema import TrainingData
from rest_api.db.db_saver import save_dataframe_to_table

# Map *allowed* filter fields to model columns (prevents SQL injection)
FILTER_COLS = {
    "postcode": TrainingData.postcode,
    "prefix": TrainingData.prefix,
    "sector": TrainingData.sector,
    "date": TrainingData.date,
    "full_address": TrainingData.full_address,
    # add others you actually support
}

def get_training_data(
    field: Optional[str] = None,
    value: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
    min_date: Optional[Union[date, datetime]] = None,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    order_desc_by_date: bool = False,
) -> pd.DataFrame:
    """
    Safely fetch rows from training_data with optional filters.
    - field/value: equality or IN filter over a whitelisted column
    - min_date: lower bound on TrainingData.date
    - columns: restrict selected columns (default: all)
    - limit: cap rows
    - order_desc_by_date: True to ORDER BY date DESC
    """
    with session_scope() as session:  # type: Session
        # columns
        if columns:
            bad = [c for c in columns if c not in TrainingData.__table__.c]
            if bad:
                raise ValueError(f"Unknown columns requested: {bad}")
            cols = [TrainingData.__table__.c[c] for c in columns]
            stmt = select(*cols)
        else:
            stmt = select(TrainingData)

        # filters
        if min_date is not None:
            stmt = stmt.where(TrainingData.date >= min_date)

        if field is not None and value is not None:
            if field not in FILTER_COLS:
                raise ValueError(f"Unsupported filter field: {field}")
            col = FILTER_COLS[field]
            if isinstance(value, (list, tuple, set)):
                if len(value) == 0:
                    return pd.DataFrame(columns=columns or [c.name for c in TrainingData.__table__.columns])
                stmt = stmt.where(col.in_(list(value)))
            else:
                stmt = stmt.where(col == value)

        if order_desc_by_date:
            stmt = stmt.order_by(TrainingData.date.desc())

        if limit is not None:
            stmt = stmt.limit(limit)

        # Execute; pandas can read from a SQLAlchemy selectable
        df = pd.read_sql(stmt, session.bind)
        return df

def save_reduced_epc_to_db(df: pd.DataFrame) -> None:
    save_dataframe_to_table(df, TrainingData)