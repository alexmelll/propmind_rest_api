import pandas as pd
from datetime import date
from typing import List
from rest_api.db.db_saver import save_dataframe_to_table
from sqlalchemy import select
from rest_api.db.db_session import session_scope
from rest_api.db.schemas.rent_stats_schema import RentStats

def get_rent_stats(
    min_date: date,
    num_rooms: float,
    prefix: str,
    columns: List = None,
) -> pd.DataFrame:
    """
    Fetch 3-month rolling price stats from public.price_stats_3m.

    Args:
        min_date: lower bound for date filter (inclusive).

    Returns:
        pandas.DataFrame with results.
    """
    with session_scope() as session:
        if columns:
            bad = [c for c in columns if c not in RentStats.__table__.c]
            if bad:
                raise ValueError(f"Unknown columns requested: {bad}")
            cols = [RentStats.__table__.c[c] for c in columns]
            stmt = select(*cols)
        else:
            stmt = select(RentStats)

        if min_date is not None:
            stmt = stmt.where(RentStats.period_end >= min_date)
            stmt = stmt.where(RentStats.num_rooms == num_rooms)
            stmt = stmt.where(RentStats.prefix == prefix)
        # pandas can read directly from the selectable
        df = pd.read_sql(stmt, session.bind)
        return df


def save_rent_stats_to_db(df: pd.DataFrame) -> None:
    save_dataframe_to_table(df, RentStats)
