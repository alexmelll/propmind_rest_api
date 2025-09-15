import pandas as pd
from datetime import date
from rest_api.db.db_saver import save_dataframe_to_table
from sqlalchemy import select
from rest_api.db.db_session import session_scope
from rest_api.db.schemas.borrow_rate_schema import BorrowRates

def get_borrow_rates(
    min_date: date,
) -> pd.DataFrame:
    """
    Fetch 3-month rolling price stats from public.price_stats_3m.

    Args:
        min_date: lower bound for date filter (inclusive).

    Returns:
        pandas.DataFrame with results.
    """
    with session_scope() as session:
        stmt = select(BorrowRates)

        if min_date is not None:
            stmt = stmt.where(BorrowRates.date >= min_date)

        # pandas can read directly from the selectable
        df = pd.read_sql(stmt, session.bind)
        return df


def save_borrow_rates_to_db(df: pd.DataFrame) -> None:
    save_dataframe_to_table(df, BorrowRates)
