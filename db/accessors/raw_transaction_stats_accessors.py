import pandas as pd
from rest_api.db.db_saver import save_dataframe_to_table
from sqlalchemy import select
from rest_api.db.db_session import session_scope, engine
from rest_api.db.schemas.raw_transaction_stats_schema import RawTransactionStats
from rest_api.db.schemas.raw_transaction_stats_6m_schema import RawTransactionStats6m

def get_raw_transaction_stats(
    geography: str | list[str] | None = None,
    min_date=None,
    max_date=None,
    name='3m'
) -> pd.DataFrame:
    """
    Fetch 3-month rolling price stats from public.price_stats_3m.

    Args:
        geography: single geography code (e.g. "SE1") or list of codes.
        min_date: lower bound for date filter (inclusive).
        max_date: upper bound for date filter (inclusive).

    Returns:
        pandas.DataFrame with results.
    """

    table = RawTransactionStats if name=='3m' else RawTransactionStats6m

    with session_scope() as session:
        stmt = select(table)

        if geography is not None:
            if isinstance(geography, (list, tuple, set)):
                stmt = stmt.where(table.geography.in_(geography))
            else:
                stmt = stmt.where(table.geography==geography)

        if min_date is not None:
            stmt = stmt.where(table.date >= min_date)
        if max_date is not None:
            stmt = stmt.where(table.date <= max_date)

        stmt = stmt.order_by(table.geography, table.date)

        # pandas can read directly from the selectable
        df = pd.read_sql(stmt, session.bind)
        return df


def save_raw_transaction_stats_to_db(df: pd.DataFrame, name='3m') -> None:
    table = RawTransactionStats if name == '3m' else RawTransactionStats6m
    save_dataframe_to_table(df, table)
