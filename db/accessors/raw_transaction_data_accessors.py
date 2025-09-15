import pandas as pd
from rest_api.db.db_session import session_scope
from rest_api.db.db_saver import save_dataframe_to_table
from rest_api.db.schemas.raw_transaction_data_schema import RawTransactions
from datetime import date, datetime
from typing import Optional, Sequence, Union, List
from sqlalchemy import select

# import your session_scope and model
# from rest_api.db.session import session_scope
# from rest_api.db.schemas.raw_transaction_data_schema import RawTransactions

# Optional: restrict which fields are allowed in `field=` filters
FILTER_COLS_TX = {
    "full_address": RawTransactions.full_address,
    "date": RawTransactions.date,
    "price": RawTransactions.price,
    "city": RawTransactions.city,
    "property_type": RawTransactions.property_type,
    "tenure": RawTransactions.tenure,
    # add more whitelisted filterables as needed
}

def save_reduced_epc_to_db(df: pd.DataFrame) -> None:
    save_dataframe_to_table(df, RawTransactions)

def get_transaction_data(
    field: Optional[str] = None,
    value: Optional[Union[str, int, float, Sequence[Union[str, int, float]]]] = None,
    min_date: Optional[Union[date, datetime]] = None,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    order_desc_by_date: bool = False,
    as_dataframe: bool = True,
) -> Union[pd.DataFrame, List[RawTransactions]]:
    """
    Fetch rows from public.raw_transactions with optional filters.

    Args
    ----
    field/value : optional equality or IN filter (whitelisted via FILTER_COLS_TX)
    min_date    : lower bound on RawTransactions.date
    columns     : list of column names to select (strings). Default: all columns
    limit       : LIMIT N
    order_desc_by_date : order by date DESC
    as_dataframe : True -> return pandas.DataFrame (default)
                   False -> return list[RawTransactions] ORM instances

    Returns
    -------
    pandas.DataFrame or list[RawTransactions]
    """
    with session_scope() as session:
        # ----- columns to select -----
        if columns:
            # validate column names exist on the mapped table
            tbl_cols = RawTransactions.__table__.c
            bad = [c for c in columns if c not in tbl_cols]
            if bad:
                raise ValueError(f"Unknown columns requested: {bad}")
            sel_cols = [tbl_cols[c] for c in columns]
            stmt = select(*sel_cols)
        else:
            stmt = select(RawTransactions)

        # ----- filters -----
        if min_date is not None:
            stmt = stmt.where(RawTransactions.date >= min_date)

        if field is not None and value is not None:
            if field not in FILTER_COLS_TX:
                raise ValueError(f"Unsupported filter field: {field}")
            col = FILTER_COLS_TX[field]
            if isinstance(value, (list, tuple, set)):
                if len(value) == 0:
                    # return empty result with correct columns
                    if as_dataframe:
                        out_cols = columns or [c.name for c in RawTransactions.__table__.columns]
                        return pd.DataFrame(columns=out_cols)
                    else:
                        return []
                stmt = stmt.where(col.in_(list(value)))
            else:
                stmt = stmt.where(col == value)

        if order_desc_by_date:
            stmt = stmt.order_by(RawTransactions.date.desc())

        if limit is not None:
            stmt = stmt.limit(limit)

        # ----- execute -----
        if as_dataframe:
            # pandas can read a SQLAlchemy selectable with the same engine/connection
            df = pd.read_sql(stmt, session.bind)
            return df
        else:
            # ORM instances
            return session.execute(stmt).scalars().all()
