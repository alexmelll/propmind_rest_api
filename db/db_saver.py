import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from rest_api.db.db_session import session_scope


def save_dataframe_to_table(df: pd.DataFrame, orm_class) -> None:
    """
    Generic bulk upsert into a table mapped by an ORM class.

    - df: pandas DataFrame
    - orm_class: SQLAlchemy ORM mapped class (with __table__)

    Assumes the table has a primary key (single or composite).
    On conflict, updates all non-PK columns.
    """
    if df.empty:
        return

    # normalize column names
    df = df.rename(columns=str.lower)

    # validate DataFrame columns
    table_cols = set(orm_class.__table__.columns.keys())
    bad_cols = [c for c in df.columns if c not in table_cols]
    if bad_cols:
        raise ValueError(f"Unexpected columns in DataFrame for {orm_class.__tablename__}: {bad_cols}")

    records = df.to_dict(orient="records")

    with session_scope() as session:
        table = orm_class.__table__
        stmt = pg_insert(table).values(records)

        # detect primary key(s)
        pk_cols = [col.name for col in table.primary_key]

        # update all non-PK cols on conflict
        update_cols = {c: stmt.excluded[c] for c in df.columns if c not in pk_cols}

        stmt = stmt.on_conflict_do_update(
            index_elements=pk_cols,
            set_=update_cols
        )

        session.execute(stmt)
        session.commit()
