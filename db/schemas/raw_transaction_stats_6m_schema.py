from sqlalchemy import Column, String, Date, Numeric
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RawTransactionStats6m(Base):
    __tablename__ = "raw_transaction_stats_6m"
    __table_args__ = {"schema": "public"}

    geography = Column(String(10), primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)

    num_price = Column(Numeric(8, 0))
    num_ppm2 = Column(Numeric(8, 0))

    median_price_6m = Column(Numeric(12, 0))
    median_ppm2_6m = Column(Numeric(9, 2))

    p10_price_6m = Column(Numeric(12, 0))
    p10_ppm2_6m = Column(Numeric(9, 2))
    p25_price_6m = Column(Numeric(12, 0))
    p25_ppm2_6m = Column(Numeric(9, 2))
    p75_price_6m = Column(Numeric(12, 0))
    p75_ppm2_6m = Column(Numeric(9, 2))
    p90_price_6m = Column(Numeric(12, 0))
    p90_ppm2_6m = Column(Numeric(9, 2))

    std_price_6m = Column(Numeric(12, 0))
    std_ppm2_6m = Column(Numeric(9, 2))

    mom_pct_price = Column(Numeric(7, 4))
    yoy_pct_price = Column(Numeric(7, 4))
    mom_pct_ppm2 = Column(Numeric(7, 4))
    yoy_pct_ppm2 = Column(Numeric(7, 4))

# from rest_api.db.db_session import engine
#
# Base.metadata.create_all(engine, tables=[RawTransactionStats6m.__table__])
