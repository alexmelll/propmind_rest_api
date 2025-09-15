from sqlalchemy import Column, String, Date, Numeric
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RawTransactionStats(Base):
    __tablename__ = "raw_transaction_stats"
    __table_args__ = {"schema": "public"}

    geography = Column(String(10), primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)

    num_price = Column(Numeric(8, 0))
    num_ppm2 = Column(Numeric(8, 0))

    median_price_3m = Column(Numeric(12, 0))
    median_ppm2_3m = Column(Numeric(9, 2))

    p10_price_3m = Column(Numeric(12, 0))
    p10_ppm2_3m = Column(Numeric(9, 2))
    p25_price_3m = Column(Numeric(12, 0))
    p25_ppm2_3m = Column(Numeric(9, 2))
    p75_price_3m = Column(Numeric(12, 0))
    p75_ppm2_3m = Column(Numeric(9, 2))
    p90_price_3m = Column(Numeric(12, 0))
    p90_ppm2_3m = Column(Numeric(9, 2))

    std_price_3m = Column(Numeric(12, 0))
    std_ppm2_3m = Column(Numeric(9, 2))

    mom_pct_price = Column(Numeric(7, 4))
    yoy_pct_price = Column(Numeric(7, 4))
    mom_pct_ppm2 = Column(Numeric(7, 4))
    yoy_pct_ppm2 = Column(Numeric(7, 4))
