from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Date, Numeric, Column, String

class Base(DeclarativeBase): pass

class RentStats(Base):
    __tablename__ = "rent_stats"
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, primary_key=True, nullable=False)
    prefix = Column(String(5), primary_key=True, nullable=False)
    num_rooms = Column(Numeric(3, 0), nullable=False)
    mean_pcm = Column(Numeric(5, 0))
    median_pcm = Column(Numeric(5, 0))
    p25_pcm = Column(Numeric(5, 0))
    p75_pcm = Column(Numeric(5, 0))

    rate_2y = Column(Numeric(4, 2))
    rate_5y = Column(Numeric(4, 2))
