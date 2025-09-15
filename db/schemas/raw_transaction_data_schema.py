from sqlalchemy import Column, Numeric, String, Date
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RawTransactions(Base):
    __tablename__ = "raw_transactions"
    __table_args__ = {"schema": "public"}

    full_address   = Column(String(1000), nullable=False, primary_key=True)
    date           = Column(Date, nullable=False, primary_key=True)
    epc_date       = Column(Date, nullable=True)
    price          = Column(Numeric(11, 0), nullable=False)
    old_new        = Column(String(2), nullable=True)
    tenure         = Column(String(10), nullable=True)
    energy_eff     = Column(Numeric(3, 0), nullable=True)
    property_type  = Column(String(20), nullable=True)
    built_form     = Column(String(20), nullable=True)
    num_rooms      = Column(Numeric(3, 0), nullable=True)
    age_band       = Column(String(50), nullable=True)
    floor_area     = Column(Numeric(4, 0), nullable=True)
    city           = Column(String(500), nullable=False)
    floor_level = Column(Numeric(2, 0))
