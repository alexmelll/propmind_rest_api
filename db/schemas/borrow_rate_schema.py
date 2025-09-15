from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Date, Numeric, Column
from datetime import date

class Base(DeclarativeBase): pass

class BorrowRates(Base):
    __tablename__ = "borrow_rates"
    date = Column(Date, primary_key=True, nullable=False)
    rate_2y = Column(Numeric(4, 2))
    rate_5y = Column(Numeric(4, 2))
