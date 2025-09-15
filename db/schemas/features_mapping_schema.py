from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Date, Numeric, BigInteger, UniqueConstraint, Index

class Base(DeclarativeBase): pass

class FeaturesMapping(Base):
    __tablename__ = "features_mapping"
    feature: Mapped[str] = mapped_column(String(50), nullable=False, primary_key=True)
    key: Mapped[str] = mapped_column(String(50), nullable=False, primary_key=True)
    value: Mapped[str] = mapped_column(String(50), nullable=False)
