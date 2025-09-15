from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Date, Numeric, BigInteger, UniqueConstraint, Index

class Base(DeclarativeBase): pass

class EPCReduced(Base):
    __tablename__ = "epc_reduced"
    full_address: Mapped[str] = mapped_column(String(100), nullable=False, primary_key=True)
    postcode: Mapped[str | None] = mapped_column(String(10))
    property_type: Mapped[str | None] = mapped_column(String(50))
    built_form: Mapped[str | None] = mapped_column(String(50))
    epc_date: Mapped[Date | None] = mapped_column(Date)
    floor_area: Mapped[float | None] = mapped_column(Numeric(4, 0))
    num_rooms: Mapped[float | None] = mapped_column(Numeric(4, 0))
    property_age: Mapped[str | None] = mapped_column(String(50))
    city: Mapped[str] = mapped_column(String(50), nullable=False)
    energy_eff: Mapped[float | None] = mapped_column(Numeric(3, 0))

    __table_args__ = (
        UniqueConstraint("full_address", name="uq_epc_full_address"),
    )
