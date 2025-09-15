from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, declared_attr
from sqlalchemy import String, Date, Numeric, Integer, Boolean, UniqueConstraint, Index


class Base(DeclarativeBase):
    pass


class TrainingData(Base):   # <-- make it inherit Base
    __tablename__ = "training_data"   # <-- mapped class now
    __table_args__ = {"schema": "public"}  # optional schema

    # Core fields
    full_address: Mapped[str] = mapped_column(String(500), nullable=False, primary_key=True)
    date: Mapped[Date] = mapped_column(Date, nullable=False, primary_key=True)
    price: Mapped[float] = mapped_column(Numeric(12, 0), nullable=False)

    # Property details
    old_new: Mapped[bool | None] = mapped_column(Boolean)
    tenure: Mapped[str | None] = mapped_column(String(15))
    energy_eff: Mapped[float | None] = mapped_column(Numeric(3, 0))
    property_type: Mapped[str | None] = mapped_column(String(30))
    built_form: Mapped[str | None] = mapped_column(String(30))
    num_rooms: Mapped[float | None] = mapped_column(Numeric(4, 0))
    property_age: Mapped[float | None] = mapped_column(Numeric(4, 0))
    floor_area: Mapped[float | None] = mapped_column(Numeric(4, 0))
    floor_level: Mapped[float | None] = mapped_column(Numeric(2, 0))

    # Location info
    postcode: Mapped[str | None] = mapped_column(String(10))
    prefix: Mapped[str | None] = mapped_column(String(5))
    lat: Mapped[float | None] = mapped_column(Numeric(9, 6))
    lon: Mapped[float | None] = mapped_column(Numeric(9, 6))
    dist_to_park: Mapped[float | None] = mapped_column(Numeric(5, 2))
    dist_to_tube: Mapped[float | None] = mapped_column(Numeric(5, 2))
    dist_to_school: Mapped[float | None] = mapped_column(Numeric(5, 2))

    # Derived fields
    year: Mapped[int | None] = mapped_column(Integer)
    month: Mapped[int | None] = mapped_column(Integer)
    room_density: Mapped[float | None] = mapped_column(Numeric(6, 2))
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    price_per_m2: Mapped[float | None] = mapped_column(Numeric(10, 2))
    sector: Mapped[str | None] = mapped_column(String(6))

    @declared_attr.directive
    def __table_args__(cls):
        t = cls.__tablename__
        return (
            UniqueConstraint("full_address", "date", name=f"uq_{t}_fulladdr_date"),
            Index(f"ix_{t}_postcode", "postcode"),
            Index(f"ix_{t}_prefix", "prefix"),
            Index(f"ix_{t}_city", "city"),
            Index(f"ix_{t}_date", "date"),
            Index(
                f"ix_{t}_date_covering",
                "date",
                postgresql_include=["full_address", "price", "city"],
            ),
            {"schema": "public"},   # keep schema inside table_args
        )
