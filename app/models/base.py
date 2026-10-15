"""ORM declarative base (SQLAlchemy 2.0 style, spec D3)."""

from sqlalchemy import BigInteger, Integer
from sqlalchemy.orm import DeclarativeBase

# DDL 用 BIGINT UNSIGNED;SQLite 上 BIGINT PRIMARY KEY 不是 rowid 别名、不会自增,
# 必须回落 Integer(spec D8 双库跑法的方言适配点)。
BigInt = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    pass
