"""ORM declarative base (SQLAlchemy 2.0 style, spec D3)."""

from sqlalchemy import BigInteger, Integer
from sqlalchemy.orm import DeclarativeBase

# DDL 用 BIGINT UNSIGNED;SQLite 上 BIGINT PRIMARY KEY 不是 rowid 别名、不会自增,
# 必须回落 Integer(spec D8 双库跑法的方言适配点)。
# 注意:unsigned 是 MySQL 方言专属 kwarg,通用 BigInteger(unsigned=True) 会直接
# TypeError(SQLAlchemy 2.0 实测);MySQL 建表走 scripts/apply_schema.py 原样执行
# ch02_schema.sql,ORM 侧只保证双库可跑,不重复 DDL 细节。
BigInt = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    pass
