from sqlalchemy import select
from sqlalchemy.engine import Engine

from customer_service.db import make_engine
from customer_service.models import Base, FAQ


FAQ_SEEDS = [
    ("退货政策", "商品符合退货条件时可申请退货；具体时效和条件请以店铺实际政策为准。", "售后"),
    ("如何申请退款", "请在订单详情中发起退款申请，或联系人工客服。", "售后"),
]


def init_database(engine: Engine) -> None:
    if engine.dialect.name == "mysql" and engine.url.database != "customer_service":
        raise ValueError("MySQL initialization is restricted to customer_service")
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import Session

    with Session(engine) as session, session.begin():
        existing = set(session.scalars(select(FAQ.question)).all())
        session.add_all(FAQ(question=q, answer=a, category=c) for q, a, c in FAQ_SEEDS if q not in existing)


def main() -> None:
    import os
    from dotenv import load_dotenv

    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is required")
    init_database(make_engine(url))
    print("Database initialized")


if __name__ == "__main__":
    main()
