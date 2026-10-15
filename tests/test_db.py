"""engine / session factory wiring (plan T1 Step 4)."""

from app import db
from app.config import get_settings


def test_get_engine_uses_settings_url():
    # create_async_engine 懒连接,不会真连 MySQL;URL 必须来自 settings。
    # 注意 str(URL) 会把密码掩码成 ***,比对要用 render_as_string(hide_password=False)。
    settings = get_settings()
    assert db.get_engine().url.render_as_string(hide_password=False) == settings.database_url


def test_get_sessionmaker_singleton():
    assert db.get_sessionmaker() is db.get_sessionmaker()
