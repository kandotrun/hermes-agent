from __future__ import annotations

import sqlite3

import hermes_state
from hermes_state import SessionDB


def test_fast_health_probe_skips_full_integrity_scan(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()

    real_connect = sqlite3.connect
    statements: list[str] = []

    class ConnectionProxy:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args, **kwargs):
            statements.append(str(sql))
            if str(sql).strip().lower().startswith("pragma integrity_check"):
                raise AssertionError("fast probe must not scan the whole database")
            return self._conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def connect(*args, **kwargs):
        return ConnectionProxy(real_connect(*args, **kwargs))

    monkeypatch.setattr(hermes_state.sqlite3, "connect", connect)

    assert hermes_state._db_opens_cleanly(db_path, full_integrity=False) is None
    assert not any(
        sql.strip().lower().startswith("pragma integrity_check")
        for sql in statements
    )
