"""Migration: add User, UserIntegration, CalendarEvent tables; add user_id to Meeting."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.models.db import create_db_and_tables, get_engine


def migrate():
    engine = get_engine()
    # SQLite requires raw ALTER TABLE for new columns on existing tables
    with engine.connect() as conn:
        for stmt in [
            "ALTER TABLE meeting ADD COLUMN user_id TEXT REFERENCES user(id)",
        ]:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except OperationalError:
                pass  # column already exists
    # Create all new tables (checkfirst=True via create_all skips existing ones)
    create_db_and_tables(engine)
    print("Migration 001 complete")


if __name__ == "__main__":
    migrate()
