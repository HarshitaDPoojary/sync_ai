import os
import pytest
from sqlmodel import create_engine

os.environ.setdefault("GROQ_API_KEY", "test_key")
os.environ.setdefault("DATABASE_URL", ":memory:")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma")

from app.models.db import create_db_and_tables


@pytest.fixture
def engine():
    e = create_engine("sqlite:///:memory:")
    create_db_and_tables(e)
    return e
