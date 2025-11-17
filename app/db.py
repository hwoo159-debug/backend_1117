from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://policy:policypw@localhost:5432/policydb")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def init_pgvector():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS documents (
          id SERIAL PRIMARY KEY,
          title TEXT,
          content TEXT NOT NULL,
          metadata JSONB DEFAULT '{}'::jsonb,
          embedding vector(384)
        )
        """))
        conn.commit()
