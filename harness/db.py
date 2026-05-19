# harness/db.py
import os
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Boolean,
    DateTime,
    Integer,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://eval:eval@localhost:5432/evaldb")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class EvalRun(Base):
    """One full execution of a YAML task across all providers."""

    __tablename__ = "eval_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_name = Column(String, nullable=False, index=True)
    mode = Column(String, default="direct")  # direct | rag
    created_at = Column(DateTime, default=datetime.utcnow)
    ci_passed = Column(Boolean, nullable=False)
    avg_hallucination = Column(Float)
    avg_faithfulness = Column(Float)
    avg_relevance = Column(Float)
    avg_latency_ms = Column(Float)
    total_cases = Column(Integer)
    passed_cases = Column(Integer)

    results = relationship("EvalResultRow", back_populates="run", cascade="all, delete")


class EvalResultRow(Base):
    """One scored case within a run."""

    __tablename__ = "eval_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("eval_runs.id"), nullable=False, index=True)
    case_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False, index=True)
    question = Column(Text)
    context = Column(Text)
    expected = Column(Text)
    actual = Column(Text)
    faithfulness = Column(Float)
    relevance = Column(Float)
    hallucination = Column(Float)
    latency_ms = Column(Float)
    passed = Column(Boolean)

    run = relationship("EvalRun", back_populates="results")


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a session, closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
