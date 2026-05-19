from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel
from pathlib import Path
import subprocess, sys

from api.jobs import start_job, get_job, list_jobs

from harness.db import init_db, get_db, EvalRun, EvalResultRow

app = FastAPI(title="LLM Eval Harness", version="1.0")

templates = Jinja2Templates(directory="api/templates")

@app.on_event("startup")
def startup():
    init_db()


# Pydantic response schemas
class RunSummary(BaseModel):
    id: int
    task_name: str
    mode: str
    created_at: str
    ci_passed: bool
    avg_hallucination: float
    avg_faithfulness: float
    avg_relevance: float
    avg_latency_ms: float
    total_cases: int
    passed_cases: int

    class Config:
        from_attributes = True


class ResultDetail(BaseModel):
    id: int
    case_id: str
    model_id: str
    question: str
    actual: str
    faithfulness: float
    relevance: float
    hallucination: float
    latency_ms: float
    passed: bool

    class Config:
        from_attributes = True


class TriggerRequest(BaseModel):
    task_path: str   # e.g. "evals/tasks/sample_rag.yaml"


# Routes
@app.get("/runs", response_model=list[RunSummary])
def list_runs(
    task: str | None = None,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """List recent eval runs, optionally filtered by task name."""
    q = db.query(EvalRun).order_by(desc(EvalRun.created_at))
    if task:
        q = q.filter(EvalRun.task_name == task)
    runs = q.limit(limit).all()
    return [
        RunSummary(
            **{c: getattr(r, c) for c in RunSummary.model_fields if c != "created_at"},
            created_at=r.created_at.isoformat(),
        )
        for r in runs
    ]


@app.get("/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunSummary(
        **{c: getattr(run, c) for c in RunSummary.model_fields if c != "created_at"},
        created_at=run.created_at.isoformat(),
    )


@app.get("/runs/{run_id}/results", response_model=list[ResultDetail])
def get_run_results(
    run_id: int,
    model: str | None = None,
    db: Session = Depends(get_db),
):
    """Get per-case results for a run, optionally filtered by model."""
    q = db.query(EvalResultRow).filter(EvalResultRow.run_id == run_id)
    if model:
        q = q.filter(EvalResultRow.model_id == model)
    return q.all()


@app.post("/trigger")
def trigger_eval(req: TriggerRequest):
    job_id = start_job(req.task_path)
    return {"job_id": job_id, "status": "running"}

@app.get("/jobs")
def get_jobs():
    return [
        {
            "id": j.id,
            "task_path": j.task_path,
            "status": j.status,
            "started_at": j.started_at,
            "finished_at": j.finished_at,
            "log_tail": j.logs[-10:],   # last 10 lines
        }
        for j in list_jobs()
    ]

@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"id": job_id, "status": job.status, "logs": job.logs}


@app.get("/runs/{run_id}/report")
def get_report(run_id: int, db: Session = Depends(get_db)):
    """Serve the HTML report for a run."""
    run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    reports = sorted(Path("reports").glob(f"{run.task_name}_*.html"))
    if not reports:
        raise HTTPException(status_code=404, detail="No report found")
    return FileResponse(reports[-1], media_type="text/html")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
    )