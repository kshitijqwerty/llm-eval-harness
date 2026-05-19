# api/jobs.py
import subprocess, sys, threading, time
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class Job:
    id: str
    task_path: str
    status: str = "running"   # running | done | failed
    started_at: str = ""
    finished_at: str = ""
    logs: list[str] = field(default_factory=list)

_jobs: dict[str, Job] = {}

def start_job(task_path: str) -> str:
    job_id = str(int(time.time() * 1000))
    job = Job(
        id=job_id,
        task_path=task_path,
        started_at=datetime.utcnow().isoformat(),
    )
    _jobs[job_id] = job

    def run():
        try:
            proc = subprocess.Popen(
                [sys.executable, "main.py", task_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                line = line.rstrip()
                job.logs.append(line)
                if len(job.logs) > 200:      # cap memory
                    job.logs.pop(0)
            proc.wait()
            job.status = "done" if proc.returncode == 0 else "failed"
        except Exception as e:
            job.logs.append(f"ERROR: {e}")
            job.status = "failed"
        finally:
            job.finished_at = datetime.utcnow().isoformat()

    threading.Thread(target=run, daemon=True).start()
    return job_id

def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)

def list_jobs() -> list[Job]:
    return sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)