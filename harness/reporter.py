# harness/reporter.py
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from jinja2 import Environment, FileSystemLoader
from harness.metrics import EvalResult

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_FILE = "report_template.html"

def generate_report(
    task_name: str,
    results: list[EvalResult],
    output_dir: str = "reports",
) -> str:
    """Render results to an HTML report. Returns the output file path."""

    # ── Aggregate stats ──────────────────────────────────────────
    total   = len(results)
    passed  = sum(1 for r in results if r.passed)

    avg = lambda key: sum(getattr(r, key) for r in results) / total if total else 0
    avg_faithfulness  = avg("faithfulness")
    avg_relevance     = avg("relevance")
    avg_hallucination = avg("hallucination")
    avg_latency       = avg("latency_ms")
    ci_passed         = avg_hallucination <= 0.25

    # ── Group by provider for per-model tables ───────────────────
    results_by_provider: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        results_by_provider[r.model_id].append(r)

    # ── Render ───────────────────────────────────────────────────
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template(TEMPLATE_FILE)

    html = template.render(
        task_name=task_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_runs=total,
        passed_runs=passed,
        providers=list(results_by_provider.keys()),
        avg_faithfulness=avg_faithfulness,
        avg_relevance=avg_relevance,
        avg_hallucination=avg_hallucination,
        avg_latency=avg_latency,
        ci_passed=ci_passed,
        results_by_provider=results_by_provider,
    )

    # ── Write file ───────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{task_name}_{timestamp}.html"
    out_path.write_text(html)

    return str(out_path)