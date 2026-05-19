import yaml
from pathlib import Path
from rich.console import Console
from rich.progress import track
from harness.metrics import evaluate_case, EvalResult
from harness.models import get_adapter

console = Console()

def load_task(yaml_path: str) -> dict:
    """Load and validate an eval task YAML file"""
    path = Path(yaml_path)
    if not path.exists():
        raise FileExistsError(f"Task file not found: {yaml_path}")
    with open(path) as f:
        task = yaml.safe_load(f)
    
    # Basic Validation
    required_keys = {"name", "model_providers", "cases"}
    missing = required_keys - task.keys()
    if missing:
        raise ValueError(f"Task YAML missing keys: {missing}")
    
    return task

def run_task(yaml_path: str) -> list[EvalResult]:
    """
    Main Entry point, run all cases x all providers
    Returns flat list of EvalResults
    """

    task = load_task(yaml_path)
    results: list[EvalResult] = []

    console.rule(f"[bold cyan]Eval Task: {task['name']}")
    console.print(f"    Providers   :   {task['model_providers']}")
    console.print(f"    Cases       :   {len(task['cases'])}")
    console.print(f"    Total runs  :   {len(task['model_providers']) * len(task['cases'])}\n")

    for provider in task["model_providers"]:
        console.print(f"[bold yellow] Running provider: {provider}")

        try:
            adapter = get_adapter(provider)
        except ValueError as e:
            console.print(f"  [red]Skipping — {e}")
            continue

        for case in track(task["cases"], description=f"  {provider}"):
            try:
                result = evaluate_case(
                    case_id=case["id"],
                    question=case["question"],
                    context=case["context"],
                    expected=case["expected"],
                    adapter=adapter,
                )
                results.append(result)

                # Live feedback per case
                status = "[green]PASS" if result.passed else "[red]FAIL"
                console.print(
                    f"  {status}[/]  {case['id']}"
                    f"  faith={result.faithfulness:.2f}"
                    f"  rel={result.relevance:.2f}"
                    f"  hall={result.hallucination:.2f}"
                    f"  {result.latency_ms:.0f}ms"
                )

            except Exception as e:
                console.print(f"  [red]ERROR on {case['id']}: {e}")
    

    # Summary
    total   = len(results)
    passed  = sum(1 for r in results if r.passed)
    avg_hall = sum(r.hallucination for r in results) / total if total else 0

    console.rule("[bold cyan]Summary")
    console.print(f"  Passed           : {passed}/{total}")
    console.print(f"  Avg hallucination: {avg_hall:.2f}")

    if avg_hall > 0.25:
        console.print("\n  [bold red]✗ CI GATE FAILED — hallucination rate exceeds threshold")
    else:
        console.print("\n  [bold green]✓ CI GATE PASSED")

    # ── Generate HTML report ──────────────────────────────────────
    from harness.reporter import generate_report
    report_path = generate_report(task["name"], results)
    console.print(f"\n  [bold]Report saved:[/] {report_path}")

    return results