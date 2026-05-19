# main.py
import sys
from harness.runner import run_task

if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "evals/tasks/sample_rag.yaml"
    results = run_task(yaml_path)

    # Exit code 1 = CI pipeline fails the build
    failed = [r for r in results if not r.passed]
    sys.exit(1 if failed else 0)