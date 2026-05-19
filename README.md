# llm-eval-harness

Automated evaluation pipeline for RAG and agentic LLM systems. Define test cases in YAML, run them across multiple model providers, score outputs with LLM-as-judge metrics, persist results to Postgres, and ship a CI/CD gate that blocks hallucination regressions.

Most LLM pipelines ship without any evaluation layer. When hallucinations 
slip through, there's no systematic way to catch regressions before they 
reach production. This harness fixes that — define your test cases in YAML 
once, run them across any model provider, and block bad deployments 
automatically via a CI/CD gate.

```
evals/tasks/*.yaml  →  RAG pipeline  →  LLM-as-judge scoring  →  Postgres  →  dashboard + CI gate
```

---

## Features

- **YAML-defined eval tasks** — write test cases once, run anywhere, works across any domain
- **RAG pipeline** — ingest PDFs and text into ChromaDB, retrieve live context per question
- **LLM-as-judge scoring** — faithfulness, relevance, hallucination rate, latency
- **Multi-provider** — run the same cases against Ollama, Groq, OpenAI, and Claude simultaneously
- **Postgres persistence** — every run and result stored, queryable via REST API
- **FastAPI dashboard** — filter runs by task/model, trigger evals, view live job logs
- **Background job tracking** — live log tail with status polling, auto-refreshes on completion
- **HTML reports** — per-question score breakdowns, CI gate banner, model comparison
- **CI/CD gate** — exits with code `1` if hallucination rate exceeds threshold, blocking deployment

---

## Stack

| Layer | Technology |
|---|---|
| Eval scoring | LLM-as-judge (Groq) |
| Vector store | ChromaDB (local) |
| Embeddings | sentence-transformers (local) |
| Model providers | Ollama · Groq · OpenAI · Anthropic |
| Database | Postgres + SQLAlchemy |
| API | FastAPI |
| Reports | Jinja2 HTML |
| CI/CD | GitHub Actions |

---

## Quickstart

**1. Clone and install**

```bash
git clone https://github.com/kshitijqwerty/llm-eval-harness.git
cd llm-eval-harness
pip install -r requirements.txt
```

**2. Start Postgres**

```bash
docker run --name eval-db \
  -e POSTGRES_USER=eval \
  -e POSTGRES_PASSWORD=eval \
  -e POSTGRES_DB=evaldb \
  -p 5432:5432 -d postgres:16
```

**3. Configure environment**

```bash
cp .env.example .env
# Add GROQ_API_KEY — free at console.groq.com
# DATABASE_URL is pre-filled for the Docker setup above
```

**4. Start Ollama**

```bash
ollama serve          # separate terminal
ollama pull llama3.2
```

**5. Run an eval**

```bash
# Customer support RAG eval
python main.py evals/tasks/rag_support.yaml

# Medical FAQ RAG eval
python main.py evals/tasks/medical_faq.yaml
```

**6. Open the dashboard**

```bash
uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

---

## Eval task format

### Direct mode — provide context in the YAML

```yaml
name: my_eval
model_providers:
  - ollama
  - groq

cases:
  - id: q001
    question: "What is the refund policy?"
    context: "All sales are final except in cases of defect."
    expected: "Sales are final unless the product is defective."
```

### RAG mode — context retrieved live from documents

```yaml
name: my_rag_eval
mode: rag
documents:
  - docs/policy.txt
  - docs/faq.pdf

model_providers:
  - ollama
  - groq

cases:
  - id: r001
    question: "What is the refund policy?"
    expected: "Sales are final unless the product is defective."
```

No `context` field needed in RAG mode — the pipeline chunks, embeds, and retrieves it automatically.

---

## Metrics

| Metric | What it measures | Direction |
|---|---|---|
| Faithfulness | Answer grounded in context? | Higher is better |
| Relevance | Answer addresses the question? | Higher is better |
| Hallucination | Facts not in context asserted? | Lower is better |
| Latency | Response time in ms | Lower is better |

Scoring uses the **LLM-as-judge pattern** — a fast Groq model evaluates each answer against the context and question. No brittle string matching. Hallucination threshold for CI gate is `0.25`, configurable in `harness/metrics.py`.

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/runs` | List eval runs, filter with `?task=` |
| `GET` | `/runs/{id}` | Single run summary |
| `GET` | `/runs/{id}/results` | Per-case results, filter with `?model=` |
| `GET` | `/runs/{id}/diff` | Regression diff vs previous run for same task |
| `GET` | `/runs/{id}/report` | Serve HTML report |
| `POST` | `/trigger` | Start a background eval run |
| `GET` | `/jobs` | List background jobs with live log tails |
| `GET` | `/jobs/{id}/logs` | Full log output for a job |

---

## CI/CD integration

```yaml
# .github/workflows/eval_gate.yml
name: LLM Eval Gate

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  eval:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: eval
          POSTGRES_PASSWORD: eval
          POSTGRES_DB: evaldb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Pull Ollama model
        run: |
          curl -fsSL https://ollama.com/install.sh | sh
          ollama serve &
          sleep 5
          ollama pull llama3.2

      - name: Run eval harness
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          DATABASE_URL: postgresql://eval:eval@localhost:5432/evaldb
        run: python main.py evals/tasks/rag_support.yaml

      - name: Upload HTML report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: reports/*.html
```

Add `GROQ_API_KEY` under **Settings → Secrets → Actions**. The pipeline exits with code `1` if hallucination exceeds threshold, blocking the deployment.

---

## Project structure

```
llm-eval-harness/
├── evals/
│   └── tasks/
│       ├── sample_rag.yaml          # direct mode example
│       ├── rag_support.yaml         # customer support RAG eval
│       └── medical_faq.yaml         # medical FAQ RAG eval
├── docs/
│   ├── support_policy.txt           # source doc for rag_support eval
│   └── medical_faq.txt              # source doc for medical_faq eval
├── harness/
│   ├── models.py                    # provider adapters (Ollama, Groq, OpenAI)
│   ├── metrics.py                   # LLM-as-judge scoring engine
│   ├── rag.py                       # ChromaDB ingestion + retrieval
│   ├── runner.py                    # orchestrator
│   ├── reporter.py                  # HTML report generator
│   ├── regression.py                # regression diffing vs previous run
│   ├── db.py                        # SQLAlchemy models + session
│   └── report_template.html         # Jinja2 report template
├── api/
│   ├── main.py                      # FastAPI app + routes
│   ├── jobs.py                      # background job tracker
│   └── templates/
│       └── dashboard.html           # web dashboard
├── reports/                         # generated reports (gitignored)
├── screenshots/                     # for README
├── .github/workflows/
│   └── eval_gate.yml                # CI/CD gate
├── main.py                          # entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Extending the harness

**Add a model provider** — extend `BaseLLMAdapter` in `harness/models.py` and register it in `get_adapter()`:

```python
class MyProviderAdapter(BaseLLMAdapter):
    @property
    def model_id(self) -> str:
        return "myprovider/model-name"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # call your provider here
        ...
```

**Add a metric** — add a scoring function to `harness/metrics.py`, add the field to `EvalResult`, and include it in `evaluate_case()`:

```python
def score_completeness(expected: str, answer: str) -> float:
    prompt = f"Expected: {expected}\nAnswer: {answer}\n..."
    return _judge_score(prompt)
```

**Add a new domain** — create a YAML task file and a corresponding source document:

```bash
evals/tasks/legal_faq.yaml
docs/legal_faq.txt
```

No code changes required.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Judge model + optional test provider. Free at console.groq.com |
| `DATABASE_URL` | Yes | Postgres connection string |
| `OPENAI_API_KEY` | No | For GPT-4o as a tested provider |
| `ANTHROPIC_API_KEY` | No | For Claude as a tested provider |

Ollama runs locally and needs no API key.