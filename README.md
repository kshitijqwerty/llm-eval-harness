# llm-eval-harness

Automated evaluation pipeline for RAG and agentic LLM systems. Define test cases in YAML, run them across multiple model providers, score outputs with LLM-as-judge metrics, and ship a CI/CD gate that blocks hallucination regressions.

```
evals/tasks/*.yaml  →  RAG pipeline  →  LLM-as-judge scoring  →  Postgres  →  dashboard + CI gate
```

---

## Features

- **YAML-defined eval tasks** — write test cases once, run anywhere
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

**1. Install dependencies**

```bash
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
# Required: GROQ_API_KEY (free at console.groq.com)
# Optional: OPENAI_API_KEY, ANTHROPIC_API_KEY
# Set:      DATABASE_URL=postgresql://eval:eval@localhost:5432/evaldb
```

**4. Start Ollama**

```bash
ollama serve          # separate terminal
ollama pull llama3.2
```

**5. Run an eval**

```bash
# Direct mode — context provided in YAML
python main.py evals/tasks/sample_rag.yaml

# RAG mode — context retrieved live from documents
python main.py evals/tasks/rag_support.yaml
```

**6. Open the dashboard**

```bash
uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

---

## Eval task format

### Direct mode

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

### RAG mode

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

No `context` field needed in RAG mode — the pipeline retrieves it from your documents.

---

## Metrics

| Metric | What it measures | Direction |
|---|---|---|
| Faithfulness | Answer grounded in context? | Higher is better |
| Relevance | Answer addresses the question? | Higher is better |
| Hallucination | Facts not in context asserted? | Lower is better |
| Latency | Response time in ms | Lower is better |

Scoring uses the **LLM-as-judge pattern** — a fast Groq model evaluates each answer. No brittle string matching. Hallucination threshold for CI gate: `0.25` (configurable in `harness/metrics.py`).

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/runs` | List eval runs, filter with `?task=` |
| `GET` | `/runs/{id}` | Single run summary |
| `GET` | `/runs/{id}/results` | Per-case results, filter with `?model=` |
| `GET` | `/runs/{id}/report` | Serve HTML report |
| `POST` | `/trigger` | Start a background eval run |
| `GET` | `/jobs` | List background jobs with log tails |
| `GET` | `/jobs/{id}/logs` | Full log output for a job |

---

## CI/CD integration

```yaml
# .github/workflows/eval_gate.yml
- name: Run eval harness
  env:
    GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
    DATABASE_URL: ${{ secrets.DATABASE_URL }}
  run: python main.py evals/tasks/your_task.yaml
```

Exits with code `1` if hallucination rate exceeds threshold — blocks the deployment. Add `GROQ_API_KEY` and `DATABASE_URL` under **Settings → Secrets → Actions**.

---

## Project structure

```
llm-eval-harness/
├── evals/
│   └── tasks/
│       ├── sample_rag.yaml          # direct mode example
│       └── rag_support.yaml         # RAG mode example
├── docs/                            # source documents for RAG ingestion
├── harness/
│   ├── models.py                    # provider adapters (Ollama, Groq, OpenAI)
│   ├── metrics.py                   # LLM-as-judge scoring engine
│   ├── rag.py                       # ChromaDB ingestion + retrieval
│   ├── runner.py                    # orchestrator
│   ├── reporter.py                  # HTML report generator
│   ├── db.py                        # SQLAlchemy models + session
│   └── report_template.html         # Jinja2 report template
├── api/
│   ├── main.py                      # FastAPI app + routes
│   ├── jobs.py                      # background job tracker
│   └── templates/
│       └── dashboard.html           # web dashboard
├── reports/                         # generated reports (gitignored)
├── .github/workflows/
│   └── eval_gate.yml                # CI/CD gate
├── main.py                          # entry point
├── requirements.txt
└── .env.example
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
        ...
```

**Add a metric** — add a scoring function to `harness/metrics.py`, add the field to `EvalResult`, and include it in `evaluate_case()`:

```python
def score_completeness(expected: str, answer: str) -> float:
    prompt = f"Expected: {expected}\nAnswer: {answer}\n..."
    return _judge_score(prompt)
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Judge model + optional test provider. Free at console.groq.com |
| `DATABASE_URL` | Yes | Postgres connection string |
| `OPENAI_API_KEY` | No | For GPT-4o as a tested provider |
| `ANTHROPIC_API_KEY` | No | For Claude as a tested provider |

Ollama runs locally and needs no key.