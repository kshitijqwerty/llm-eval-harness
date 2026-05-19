# llm-eval-harness

Automated evaluation pipeline for RAG and agentic LLM systems. Define test cases in YAML, run them across multiple model providers simultaneously, and get scored HTML reports with CI/CD integration.

```
evals/tasks/my_task.yaml  →  runner  →  scores  →  HTML report + CI gate
```

---

## Features

- **YAML-defined eval tasks** — write test cases once, run anywhere
- **RAG pipeline support** — ingest PDFs/text into ChromaDB, retrieve live context per question
- **LLM-as-judge scoring** — faithfulness, relevance, hallucination rate, latency
- **Multi-provider** — run the same cases against Ollama, Groq, OpenAI, Claude simultaneously
- **HTML reports** — per-question score breakdowns with regression diffs
- **CI/CD gate** — fails the pipeline if hallucination rate exceeds threshold

---

## Stack

| Layer | Technology |
|---|---|
| Eval scoring | LLM-as-judge (Groq) |
| Vector store | ChromaDB (local) |
| Embeddings | sentence-transformers (local) |
| Model providers | Ollama · Groq · OpenAI · Anthropic |
| Reports | Jinja2 HTML |
| CI/CD | GitHub Actions |

---

## Quickstart

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Add API keys**

```bash
cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com)
# Optionally add OPENAI_API_KEY, ANTHROPIC_API_KEY
```

**3. Start Ollama (for local models)**

```bash
ollama serve          # in a separate terminal
ollama pull llama3.2
```

**4. Run an eval**

```bash
# Direct mode (context provided in YAML)
python main.py evals/tasks/sample_rag.yaml

# RAG mode (context retrieved live from documents)
python main.py evals/tasks/rag_support.yaml
```

Open the generated report in `reports/`.

---

## Eval Task Format

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

No `context` field needed in RAG mode — the pipeline retrieves it.

---

## Metrics

| Metric | What it measures | Direction |
|---|---|---|
| **Faithfulness** | Answer grounded in context? | Higher is better |
| **Relevance** | Answer addresses the question? | Higher is better |
| **Hallucination** | Facts not present in context? | Lower is better |
| **Latency** | Time to first token (ms) | Lower is better |

Scoring uses the **LLM-as-judge pattern** — a fast Groq model evaluates each answer against the context and question. No brittle string matching.

---

## CI/CD Integration

Add to `.github/workflows/eval_gate.yml`:

```yaml
- name: Run eval harness
  env:
    GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
  run: python main.py evals/tasks/your_task.yaml
```

The pipeline exits with code `1` if average hallucination rate exceeds `0.25`, blocking the deployment. Configure the threshold in `harness/metrics.py`.

---

## Project Structure

```
llm-eval-harness/
├── evals/
│   └── tasks/
│       ├── sample_rag.yaml      # direct mode example
│       └── rag_support.yaml     # RAG mode example
├── docs/                        # source documents for RAG ingestion
├── harness/
│   ├── models.py                # provider adapters (Ollama, Groq, OpenAI)
│   ├── metrics.py               # LLM-as-judge scoring engine
│   ├── rag.py                   # ChromaDB ingestion + retrieval
│   ├── runner.py                # orchestrator
│   └── reporter.py              # HTML report generator
├── reports/                     # generated reports (gitignored)
├── .github/workflows/
│   └── eval_gate.yml            # CI/CD gate
├── main.py
└── requirements.txt
```

---

## Adding a New Model Provider

Create a class in `harness/models.py` that extends `BaseLLMAdapter`:

```python
class MyProviderAdapter(BaseLLMAdapter):
    @property
    def model_id(self) -> str:
        return "myprovider/model-name"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # call your provider here
        ...
```

Register it in `get_adapter()` and reference it by name in any YAML task.

---

## Adding a New Metric

Add a scoring function to `harness/metrics.py`:

```python
def score_completeness(expected: str, answer: str) -> float:
    prompt = f"Expected: {expected}\nAnswer: {answer}\n..."
    return _judge_score(prompt)
```

Then add it to the `evaluate_case` function and the `EvalResult` dataclass.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Used as the judge model. Free at console.groq.com |
| `OPENAI_API_KEY` | Optional | For GPT-4o as a tested provider |
| `ANTHROPIC_API_KEY` | Optional | For Claude as a tested provider |

Ollama runs locally and needs no key.