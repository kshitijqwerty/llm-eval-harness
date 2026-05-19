# test_metrics.py (temporary)
from harness.metrics import evaluate_case
from harness.models import OllamaAdapter

result = evaluate_case(
    case_id="q001",
    question="What is the refund policy for digital products?",
    context="Digital products are non-refundable once downloaded. Exceptions apply if the product is defective.",
    expected="Digital products cannot be refunded after download, unless defective.",
    adapter=OllamaAdapter(),
)

print(f"Answer:        {result.actual}")
print(f"Faithfulness:  {result.faithfulness:.2f}")
print(f"Relevance:     {result.relevance:.2f}")
print(f"Hallucination: {result.hallucination:.2f}")
print(f"Latency:       {result.latency_ms:.0f}ms")
print(f"Passed:        {result.passed}")