import time
from dataclasses import dataclass
from harness.models import BaseLLMAdapter, GroqAdapter

# Result Container
@dataclass
class EvalResult:
    case_id: str
    question: str
    context: str
    expected: str
    actual: str
    model_id: str

    # floats (0.0 - 1.0)
    faithfulness: float
    relevance: float
    hallucination: float #higher is worse
    latency_ms: float
    passed: bool # Confidence interval threshold




# LLM Judge
# Using Groq as LLM Judge
_judge : BaseLLMAdapter = GroqAdapter()


def _judge_score(prompt: str) -> float:
    """
    Ask the judge model to return a score between 0 and 1.
    We parse just the float out of its response.
    """
    response = _judge.complete(
        system_prompt=(
            "You are a strict evaluator. "
            "Respond with ONLY a number between 0.0 and 1.0. "
            "No explanation, no punctuation, just the number."
        ),
        user_prompt=prompt,
    )

    for token in response.strip().split():
        try:
            score = float(token)
            return max(0.0, min(1.0, score))

        except ValueError:
            continue
    
    return 0.0

# Individual Metrics

def score_faithfulness(context: str, answer: str) -> float:
    """
    Did the answer only use information from the context?
    1.0 = fully grounded, 0.0 = completely made up
    """
    prompt = f"""
Context:
{context}

Answer:
{answer}

Score how faithfully the answer is ground in the context.
1.0 means every claim in the answer is supported by the context.
0.0 mean the answer ingnores or contradicts the context entirely.
"""
    return _judge_score(prompt)


def score_relevance(question: str, answer: str) -> float:
    """
    Did the answer actually address the question?
    1.0 = directly answers it, 0.0 = completely off-topic
    """
    prompt = f"""
Question:
{question}

Answer:
{answer}

Score how relevant the answer is to the question.
1.0 means it directly and completely answers the question.
0.0 means it does not address the question at all.
"""
    return _judge_score(prompt)


def score_hallucination(context: str, answer: str) -> float:
    """
    Did the answer assert facts NOT present in the context?
    1.0 = severe hallucination, 0.0 = no hallucination
    Note: this is the one metric where LOWER is better.
    """
    prompt = f"""
Context:
{context}

Answer:
{answer}

Score how much the answer contains fabricated information
not found in the context.
1.0 means the answer is full of made-up facts.
0.0 means the answer contains zero fabricated information.
"""
    return _judge_score(prompt)


# Main Eval Function

HALLUCINATION_THRESHOLD = 0.25

def evaluate_case(
        case_id: str,
        question: str,
        context: str,
        expected:str,
        adapter: BaseLLMAdapter,
) -> EvalResult:
    """
    Run a single eval case against one model adapter.
    Returns a fully scored EvalResult.
    """

# Build the prompt the model-under-test receives
    user_prompt = f"""Answer the question using only the context below.

Context:
{context}

Question:
{question}"""

    # Time the model call
    start = time.time()
    actual = adapter.complete(
        system_prompt="You are a helpful assistant. Be concise.",
        user_prompt=user_prompt,
    )
    latency_ms = (time.time() - start) * 1000

    # Score it
    faithfulness = score_faithfulness(context, actual)
    relevance    = score_relevance(question, actual)
    hallucination = score_hallucination(context, actual)

    passed = hallucination <= HALLUCINATION_THRESHOLD

    return EvalResult(
        case_id=case_id,
        question=question,
        context=context,
        expected=expected,
        actual=actual,
        model_id=adapter.model_id,
        faithfulness=faithfulness,
        relevance=relevance,
        hallucination=hallucination,
        latency_ms=latency_ms,
        passed=passed,
    )
