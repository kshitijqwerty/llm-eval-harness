# test_models.py (temporary, delete after)
from harness.models import OllamaAdapter, GroqAdapter

question = "What is 2 + 2? Answer in one word."

print("Testing Ollama...")
ollama = OllamaAdapter()
print(" →", ollama.complete("You are helpful.", question))

print("Testing Groq...")
groq = GroqAdapter()
print(" →", groq.complete("You are helpful.", question))