"""Interface LLM remplaçable (L2.5 — stub minimal pour L2.3)."""

from packages.analyse.llm.client import FakeLLMClient, LLMClient

__all__ = ["FakeLLMClient", "LLMClient"]
