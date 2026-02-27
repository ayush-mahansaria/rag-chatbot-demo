"""
conftest.py — pytest configuration and shared fixtures.

Adds the repo root to sys.path so `from src.rag_chatbot import ...`
and `from api import app` work in both local and CI environments
without needing an editable install.
"""
import sys
import os
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure test runs without a real OpenAI key
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-ci")
