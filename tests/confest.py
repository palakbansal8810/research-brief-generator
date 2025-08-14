import asyncio
import importlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

MODULE_NAMES = ["app"]
project = None
for name in MODULE_NAMES:
    try:
        project = importlib.import_module(name)
        break
    except ImportError:
        continue

if project is None:
    raise ImportError(
        "Could not import project module. Please ensure your main application is named 'research_brief.py', 'main.py', or 'app.py' and is on PYTHONPATH."
    )

@pytest.fixture(scope="session")
def module():
    return project


@pytest.fixture
def event_loop(request):
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    # Ensure tests don't write to the real data/ directory
    monkeypatch.setenv("HISTORY_DIR", str(tmp_path / "data"))
    # Also override module-level HISTORY_DIR if present
    if hasattr(project, "HISTORY_DIR"):
        monkeypatch.setattr(project, "HISTORY_DIR", str(tmp_path / "data"))
    yield


@pytest.fixture
def disable_llms(monkeypatch):
    """Disable real LLMs so nodes use fallback logic in your code."""
    if hasattr(project, "llm_smart"):
        monkeypatch.setattr(project, "llm_smart", None)
    if hasattr(project, "llm_fast"):
        monkeypatch.setattr(project, "llm_fast", None)
    return True


@pytest.fixture
def fake_search_tool(monkeypatch):
    """Replace search tool with deterministic stub."""
    def _stub(query: str, num_results: int = 5):
        return [
            {"title": f"Result for {query}", "href": f"https://example.com/{i}", "link": f"https://example.com/{i}", "body": "Snippet text"}
            for i in range(1, num_results + 1)
        ]

    # If your module exposes `search_tool` or `duckduckgo_search`, patch both
    if hasattr(project, "search_tool"):
        monkeypatch.setattr(project, "search_tool", _stub)
    if hasattr(project, "duckduckgo_search"):
        monkeypatch.setattr(project, "duckduckgo_search", _stub)
    return _stub


@pytest.fixture
def fake_aiohttp(monkeypatch):
    """Monkeypatch aiohttp.ClientSession to return predictable HTML content."""
    class DummyResponse:
        def __init__(self, url):
            self.status = 200
            self._url = url

        async def text(self):
            return f"<html><body><p>Content for {self._url}</p></body></html>"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class DummySession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, timeout=None):
            return DummyResponse(url)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(project, "aiohttp", project.aiohttp if hasattr(project, "aiohttp") else __import__("aiohttp"))
    monkeypatch.setattr(project.aiohttp, "ClientSession", DummySession)
    return DummySession