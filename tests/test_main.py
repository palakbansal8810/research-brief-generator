# tests/test_main.py
import pytest
import asyncio
from app import ResearchState, create_research_graph, FinalBrief, context_summarization_node
from datetime import datetime
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage
import os

@pytest.mark.asyncio
async def test_context_summarization_node_with_follow_up():
    state = ResearchState(topic="Test", user_id="test_user", follow_up=True)
    result = await context_summarization_node(state)
    assert result.context_summary.user_id == "test_user"
    assert result.context_summary.context_relevance in [
        "No prior research found",
        "Error in context processing",
        "Context available but could not be processed"
    ]

@pytest.mark.asyncio
async def test_context_summarization_node_with_follow_up():
    state = ResearchState(topic="Test", user_id="test_user", follow_up=True)
    result = await context_summarization_node(state)
    assert result.context_summary.user_id == "test_user"
    assert result.context_summary.context_relevance in ["No prior research found", "Error in context processing"]

def test_final_brief_schema():
    brief = FinalBrief(
        title="Test",
        summary="Test summary",
        sections=[],
        references=[],
    )
    assert brief.id is not None
    assert isinstance(brief.created_at, datetime)
    assert brief.metadata == {}

@pytest.mark.asyncio
async def test_resume_workflow():
    state = ResearchState(topic="Test", user_id="test_user", depth=1)
    config = {"configurable": {"thread_id": "test_resume"}}
    graph = create_research_graph()
    # Run partial workflow
    async for _ in graph.astream(state.model_dump(), config=config):
        break  # Simulate interruption after first step
    # Resume
    resumed_state = await graph.ainvoke({}, config=config)  # Add await
    assert resumed_state["execution_id"] == state.execution_id
    # Remove graph_ckpt.db check since using MemorySaver
   

@pytest.mark.asyncio
async def test_generate_brief():
    with patch("main.ChatGroq.ainvoke", new=AsyncMock(return_value=AIMessage(content='{"title":"Mock","summary":"Mock summary","sections":[],"key_findings":[]}'))):
        with patch("main.duckduckgo_search", return_value=[{"title": "Mock Source", "link": "http://example.com", "snippet": "Mock content"}]):
            graph = create_research_graph()
            state = ResearchState(topic="Test", user_id="test_user", depth=1)
            result = await graph.ainvoke(state.model_dump(), config={"configurable": {"thread_id": "test"}})
            assert result["final_brief"]["title"] == "Mock"
            assert result["final_brief"]["summary"] == "Mock summary"