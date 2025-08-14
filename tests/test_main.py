# tests/test_app.py
import pytest
import asyncio
from app import (
    ResearchState, BriefRequest, context_summarization_node, planning_node,
    search_node, content_fetch_node, summarization_node, synthesis_node,
    post_processing_node, FinalBrief, ContextSummary, ResearchPlan
)
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_context_summarization_node_no_follow_up():
    state = ResearchState(topic="AI in Education", user_id="test_user", follow_up=False)
    result_state = await context_summarization_node(state)
    assert result_state.context_summary is not None
    assert result_state.context_summary.context_relevance == "No prior context - initial query"
    assert result_state.context_summary.user_id == "test_user"
    assert result_state.context_summary.past_topics == []

@pytest.mark.asyncio
async def test_planning_node():
    state = ResearchState(topic="AI in Education", user_id="test_user", depth=2)
    with patch("app.llm_fast") as mock_llm:
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=ResearchPlan(
            topic="AI in Education",
            steps=["Step 1", "Step 2", "Step 3"],
            estimated_time=15,
            priority_areas=["Area 1", "Area 2"]
        ))
        result_state = await planning_node(state)
    assert result_state.plan is not None
    assert result_state.plan.topic == "AI in Education"
    assert len(result_state.plan.steps) >= 1
    assert result_state.final_brief is None  # Should not be updated

@pytest.mark.asyncio
async def test_search_node():
    state = ResearchState(topic="AI in Education", user_id="test_user", depth=2, plan=ResearchPlan(
        topic="AI in Education",
        steps=["Step 1", "Step 2"]
    ))
    with patch("app.search_tool") as mock_search:
        mock_search.return_value = [
            {"title": "Test", "link": "https://example.com", "snippet": "Test snippet"}
        ]
        result_state = await search_node(state)
    assert result_state.search_results
    assert len(result_state.search_results) <= 10
    assert result_state.search_results[0]["link"] == "https://example.com"

@pytest.mark.asyncio
async def test_content_fetch_node():
    state = ResearchState(
        topic="AI in Education",
        user_id="test_user",
        search_results=[{"title": "Test", "link": "https://example.com", "snippet": "Test snippet"}]
    )
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<p>Test content</p>")
        mock_get.return_value.__aenter__.return_value = mock_response
        result_state = await content_fetch_node(state)
    assert result_state.fetched_content
    assert result_state.fetched_content[0]["content"] == "Test content"