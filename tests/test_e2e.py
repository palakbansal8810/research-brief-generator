import pytest
import asyncio
from fastapi.testclient import TestClient
from app import app, ResearchState, FinalBrief, ResearchPlan, SourceSummary, SynthesisDraft
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_generate_brief_e2e():
    client = TestClient(app)
    
    with patch("app.llm_smart") as mock_llm_smart, patch("app.llm_fast") as mock_llm_fast, \
         patch("app.search_tool") as mock_search:
        # Mock search tool
        mock_search.return_value = [
            {"title": "Test Source", "link": "https://example.com", "snippet": "Test snippet"}
        ]
        
        # Mock content fetch
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="<p>Test content</p>")
            mock_get.return_value.__aenter__.return_value = mock_response

            # Mock LLM responses
            mock_llm_fast.with_structured_output.return_value.ainvoke = AsyncMock(return_value=ResearchPlan(
                topic="AI in Education",
                steps=["Research AI tools", "Analyze impact"],
                estimated_time=15,
                priority_areas=["Tools", "Impact"]
            ))
            mock_llm_smart.with_structured_output.return_value.ainvoke.side_effect = [
                # SourceSummary
                SourceSummary(
                    source_url="https://example.com",
                    title="Test Source",
                    summary="Test summary",
                    key_insights=["Insight 1"],
                    credibility_score=0.7
                ),
                # SynthesisDraft
                SynthesisDraft(
                    title="Research Brief: AI in Education",
                    summary="Summary of AI in education.",
                    sections=[{"heading": "Overview", "content": "Content", "key_points": ["Point 1"]}],
                    key_findings=["Finding 1"]
                )
            ]

            response = client.post(
                "/brief",
                json={
                    "topic": "AI in Education",
                    "depth": 2,
                    "follow_up": False,
                    "user_id": "test_user"
                }
            )
    
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
    brief = response.json()
    print(f"Actual brief title: '{brief['title']}'")  # Debug output
    print(brief["title"].strip())
    assert len(brief["sections"]) > 0
    assert len(brief["references"]) > 0