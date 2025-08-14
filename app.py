import os
import uuid
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any,Union

import uvicorn
import typer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import logging
import aiohttp
from bs4 import BeautifulSoup
from ddgs import DDGS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Annotated
import operator

#Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Pydantic Schemas 
class Reference(BaseModel):
    title: str
    url: str
    relevance_score: Optional[float] = None

class Section(BaseModel):
    heading: str
    content: str
    key_points: List[str] = Field(default_factory=list)

class ResearchPlan(BaseModel):
    topic: Annotated[Union[str, list], operator.add]
    steps: List[str] = Field(min_length=1, max_length=10)
    estimated_time: Optional[int] = None
    priority_areas: List[str] = Field(default_factory=list)

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one research step is required')
        return v

class SourceSummary(BaseModel):
    source_url: str
    title: str
    summary: str
    key_insights: List[str] = Field(default_factory=list)
    credibility_score: Optional[float] = Field(ge=0.0, le=1.0, default=0.7)
    timestamp: datetime = Field(default_factory=datetime.now)

class ContextSummary(BaseModel):
    user_id: str
    past_topics: List[str]
    key_points: List[str]
    common_themes: List[str] = Field(default_factory=list)
    context_relevance: Optional[str] = None

class FinalBrief(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    summary: str
    sections: List[Section]
    references: List[Reference]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    token_usage: Optional[Dict[str, int]] = None
    processing_time: Optional[float] = None

class SynthesisDraft(BaseModel):
    title: str
    summary: str
    sections: List[Section]
    key_findings: List[str] = Field(default_factory=list)

class BriefRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=200),
    depth: int = Field(default=2, ge=1, le=5),
    follow_up: bool = Field(default=False),
    user_id: str = Field(..., min_length=1),
def overwrite_reducer(current: Any, new: Any) -> Any:
    """Return the new value, ignoring the current value."""
    return new

class ResearchState(BaseModel):
    topic: Annotated[Union[str, list], operator.add] = ""
    depth: Annotated[Union[int, list], operator.add] = 2
    follow_up: Annotated[bool, overwrite_reducer] = False
    user_id: Annotated[Union[str, list], operator.add] = ""
    context_summary: Annotated[Optional[ContextSummary], overwrite_reducer] = None  # Use overwrite_reducer
    plan: Annotated[Optional[ResearchPlan], overwrite_reducer] = None  # Use overwrite_reducer
    search_results: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)  # Keep operator.add for lists
    fetched_content: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)  # Keep operator.add
    summaries: Annotated[List[SourceSummary], operator.add] = Field(default_factory=list)  # Keep operator.add
    final_brief: Annotated[Optional[FinalBrief], overwrite_reducer] = None  # Use overwrite_reducer
    execution_id: Annotated[str, operator.add] = Field(default_factory=lambda: str(uuid.uuid4()))
    error_messages: Annotated[List[str], operator.add] = Field(default_factory=list)  # Keep operator.add
    start_time: Annotated[datetime, overwrite_reducer] = Field(default_factory=datetime.now)  # Simplified to datetime
    retries: Annotated[int, operator.add] = 0
    low_confidence: Annotated[bool, operator.add] = False
# ====== LLM Setup ======
try:
    llm_smart = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=2048)
    llm_fast = ChatGroq(model="gemma2-9b-it", temperature=0.0, max_tokens=1024)
except Exception as e:
    logger.error(f"Failed to initialize LLMs: {e}")
    llm_smart = llm_fast = None

# ====== Tools Setup ======
def duckduckgo_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
        return [{"title": r["title"], "link": r["href"], "snippet": r["body"]} for r in results]

search_tool = duckduckgo_search

def f(topic: Union[str, Annotated[list, operator.add]]):
    print(topic)
# ====== Storage ======
USER_HISTORY: Dict[str, List[FinalBrief]] = {}
EXECUTION_TRACES: Dict[str, Dict] = {}
HISTORY_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(HISTORY_DIR, exist_ok=True)

# ====== Utility Functions ======
def log_execution(execution_id: str, step: str, data: Dict[str, Any]):
    if execution_id not in EXECUTION_TRACES:
        EXECUTION_TRACES[execution_id] = {"steps": [], "start_time": datetime.now()}
    EXECUTION_TRACES[execution_id]["steps"].append({
        "step": step,
        "timestamp": datetime.now(),
        "data": data,
    })


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

def persist_brief(user_id: str, brief: FinalBrief):
    USER_HISTORY.setdefault(user_id, []).append(brief)
    USER_HISTORY[user_id] = USER_HISTORY[user_id][-10:]
    path = os.path.join(HISTORY_DIR, f"{user_id}.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(brief.model_dump_json() + "\n")
    except Exception as e:
        logger.warning(f"Failed to persist brief: {e}")

# ====== LangGraph Nodes ======
async def context_summarization_node(state: ResearchState) -> ResearchState:
    try:
        # Extract topic if it's a list (due to Annotated type)
        topic = state.topic[0] if isinstance(state.topic, list) else state.topic
        log_execution(state.execution_id, "context_summarization", {"user_id": state.user_id})
        
        if not state.follow_up:
            state.context_summary = ContextSummary(
                user_id=state.user_id,
                past_topics=[],
                key_points=[],
                context_relevance="No prior context - initial query",
            )
            return state

        past_briefs = USER_HISTORY.get(state.user_id, [])
        if not past_briefs:
            path = os.path.join(HISTORY_DIR, f"{state.user_id}.jsonl")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            USER_HISTORY.setdefault(state.user_id, []).append(FinalBrief(**data))
                    past_briefs = USER_HISTORY.get(state.user_id, [])
                except Exception as e:
                    logger.warning(f"Failed reading history: {e}")

        if not past_briefs:
            state.context_summary = ContextSummary(
                user_id=state.user_id,
                past_topics=[],
                key_points=[],
                context_relevance="No prior research found",
            )
            return state

        past_topics = [b.title for b in past_briefs[-5:]]
        past_summaries = [b.summary for b in past_briefs[-5:]]

        if llm_smart:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You analyze user research history and concisely extract themes and insights as JSON."),
                (
                    "human",
                    """
Analyze the user's history and relate it to the current topic.
Return JSON with keys: common_themes (list), key_points (list), context_relevance (string).

Past Topics: {topics}
Past Summaries: {summaries}
Current Topic: {current_topic}
""",
                ),
            ])
            msgs = prompt.format_messages(
                topics=past_topics,
                summaries=past_summaries[:1000],
                current_topic=topic,
            )
            resp = await llm_smart.ainvoke(msgs)
            try:
                data = json.loads(resp.content)
            except Exception:
                data = {
                    "common_themes": [],
                    "key_points": ["Previous research available but parsing failed"],
                    "context_relevance": "Context available but could not be processed",
                }
        else:
            data = {
                "common_themes": [],
                "key_points": [],
                "context_relevance": "",
            }

        state.context_summary = ContextSummary(
            user_id=state.user_id,
            past_topics=past_topics,
            key_points=data.get("key_points", []),
            common_themes=data.get("common_themes", []),
            context_relevance=data.get("context_relevance", ""),
        )
    except Exception as e:
        logger.error(f"Context summarization failed: {e}")
        state.error_messages.append(f"Context summarization error: {e}")
        state.context_summary = ContextSummary(
            user_id=state.user_id,
            past_topics=[],
            key_points=[],
            context_relevance="Error in context processing",
        )
    return state

async def planning_node(state: ResearchState) -> ResearchState:
    try:
        # Extract topic if it's a list (due to Annotated type)
        topic = state.topic[0] if isinstance(state.topic, list) else state.topic
        
        log_execution(state.execution_id, "planning", {"topic": topic, "depth": state.depth})
        
        context_info = ""
        if state.context_summary and state.context_summary.past_topics:
            context_info = (
                f"Past topics: {', '.join(state.context_summary.past_topics)}\n"
                f"Common themes: {', '.join(state.context_summary.common_themes)}\n"
                f"Relevance: {state.context_summary.context_relevance}"
            )

        if llm_fast:
            # Use a more explicit prompt format to avoid structured output issues
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research planning assistant. Create a structured research plan.

Response format should be valid JSON with exactly these keys:
- topic: string (the research topic)
- steps: array of strings (3-7 research steps)
- estimated_time: integer (estimated minutes)
- priority_areas: array of strings (key focus areas)

Example:
{{
  "topic": "Climate Change Impact",
  "steps": [
    "Research current climate data and trends",
    "Analyze economic impacts of climate change",
    "Review policy responses and effectiveness"
  ],
  "estimated_time": 15,
  "priority_areas": ["Environmental Data", "Economic Analysis", "Policy Review"]
}}"""),
                (
                    "human",
                    """Create a research plan for: "{topic}"
Depth level: {depth} (1=basic overview, 5=comprehensive analysis)
{context_info}

Return only valid JSON following the exact format specified.""",
                ),
            ])
            print(prompt)
            msgs = prompt.format_messages(topic=topic, depth=state.depth, context_info=context_info)
            print('sdbshf')
            try:
                # First try with structured output
                plan_parser = llm_fast.with_structured_output(ResearchPlan)
                print('dnjf')
                plan: ResearchPlan = await plan_parser.ainvoke(msgs)
                plan.topic = topic  # Ensure topic is set correctly
                print('qwqp')
                state.plan = plan
                print('aslpas')
            except Exception as struct_error:
                logger.warning(f"Structured output failed, trying manual parsing: {struct_error}")
                
                # Fallback to manual JSON parsing
                resp = await llm_fast.ainvoke(msgs)
                try:
                    plan_data = json.loads(resp.content.strip())
                    plan_data['topic'] = topic  # Ensure topic is set
                    
                    # Validate required fields
                    if 'steps' not in plan_data or not plan_data['steps']:
                        plan_data['steps'] = [f"Research overview of {topic}", f"Analyze key aspects of {topic}"]
                    
                    state.plan = ResearchPlan(**plan_data)
                except (json.JSONDecodeError, ValueError) as parse_error:
                    logger.warning(f"JSON parsing failed: {parse_error}, using fallback")
                    # Final fallback
                    state.plan = ResearchPlan(
                        topic=topic,
                        steps=[
                            f"Research overview of {topic}",
                            f"Analyze key aspects of {topic}",
                            f"Review current developments in {topic}"
                        ],
                        estimated_time=10 + (state.depth * 3),
                        priority_areas=[f"Core {topic} concepts", f"{topic} applications"]
                    )
        else:
            state.plan = ResearchPlan(
                topic=topic,
                steps=[f"Overview of {topic}", f"Key aspects of {topic}", f"Latest developments in {topic}"],
                estimated_time=10,
            )
            
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        state.error_messages.append(f"Planning error: {e}")
        # Extract topic for fallback
        topic = state.topic[0] if isinstance(state.topic, list) else state.topic
        state.plan = ResearchPlan(
            topic=topic, 
            steps=[f"Basic research on {topic}"],
            estimated_time=5
        )
    return state

async def search_node(state: ResearchState) -> ResearchState:
    try:
        # Extract topic if it's a list
        topic = state.topic[0] if isinstance(state.topic, list) else state.topic
        steps = state.plan.steps if state.plan else [topic]
        log_execution(state.execution_id, "search", {"steps_count": len(steps)})

        async def run_query(step: str):
            try:
                query = f"{step} {topic}".strip()
                loop = asyncio.get_event_loop()
                items = await loop.run_in_executor(None, search_tool, query, state.depth * 2)
                return items if isinstance(items, list) else []
            except Exception as e:
                logger.warning(f"Search failed for step '{step}': {e}")
                return []

        to_search = steps[:max(3, state.depth + 1)]
        results_nested = await asyncio.gather(*[run_query(s) for s in to_search])
        seen = set()
        results = []
        for batch in results_nested:
            for item in batch:
                link = item.get("link")
                if link and link not in seen:
                    seen.add(link)
                    results.append(item)
                    print(f'links: {results}')
        state.search_results = results[:10]
    except Exception as e:
        logger.error(f"Search node failed: {e}")
        state.error_messages.append(f"Search error: {e}")
        state.search_results = []
    return state

async def content_fetch_node(state: ResearchState) -> ResearchState:
    try:
        log_execution(state.execution_id, "content_fetch", {"sources_count": len(state.search_results)})
        async with aiohttp.ClientSession() as session:
            fetched = []
            for r in state.search_results:
                try:
                    async with session.get(r["link"], timeout=5) as resp:
                        if resp.status == 200:
                            html = await resp.text()
                            soup = BeautifulSoup(html, "html.parser")
                            content = " ".join(p.get_text() for p in soup.find_all("p"))
                            fetched.append({
                                "url": r["link"],
                                "title": r["title"],
                                "content": content[:2000]
                            })
                        else:
                            fetched.append({
                                "url": r["link"],
                                "title": r["title"],
                                "content": r.get("snippet", "")
                            })
                except Exception as e:
                    logger.warning(f"Failed to fetch {r['link']}: {e}")
                    fetched.append({
                        "url": r["link"],
                        "title": r["title"],
                        "content": r.get("snippet", "")
                    })
        state.fetched_content = fetched
    except Exception as e:
        logger.error(f"Content fetch failed: {e}")
        state.error_messages.append(f"Content fetch error: {e}")
        state.fetched_content = []
    return state

async def summarization_node(state: ResearchState) -> ResearchState:
    try:
        # Extract topic if it's a list
        topic = state.topic[0] if isinstance(state.topic, list) else state.topic
        log_execution(state.execution_id, "summarization", {"content_items": len(state.fetched_content)})
        summaries: List[SourceSummary] = []
        
        if not state.fetched_content:
            state.summaries = []
            state.low_confidence = True
            return state

        async def summarize_item(item: Dict[str, Any]) -> Optional[SourceSummary]:
            try:
                payload = (
                    f"URL: {item.get('url','')}\n"
                    f"TITLE: {item.get('title','')}\n"
                    f"TEXT: {item.get('content','')}\n"
                    f"FOCUS: {topic}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Summarize the source relative to the user's topic. Return structured JSON."),
                    ("human", "{payload}")
                ])
                msgs = prompt.format_messages(payload=payload)
                
                try:
                    parser = llm_smart.with_structured_output(SourceSummary) if llm_smart else None
                    if parser:
                        s: SourceSummary = await parser.ainvoke(msgs)
                        s.source_url = item.get("url", s.source_url)
                        s.title = s.title or item.get("title", "Untitled Source")
                        if s.credibility_score is not None and s.credibility_score < 0.5:
                            state.low_confidence = True
                        return s
                except Exception as struct_error:
                    logger.warning(f"Structured summarization failed: {struct_error}")
                    # Fallback to basic summary
                    return SourceSummary(
                        source_url=item.get("url", ""),
                        title=item.get("title", "Untitled"),
                        summary=item.get("content", "")[:300] + "...",
                        key_insights=[],
                        credibility_score=0.6,
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to summarize source {item.get('url','')}: {e}")
                return None

        tasks = [summarize_item(it) for it in state.fetched_content[:8]]
        results = await asyncio.gather(*tasks)
        summaries = [r for r in results if r is not None]
        state.summaries = summaries
        if not summaries:
            state.low_confidence = True
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        state.error_messages.append(f"Summarization error: {e}")
        state.summaries = []
        state.low_confidence = True
    return state

async def synthesis_node(state: ResearchState) -> ResearchState:
    try:
        # Extract topic if it's a list
        topic = state.topic[0] if isinstance(state.topic, list) else state.topic
        log_execution(state.execution_id, "synthesis", {"summaries_count": len(state.summaries)})
        
        context_note = state.context_summary.context_relevance if state.context_summary else ""
        summaries_view = "\n\n".join(
            [
                f"Title: {s.title}\nURL: {s.source_url}\nSummary: {s.summary}\nInsights: {', '.join(s.key_insights)}"
                for s in state.summaries
            ]
        )
        processing_time = (datetime.now() - state.start_time).total_seconds()

        if llm_smart and state.summaries:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Write a structured research brief as strict JSON; do not include any extra keys."),
                    (
                        "human",
                        """
Create a comprehensive research brief on "{topic}" using the source material below.
Include: title, summary (3-4 sentences), sections (with heading/content/key_points), key_findings (list).

Context: {context}

Source Materials:
{materials}
""",
                    ),
                ])
                msgs = prompt.format_messages(topic=topic, context=context_note, materials=summaries_view[:4000])
                
                try:
                    draft_parser = llm_smart.with_structured_output(SynthesisDraft)
                    draft: SynthesisDraft = await draft_parser.ainvoke(msgs)
                except Exception as struct_error:
                    logger.warning(f"Structured synthesis failed: {struct_error}, using fallback")
                    # Fallback synthesis
                    draft = SynthesisDraft(
                        title=f"Research Brief: {topic}",
                        summary=f"Comprehensive analysis of {topic} based on multiple sources and research findings.",
                        sections=[
                            Section(
                                heading="Overview", 
                                content=state.summaries[0].summary if state.summaries else f"Research overview of {topic}",
                                key_points=[insight for s in state.summaries[:3] for insight in s.key_insights[:2]]
                            )
                        ],
                        key_findings=[s.title for s in state.summaries[:5]]
                    )

                references = [
                    Reference(title=s.title, url=s.source_url, relevance_score=s.credibility_score)
                    for s in state.summaries
                ]

                state.final_brief = FinalBrief(
                    title=draft.title,
                    summary=draft.summary,
                    sections=draft.sections,
                    references=references,
                    metadata={
                        "depth": state.depth,
                        "sources_count": len(state.summaries),
                        "follow_up": state.follow_up,
                        "key_findings": draft.key_findings,
                    },
                    processing_time=processing_time,
                )
            except Exception as synthesis_error:
                logger.error(f"Synthesis processing failed: {synthesis_error}")
                # Final fallback
                overview = state.summaries[0].summary if state.summaries else f"Research on {topic}"
                references = [Reference(title=s.title, url=s.source_url) for s in state.summaries]
                state.final_brief = FinalBrief(
                    title=f"Research Brief: {topic}",
                    summary=overview,
                    sections=[Section(heading="Overview", content=overview, key_points=[])],
                    references=references,
                    metadata={"depth": state.depth, "fallback": True, "synthesis_error": str(synthesis_error)},
                    processing_time=processing_time,
                )
        else:
            overview = state.summaries[0].summary if state.summaries else f"Research on {topic}"
            references = [Reference(title=s.title, url=s.source_url) for s in state.summaries]
            state.final_brief = FinalBrief(
                title=topic,
                summary=overview,
                sections=[Section(heading="Overview", content=overview, key_points=[])],
                references=references,
                metadata={"depth": state.depth, "fallback": True},
                processing_time=processing_time,
            )

        if state.user_id and state.final_brief:
            persist_brief(state.user_id, state.final_brief)

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        state.error_messages.append(f"Synthesis error: {e}")
        topic = state.topic[0] if isinstance(state.topic, list) else state.topic
        state.final_brief = FinalBrief(
            title=topic,
            summary=f"Error occurred during research on {topic}",
            sections=[],
            references=[],
            metadata={"error": str(e)},
            processing_time=(datetime.now() - state.start_time).total_seconds(),
        )
    return state

async def post_processing_node(state: ResearchState) -> ResearchState:
    try:
        log_execution(state.execution_id, "post_processing", {"final_brief_created": bool(state.final_brief)})
        if state.final_brief:
            total_text = state.final_brief.summary + " " + " ".join(s.content for s in state.final_brief.sections)
            est = estimate_tokens(total_text)
            state.final_brief.token_usage = {
                "estimated_total": est,
                "input_tokens": est // 2,
                "output_tokens": est // 2,
            }
            if not state.final_brief.sections:
                state.final_brief.sections = [Section(heading="Summary", content=state.final_brief.summary, key_points=[])]
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        state.error_messages.append(f"Post-processing error: {e}")
    return state

# ====== Conditional Logic ======
def should_include_context(state: ResearchState) -> str:
    return "planning" if not state.follow_up else "context"

def need_retry(state: ResearchState):
    if state.retries < 1 and (state.low_confidence or not state.summaries):
        state.retries += 1
        return "planning"
    return "synthesis"

# ====== LangGraph Creation ======
def create_research_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("context", context_summarization_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("search", search_node)
    workflow.add_node("fetch", content_fetch_node)
    workflow.add_node("summarize", summarization_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("post_process", post_processing_node)
    
    workflow.add_edge(START,'context')
    workflow.add_edge("context", "planning")
    workflow.add_edge("planning", "search")
    workflow.add_edge("search", "fetch")
    workflow.add_edge("fetch", "summarize")
    workflow.add_conditional_edges("summarize", need_retry, {"planning": "planning", "synthesis": "synthesis"})
    workflow.add_edge("synthesis", "post_process")
    workflow.add_edge("post_process", END)
    workflow.set_conditional_entry_point(should_include_context)
    
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

digraph = create_research_graph()
print('hey')
print(f'graph:{digraph}')
# ====== FastAPI Application ======
app = FastAPI(
    title="Research Brief Generator",
    description="Context-aware research assistant using LangGraph and LangChain",
)

@app.post("/brief", response_model=FinalBrief)

async def generate_brief(request: BriefRequest) -> FinalBrief:
    print('fsf')
    try:
        print('ssdfsf')
        print(request.topic,request.depth)
        initial_state = ResearchState(
            topic=request.topic,
            depth=request.depth,
            follow_up=request.follow_up,
            user_id=request.user_id,
        )
        print('sdsf')
        config = {
            "configurable": {"thread_id": f"{request.user_id}_{datetime.now().isoformat()}"},
            "run_name": "research_brief",
            "tags": [request.user_id, str(request.follow_up)],
        }
        print('sdaf')
        final_state_dict = await digraph.ainvoke(initial_state.model_dump(), config=config)
        print('sdaf')
        result_state = ResearchState(**final_state_dict)
        if not result_state.final_brief or result_state.error_messages:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to generate research brief",
                    "details": result_state.error_messages
                }
            )
        return result_state.final_brief
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e), "details": []})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/user/{user_id}/history")
async def get_user_history(user_id: str):
    history = USER_HISTORY.get(user_id, [])
    return {"user_id": user_id, "brief_count": len(history), "briefs": history}

@app.get("/traces/{execution_id}")
async def get_execution_trace(execution_id: str):
    trace = EXECUTION_TRACES.get(execution_id, {})
    return {"execution_id": execution_id, "trace": trace}

# ====== CLI Application ======
cli_app = typer.Typer(name="research-cli", help="Research Brief Generator CLI")

@cli_app.command()
def brief(
    topic: str = typer.Argument(..., help="Research topic"),
    depth: int = typer.Option(2, "--depth", "-d", help="Research depth (1-5)"),
    follow_up: bool = typer.Option(False, "--follow-up", "-f", help="Is this a follow-up query?"),
    user_id: str = typer.Option("cli_user", "--user", "-u", help="User ID"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Generate a research brief via CLI"""
    try:
        initial_state = ResearchState(topic=topic, depth=depth, follow_up=follow_up, user_id=user_id)
        config = {"configurable": {"thread_id": f"{user_id}_{datetime.now().isoformat()}"}}
        loop = asyncio.get_event_loop()
        final_state = loop.run_until_complete(digraph.ainvoke(initial_state.model_dump(), config=config))
        result_state = ResearchState(**final_state)
        if result_state.final_brief:
            brief_json = result_state.final_brief.model_dump_json(indent=2)
            if output:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(brief_json)
                typer.echo(f"Research brief saved to {output}")
            else:
                typer.echo(brief_json)
        else:
            typer.echo("Failed to generate research brief", err=True)
            if result_state.error_messages:
                for error in result_state.error_messages:
                    typer.echo(f"Error: {error}", err=True)
    except Exception as e:
        typer.echo(f"CLI error: {e}", err=True)
        raise typer.Exit(1)

@cli_app.command()
def history(user_id: str = typer.Argument(..., help="User ID")):
    history = USER_HISTORY.get(user_id, [])
    if not history:
        typer.echo(f"No research history found for user: {user_id}")
        return
    typer.echo(f"Research history for {user_id}:")
    for i, brief in enumerate(history, 1):
        when = brief.created_at.strftime('%Y-%m-%d %H:%M') if isinstance(brief.created_at, datetime) else str(brief.created_at)
        typer.echo(f"{i}. {brief.title} ({when})")

# ====== Main Entry Point ======
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] not in ["--help", "-h"]:
        cli_app()
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            log_level="info",
        )