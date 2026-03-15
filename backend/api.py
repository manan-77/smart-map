"""
FastAPI backend for Nav AI Assistant.

Provides the /chat endpoint that routes through the stateful supervisor agent.
Agent state (messages, routes, disambiguation) is persisted via PostgreSQL
checkpointing. Redis is used only for user↔session auth mapping.

Conversation management: conversations are indexed in the `conversations` table
and expire after 24 hours. Knowledge is extracted before expiry and stored
permanently in the `user_knowledge` table.
"""

import sys
from pathlib import Path

# Add parent directory to path so 'backend' package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
import uuid
import json
import re
import os
import logging
from sqlalchemy import text

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.agents.supervisor_agent import run_supervisor, call_gemini_api
from backend.auth.auth0 import get_current_user
from backend.services.session_manager import SessionManager
from backend.services.conversation_service import (
    upsert_conversation,
    get_user_conversations,
    get_conversation,
    delete_conversation,
    rename_conversation,
)
from backend.services.knowledge_service import (
    get_user_knowledge,
    build_knowledge_context,
    run_summarization,
)
from backend.services.expiry_job import start_expiry_scheduler, stop_expiry_scheduler
from backend.persistence.checkpointer import get_checkpointer, shutdown_pool
from backend.database.db import init_db, SessionLocal
from backend.tools.waze_tool import get_waze_alerts_and_jams
from backend.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

# ── Application Setup ────────────────────────

# Setup Rate Limiting
limiter = Limiter(key_func=get_remote_address)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    start_expiry_scheduler()
    yield
    # Shutdown
    stop_expiry_scheduler()
    shutdown_pool()

app = FastAPI(
    title="Nav AI Assistant API",
    version="3.0.0",
    description="AI-powered navigation assistant with stateful conversation support and user knowledge base",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize PostgreSQL checkpointer (creates tables on first run)
checkpointer = get_checkpointer()

# Create conversation & knowledge tables
init_db()

# Redis for user↔session auth mapping only
session_manager = SessionManager()

# Enable CORS — restrict to known frontend origins
_allowed = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_location: Optional[dict] = None  # {"lat": float, "lng": float}


class ChatResponse(BaseModel):
    message: str
    session_id: str
    route_data: Optional[dict] = None
    pois: Optional[list] = None
    location_candidates: Optional[list] = None
    alternative_routes: Optional[list] = None
    intent: Optional[str] = None


class AnalyzeRouteRequest(BaseModel):
    route_data: dict


class RenameConversationRequest(BaseModel):
    title: str


# ── Helper: get a DB session ─────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Endpoints ────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Main chat endpoint with Auth0 authentication.
    
    Routes through the stateful supervisor agent. Conversation state
    (messages, route data, disambiguation candidates) is automatically
    persisted and restored via the PostgreSQL checkpointer using session_id
    as the thread_id.
    """
    try:
        if len(body.message) > 2000:
            raise HTTPException(status_code=400, detail="Message too long (max 2000 characters)")
            
        user_id = current_user["user_id"]
        session_id = body.session_id or str(uuid.uuid4())
        
        # Map session to user (auth concern — stays in Redis)
        session_manager.save_user_mapping(session_id, user_id)
        
        # Upsert conversation record in the conversations table
        db = SessionLocal()
        try:
            upsert_conversation(db, session_id, user_id, body.message)
            
            # Build knowledge context — top 8 most relevant items
            knowledge_context = build_knowledge_context(db, user_id, body.message)
        finally:
            db.close()
        
        # Run stateful supervisor — checkpointer handles all state
        result = run_supervisor(
            user_message=body.message,
            session_id=session_id,
            checkpointer=checkpointer,
            location=body.user_location,
            user_id=user_id,
            knowledge_context=knowledge_context if knowledge_context else None,
        )
        
        return ChatResponse(
            message=result["message"],
            session_id=session_id,
            route_data=result.get("route_data"),
            pois=result.get("search_results"),
            location_candidates=result.get("location_candidates"),
            alternative_routes=result.get("alternative_routes"),
            intent=result.get("intent")
        )
    except HTTPException:
        raise
    except Exception as e:
        # Log the actual error, but return a generic 500 to avoid leaking internals
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing your request.")


# ── Conversation Management Endpoints ────────

@app.get("/conversations")
@limiter.limit("30/minute")
async def list_conversations(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """List all conversations for the authenticated user."""
    db = SessionLocal()
    try:
        user_id = current_user["user_id"]
        conversations = get_user_conversations(db, user_id)
        return {
            "conversations": [conv.to_dict() for conv in conversations]
        }
    finally:
        db.close()


@app.get("/conversations/{session_id}")
@limiter.limit("30/minute")
async def load_conversation(
    session_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Load a specific conversation with messages from the checkpointer.
    """
    db = SessionLocal()
    try:
        user_id = current_user["user_id"]
        conv = get_conversation(db, session_id, user_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages from checkpointer
        try:
            from langchain_core.messages import HumanMessage as HMsg, AIMessage as AMsg
            
            config = {"configurable": {"thread_id": session_id}}
            state = checkpointer.get(config)
            
            if state is None:
                return {
                    **conv.to_dict(),
                    "messages": [],
                    "route_data": None,
                }
            
            checkpoint_data = state
            channel_values = checkpoint_data.get("channel_values", {})
            messages = channel_values.get("messages", [])
            route_data = channel_values.get("route_data")
            alternative_routes = channel_values.get("alternative_routes")
            
            formatted_messages = []
            for msg in messages:
                # Only include user and assistant messages — skip
                # ToolMessage, SystemMessage, and other internal types
                if not isinstance(msg, (HMsg, AMsg)):
                    continue
                
                content = getattr(msg, 'content', '') or ''
                
                # Skip messages with empty content
                if not content.strip():
                    continue
                
                # Skip system context messages injected by knowledge system
                if content.startswith("[SYSTEM CONTEXT"):
                    # Extract only the user message part
                    marker = "User message: "
                    idx = content.find(marker)
                    if idx >= 0:
                        content = content[idx + len(marker):]
                    else:
                        continue
                
                role = "user" if isinstance(msg, HMsg) else "assistant"
                formatted_messages.append({
                    "role": role,
                    "content": content,
                })
            
            return {
                **conv.to_dict(),
                "messages": formatted_messages,
                "route_data": route_data,
                "alternative_routes": alternative_routes,
            }
        except Exception as e:
            print(f"Error loading conversation state: {e}")
            import traceback
            traceback.print_exc()
            return {
                **conv.to_dict(),
                "messages": [],
                "route_data": None,
            }
    finally:
        db.close()


@app.delete("/conversations/{session_id}")
@limiter.limit("15/minute")
async def remove_conversation(
    session_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Delete a conversation."""
    db = SessionLocal()
    try:
        user_id = current_user["user_id"]
        deleted = delete_conversation(db, session_id, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": True, "session_id": session_id}
    finally:
        db.close()


@app.patch("/conversations/{session_id}")
@limiter.limit("15/minute")
async def update_conversation(
    session_id: str,
    body: RenameConversationRequest,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Rename a conversation."""
    db = SessionLocal()
    try:
        user_id = current_user["user_id"]
        conv = rename_conversation(db, session_id, user_id, body.title)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv.to_dict()
    finally:
        db.close()


# ── Knowledge Endpoints ──────────────────────

@app.get("/knowledge")
@limiter.limit("30/minute")
async def get_knowledge(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Return the user's knowledge base."""
    db = SessionLocal()
    try:
        user_id = current_user["user_id"]
        items = get_user_knowledge(db, user_id)
        return {
            "knowledge": [item.to_dict() for item in items]
        }
    finally:
        db.close()


@app.post("/conversations/{session_id}/summarize")
@limiter.limit("10/minute")
async def summarize_conversation(
    session_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Trigger knowledge extraction for a finished conversation.
    Runs in a background thread so the response returns immediately.
    """
    import threading

    user_id = current_user["user_id"]

    def _bg_summarize():
        db = SessionLocal()
        try:
            run_summarization(db, session_id, user_id, checkpointer)
        except Exception as e:
            print(f"Background summarization error: {e}")
        finally:
            db.close()

    thread = threading.Thread(target=_bg_summarize, daemon=True)
    thread.start()

    return {"status": "summarizing", "session_id": session_id}


# ── Session History (legacy) ─────────────────

@app.get("/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get conversation history for a session.
    
    Retrieves state from the checkpointer (not Redis).
    """
    # Verify session belongs to user
    stored_user_id = session_manager.get_user_id(session_id)
    if stored_user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get state from checkpointer
    try:
        config = {"configurable": {"thread_id": session_id}}
        state = checkpointer.get(config)
        
        if state is None:
            return {"session_id": session_id, "messages": [], "route_data": None}
        
        # Extract messages from checkpoint state
        checkpoint_data = state
        channel_values = checkpoint_data.get("channel_values", {})
        messages = channel_values.get("messages", [])
        route_data = channel_values.get("route_data")
        
        # Format messages for API response
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": "user" if hasattr(msg, 'type') and msg.type == "human" else "assistant",
                "content": msg.content
            })
        
        return {
            "session_id": session_id,
            "messages": formatted_messages,
            "route_data": route_data
        }
    except Exception as e:
        print(f"Error getting history: {e}")
        return {"session_id": session_id, "messages": [], "route_data": None}


# ── Route Analysis ───────────────────────────

@app.post("/analyze-route")
@limiter.limit("5/minute")
async def analyze_route(
    body: AnalyzeRouteRequest,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze a route for traffic bottlenecks and fetch live Waze data.

    1. LLM identifies bottleneck segments from route instructions + road details
    2. Segments with risk_score >= 6 get a Waze API call (capped at 3)
    3. Returns aggregated alerts and jams for map rendering
    """
    try:
        route_data = body.route_data

        # Build compact route summary for the LLM
        instructions_summary = []
        for idx, instr in enumerate(route_data.get("detailed_instructions", [])):
            instructions_summary.append({
                "step": idx + 1,
                "text": instr.get("text", ""),
                "street": instr.get("street_name", ""),
                "distance_m": instr.get("distance_m", 0),
                "interval": instr.get("interval", []),
            })

        road_details = route_data.get("road_details", {})
        polyline = route_data.get("polyline", [])

        # Sample polyline to keep prompt small (every Nth point)
        sample_rate = max(1, len(polyline) // 40)
        sampled_polyline = polyline[::sample_rate]
        if polyline and sampled_polyline[-1] != polyline[-1]:
            sampled_polyline.append(polyline[-1])

        llm_prompt = f"""You are a traffic analyst. Given this route data, identify segments most likely to have congestion or incidents.

Route: {route_data.get('from', '?')} → {route_data.get('to', '?')}
Distance: {route_data.get('distance_km', '?')} km | Duration: {route_data.get('time_minutes', '?')} min

Step-by-step instructions:
{json.dumps(instructions_summary, indent=1)}

Road details:
- road_class: {json.dumps(road_details.get('road_class', []))}
- max_speed: {json.dumps(road_details.get('max_speed', []))}
- lanes: {json.dumps(road_details.get('lanes', []))}
- surface: {json.dumps(road_details.get('surface', []))}

Polyline coordinates (sampled, [lat, lng]):
{json.dumps(sampled_polyline)}

For each bottleneck, compute a bounding box from the polyline coords at the instruction interval indices, padded by ~0.01 degrees.

Return ONLY valid JSON:
{{
  "analysis_summary": "<2-3 sentence overview of the route's traffic risk>",
  "bottlenecks": [
    {{
      "description": "<what makes this a bottleneck>",
      "risk_score": <1-10>,
      "reason": "merge|intersection|road_class_change|speed_limit_drop|urban_dense|construction",
      "bounding_box": {{
        "bottom_left": "<lat>,<lng>",
        "top_right": "<lat>,<lng>"
      }}
    }}
  ]
}}

Focus on: motorway merges, road class downgrades, speed limit drops, dense urban areas, sharp turns at major intersections.
Return at most 5 bottlenecks. Return ONLY valid JSON, no markdown."""

        from langchain_core.messages import HumanMessage
        llm_response = call_gemini_api(
            [HumanMessage(content=llm_prompt)],
            purpose="route_bottleneck_analysis",
            reasoning_effort="high",
        )

        # Parse LLM JSON response
        response_text = llm_response.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            analysis = json.loads(response_text[json_start:json_end])
        else:
            analysis = {"analysis_summary": response_text, "bottlenecks": []}

        bottlenecks = analysis.get("bottlenecks", [])
        analysis_summary = analysis.get("analysis_summary", "")

        # Gate Waze API calls: only risk_score >= 6, capped at 3
        high_risk = sorted(
            [b for b in bottlenecks if b.get("risk_score", 0) >= 6],
            key=lambda x: x.get("risk_score", 0),
            reverse=True,
        )[:3]

        all_alerts = []
        all_jams = []
        waze_calls_made = 0

        for segment in high_risk:
            bb = segment.get("bounding_box", {})
            bl = bb.get("bottom_left", "")
            tr = bb.get("top_right", "")
            if not bl or not tr:
                continue

            waze_result = get_waze_alerts_and_jams(
                bottom_left=bl, top_right=tr, max_alerts=10, max_jams=10
            )
            waze_calls_made += 1
            all_alerts.extend(waze_result.get("alerts", []))
            all_jams.extend(waze_result.get("jams", []))

        return {
            "bottleneck_analysis": analysis_summary,
            "bottlenecks": bottlenecks,
            "alerts": all_alerts,
            "jams": all_jams,
            "waze_calls_made": waze_calls_made,
            "segments_analyzed": len(bottlenecks),
        }

    except Exception as e:
        logging.error(f"Analyze route error: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze route traffic. Please try again.")


# ── Health & Info ────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    status = {"status": "healthy"}
    
    # Check Redis
    try:
        session_manager.redis.ping()
        status["redis"] = "connected"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    # Check PostgreSQL
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            status["postgres"] = "connected"
        finally:
            db.close()
    except Exception as e:
        status["postgres"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    return status


@app.get("/")
async def root():
    return {
        "message": "Nav AI Assistant API",
        "version": "3.0.0",
        "features": [
            "stateful-conversations",
            "route-planning",
            "poi-search",
            "location-disambiguation",
            "conversation-management",
            "user-knowledge-base",
        ]
    }

# ── Voice: STT + TTS ─────────────────────────

def sanitize_text_for_tts(text: str) -> str:
    """Clean agent response text so it sounds natural when spoken by TTS.
    
    Strips coordinates, rounds decimals, removes markdown, etc.
    """
    # Remove coordinate patterns like "23.0225, 72.5714" or "(23.02, 72.57)"
    text = re.sub(r'\(?-?\d{1,3}\.\d{2,},\s*-?\d{1,3}\.\d{2,}\)?', '', text)
    # Remove phrases like "at coordinates ..." or "GPS: ..."
    text = re.sub(r'(at coordinates|GPS:?)\s*[\d.,\s()-]+', '', text, flags=re.IGNORECASE)
    
    # Round decimal distances: "25.3 km" → "25 km"
    def round_unit(m):
        return f"{round(float(m.group(1)))} {m.group(2)}"
    text = re.sub(r'(\d+\.\d+)\s*(km|miles?|mi|meters?|m|minutes?|min|hours?|hr)', round_unit, text)
    
    # Remove markdown bold/italic
    text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text)
    # Remove markdown bullet points
    text = re.sub(r'^\s*[•\-\*]\s+', '', text, flags=re.MULTILINE)
    # Convert numbered lists to flowing text
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Collapse multiple spaces/newlines
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text.strip()


class TTSRequest(BaseModel):
    text: str


@app.post("/stt")
@limiter.limit("10/minute")
async def speech_to_text(file: UploadFile = File(...), user=Depends(get_current_user), request: Request = None):
    """Transcribe audio using ElevenLabs Scribe v2."""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
    
    import requests as http_requests
    
    audio_bytes = await file.read()
    
    try:
        resp = http_requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            data={"model_id": "scribe_v2"},
            files={"file": (file.filename or "audio.webm", audio_bytes, file.content_type or "audio/webm")},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        return {"text": result.get("text", "")}
    except http_requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=resp.status_code, detail=f"ElevenLabs STT error: {str(e)}")
    except Exception as e:
        logging.error(f"STT Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse audio")


@app.post("/tts")
@limiter.limit("15/minute")
async def text_to_speech(req: TTSRequest, user=Depends(get_current_user), request: Request = None):
    """Convert text to speech using ElevenLabs TTS."""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
    
    import requests as http_requests
    
    clean_text = sanitize_text_for_tts(req.text)
    if not clean_text:
        raise HTTPException(status_code=400, detail="No text to speak after sanitization")
    
    voice_id = ELEVENLABS_VOICE_ID or "JBFqnCBsd6RMkjVDRZzb"
    
    try:
        resp = http_requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text": clean_text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        return Response(content=resp.content, media_type="audio/mpeg")
    except http_requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=resp.status_code, detail=f"ElevenLabs TTS error: {str(e)}")
    except Exception as e:
        logging.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate speech")


# ── Lifecycle (Deprecated) ───────────────────
# Note: Handled by 'lifespan' context manager above



if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Nav AI Assistant API (v3.0 - Knowledge)...")
    print("📍 Server running at http://localhost:8000")
    print("🔐 Auth0 authentication enabled")
    print("🗄️  PostgreSQL checkpointer active (stateful conversations)")
    print("💾 Redis for auth session mapping")
    print("🧠 User knowledge base active")
    print("⏰ Conversation expiry scheduler running (24hr)")
    print("💬 Ready to help with navigation!\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
