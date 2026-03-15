"""
Knowledge Service — Smart, safe, contradiction-aware knowledge extraction.

Key design principles:
1. SAFETY: Never infer personal locations from navigation requests.
   Only store explicitly stated facts. Behavioral patterns are safe.
2. CONTRADICTION DETECTION: Before inserting, the LLM reviews existing
   knowledge to avoid storing conflicting information.
3. DYNAMIC ENTITIES: No predefined categories — the LLM discovers patterns
   and creates whatever entity types capture the user's behavior.
"""

import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.models.knowledge import UserKnowledge
from backend.agents.supervisor_agent import call_gemini_api
from langchain_core.messages import HumanMessage
from backend.utils.logger import AgentLogger


EXTRACTION_PROMPT = """You are an intelligent knowledge extractor for a navigation assistant called Nav AI.

You will analyze a conversation between a user and a navigation assistant. Your job is to extract reusable knowledge about this user that will make future conversations more personalized and natural.

## CONVERSATION
{messages_text}

{route_info}

## EXISTING USER KNOWLEDGE
{existing_knowledge}

## SAFETY RULES (CRITICAL — FOLLOW STRICTLY)
1. NEVER infer personal locations (home, office, school) from navigation requests. 
   "Navigate to Thane" does NOT mean they live in Thane. Only extract a location if the user EXPLICITLY says "my home is in Thane" or "I live near Andheri".
2. DO extract tone of language, formality level, humor style, emoji usage, communication preferences — these are SAFE behavioral signals.
3. DO extract EXPLICITLY stated preferences ("I hate tolls", "I always take the highway").
4. DO NOT extract assumptions or guesses from a single mention.
5. Every item must have a safety_level:
   - "explicit": User directly stated it ("my office is in BKC", "I prefer avoiding tolls")
   - "inferred": Observed behavioral pattern (tone, communication style, frequent query types). 
     Inferred items need 2+ observations to be trusted—set initial confidence LOW (0.3-0.4).

## DYNAMIC ENTITY TYPES
You are FREE to create ANY knowledge entity type that captures useful patterns. DO NOT limit yourself to predefined categories. Discover patterns that are genuinely present in the conversation.

Examples (but create your own if you see different patterns):
- "communication_style" — formality, verbosity, emoji usage
- "language_preference" — casual/formal, short/detailed responses preferred
- "navigation_habit" — always asks about parking, prefers scenic routes
- "vehicle_info" — car type, prefers two-wheeler routing
- "time_pattern" — commutes at certain times, weekend trips
- "route_preference" — avoid tolls, prefer highways, avoid certain roads
- "emotional_tone" — patient, impatient, humorous
- Any other pattern you genuinely observe

## CONTRADICTION DETECTION
Check each new item against the EXISTING USER KNOWLEDGE above:
- If a new item CONTRADICTS an existing one (e.g., user now wants highways but old knowledge says avoid highways), mark action as "replace".
- If it REINFORCES existing knowledge, mark action as "reinforce".
- If it's entirely new, mark action as "create".

## OUTPUT FORMAT
Return ONLY valid JSON (no markdown, no explanation):
{{
    "summary": "<2-3 sentence summary of this conversation>",
    "knowledge": [
        {{
            "type": "<your dynamic entity type in snake_case>",
            "key": "<unique_snake_case_identifier>",
            "value": {{"description": "...", ...any structured fields}},
            "display_category": "personality|travel|places|preferences|patterns",
            "safety_level": "explicit|inferred",
            "confidence": 0.0,
            "action": "create|reinforce|replace",
            "replaces_key": "<key of existing item to replace, only if action=replace>"
        }}
    ]
}}

IMPORTANT:
- Only extract knowledge that is GENUINELY present — do not invent patterns.
- If NO useful knowledge can be extracted, return {{"summary": "...", "knowledge": []}}
- Be conservative. Quality over quantity. 3 high-quality items > 10 questionable ones.
"""


def extract_knowledge(messages_text: str, existing_knowledge_text: str, route_data: dict = None) -> dict:
    """
    Send conversation + existing knowledge to Gemini for smart extraction.
    Returns dict with "summary" and "knowledge" list.
    """
    route_info = ""
    if route_data:
        route_info = f"""ROUTE DATA:
- From: {route_data.get('from', '?')}
- To: {route_data.get('to', '?')}
- Distance: {route_data.get('distance_km', '?')} km
- Duration: {route_data.get('time_minutes', '?')} min"""

    existing_section = existing_knowledge_text if existing_knowledge_text else "(No existing knowledge — this is a new user)"

    prompt = EXTRACTION_PROMPT.format(
        messages_text=messages_text,
        route_info=route_info,
        existing_knowledge=existing_section,
    )

    try:
        AgentLogger.info("🧠 Running smart knowledge extraction via Gemini...")
        response = call_gemini_api(
            [HumanMessage(content=prompt)],
            purpose="knowledge_extraction",
            reasoning_effort="high",
        )

        response_text = response.strip()
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            result = json.loads(response_text[json_start:json_end])
        else:
            AgentLogger.error("Knowledge extraction returned no JSON")
            result = {"summary": "", "knowledge": []}

        knowledge_count = len(result.get("knowledge", []))
        AgentLogger.info(f"🧠 Extracted {knowledge_count} knowledge item(s)")
        return result

    except Exception as e:
        AgentLogger.error(f"Knowledge extraction failed: {e}")
        return {"summary": "", "knowledge": []}


def _format_existing_knowledge(db: Session, user_id: str) -> str:
    """Format existing knowledge as text for the LLM context."""
    items = (
        db.query(UserKnowledge)
        .filter(UserKnowledge.user_id == user_id)
        .order_by(UserKnowledge.confidence.desc())
        .all()
    )

    if not items:
        return ""

    lines = []
    for item in items:
        value_str = json.dumps(item.value) if isinstance(item.value, dict) else str(item.value)
        lines.append(
            f"- [{item.knowledge_type}] key=\"{item.key}\": {value_str} "
            f"(confidence: {item.confidence:.2f}, safety: {item.safety_level}, "
            f"seen {item.occurrence_count}x)"
        )

    return "\n".join(lines)


def merge_knowledge(db: Session, user_id: str, new_items: list, session_id: str):
    """
    Smart merge with 3 actions: create, reinforce, replace.
    Handles contradiction detection results from the LLM.
    """
    for item in new_items:
        k_type = item.get("type", "")
        k_key = item.get("key", "")
        k_value = item.get("value", {})
        k_confidence = item.get("confidence", 0.5)
        k_safety = item.get("safety_level", "inferred")
        k_display = item.get("display_category", "general")
        action = item.get("action", "create")
        replaces_key = item.get("replaces_key")

        if not k_type or not k_key:
            continue

        # Handle REPLACE action — delete the contradicted item first
        if action == "replace" and replaces_key:
            old = (
                db.query(UserKnowledge)
                .filter(
                    UserKnowledge.user_id == user_id,
                    UserKnowledge.key == replaces_key,
                )
                .first()
            )
            if old:
                AgentLogger.info(
                    f"🧠 Replacing contradicted knowledge: {old.knowledge_type}/{old.key}"
                )
                db.delete(old)
                db.flush()

        # Check if same (type, key) already exists
        existing = (
            db.query(UserKnowledge)
            .filter(
                UserKnowledge.user_id == user_id,
                UserKnowledge.knowledge_type == k_type,
                UserKnowledge.key == k_key,
            )
            .first()
        )

        if existing and action == "reinforce":
            # Reinforce existing knowledge
            existing.occurrence_count += 1
            existing.confidence = min(1.0, existing.confidence + 0.1)
            existing.updated_at = datetime.now(timezone.utc)

            # Smart merge value fields
            if isinstance(existing.value, dict) and isinstance(k_value, dict):
                merged = {**existing.value, **k_value}
                existing.value = merged
            else:
                existing.value = k_value

            # Upgrade safety if now explicit
            if k_safety == "explicit" and existing.safety_level == "inferred":
                existing.safety_level = "explicit"

            # Append session
            sources = existing.source_sessions or []
            if session_id not in sources:
                sources.append(session_id)
                existing.source_sessions = sources

            AgentLogger.info(
                f"🧠 Reinforced: {k_type}/{k_key} "
                f"(count={existing.occurrence_count}, confidence={existing.confidence:.2f})"
            )

        elif existing and action == "replace":
            # Replace existing with new values
            existing.value = k_value
            existing.confidence = k_confidence
            existing.safety_level = k_safety
            existing.occurrence_count = 1
            existing.updated_at = datetime.now(timezone.utc)
            existing.source_sessions = [session_id]

            AgentLogger.info(f"🧠 Replaced: {k_type}/{k_key} (contradiction resolved)")

        else:
            # Create new knowledge
            new_knowledge = UserKnowledge(
                user_id=user_id,
                knowledge_type=k_type,
                key=k_key,
                value=k_value,
                display_category=k_display,
                safety_level=k_safety,
                confidence=k_confidence,
                occurrence_count=1,
                source_sessions=[session_id],
            )
            db.add(new_knowledge)
            AgentLogger.info(
                f"🧠 New knowledge: {k_type}/{k_key} "
                f"(safety={k_safety}, confidence={k_confidence:.2f})"
            )

    db.commit()


def get_user_knowledge(db: Session, user_id: str) -> list[UserKnowledge]:
    """Retrieve all knowledge items for a user, sorted by confidence desc."""
    return (
        db.query(UserKnowledge)
        .filter(UserKnowledge.user_id == user_id)
        .order_by(UserKnowledge.confidence.desc())
        .all()
    )


import math

# Max knowledge items to inject into the system prompt
MAX_KNOWLEDGE_ITEMS = 8


def build_knowledge_context(db: Session, user_id: str, user_message: str = "") -> str:
    """
    Build text context from the user's knowledge base for system prompt injection.

    Scoring formula per item:
        score = confidence × recency_weight × log2(occurrence_count + 1) × relevance_boost

    - recency_weight decays over days since last use (1.0 → 0.3 over 30 days)
    - relevance_boost is 2.0 if the item's key/value matches keywords in the current message
    - Safety filter: explicit items need confidence >= 0.4; inferred need >= 0.5 AND 2+ occurrences
    - Only the top 8 items are injected.
    """
    items = (
        db.query(UserKnowledge)
        .filter(UserKnowledge.user_id == user_id)
        .all()
    )

    if not items:
        return ""

    now = datetime.now(timezone.utc)
    msg_lower = user_message.lower() if user_message else ""

    scored = []
    for item in items:
        # Safety gate
        if item.safety_level == "inferred":
            if item.confidence < 0.5 or item.occurrence_count < 2:
                continue
        elif item.safety_level == "explicit":
            if item.confidence < 0.4:
                continue

        # Recency weight: 1.0 for today → 0.3 after 30 days
        last_used = item.last_used_at or item.updated_at or item.created_at
        days_ago = max((now - last_used).total_seconds() / 86400, 0) if last_used else 0
        recency = max(0.3, 1.0 - (days_ago / 30) * 0.7)

        # Occurrence weight (logarithmic — diminishing returns)
        occ_weight = math.log2((item.occurrence_count or 1) + 1)

        # Relevance boost: check if key, type, or value description matches the message
        relevance = 1.0
        if msg_lower:
            key_words = item.key.replace("_", " ").lower().split()
            type_words = item.knowledge_type.replace("_", " ").lower().split()
            value_desc = ""
            if isinstance(item.value, dict):
                value_desc = str(item.value.get("description", "")).lower()

            match_words = key_words + type_words
            if any(w in msg_lower for w in match_words if len(w) > 2):
                relevance = 2.0
            elif value_desc and any(w in msg_lower for w in value_desc.split() if len(w) > 3):
                relevance = 1.5

        score = item.confidence * recency * occ_weight * relevance
        scored.append((score, item))

    if not scored:
        return ""

    # Sort by score descending, take top 8
    scored.sort(key=lambda x: x[0], reverse=True)
    top_items = scored[:MAX_KNOWLEDGE_ITEMS]

    # Update last_used_at for selected items
    for _, item in top_items:
        item.last_used_at = now
    try:
        db.commit()
    except Exception:
        db.rollback()

    lines = [f"Known information about this user (top {len(top_items)} from memory):"]
    for _, item in top_items:
        value_str = json.dumps(item.value) if isinstance(item.value, dict) else str(item.value)
        lines.append(f"- [{item.knowledge_type}] {item.key}: {value_str}")

    return "\n".join(lines)


def run_summarization(db: Session, session_id: str, user_id: str, checkpointer):
    """
    Run knowledge extraction for a finished conversation.
    Called when user clicks "New Chat" — processes the old conversation.
    """
    try:
        AgentLogger.info(f"🧠 Summarizing conversation {session_id[:12]}...")

        # 1. Load messages from checkpointer
        config = {"configurable": {"thread_id": session_id}}
        state = checkpointer.get(config)

        messages_text = ""
        route_data = None

        if state:
            # Handle both CheckpointTuple (has .checkpoint attr) and raw dict
            if hasattr(state, 'checkpoint'):
                checkpoint_data = state.checkpoint
            elif isinstance(state, dict) and 'channel_values' in state:
                checkpoint_data = state
            elif isinstance(state, dict):
                checkpoint_data = state.get('checkpoint', state)
            else:
                checkpoint_data = {}

            from langchain_core.messages import HumanMessage as HM, AIMessage
            channel_values = checkpoint_data.get("channel_values", {})
            messages = channel_values.get("messages", [])
            route_data = channel_values.get("route_data")

            lines = []
            for msg in messages:
                content = msg.content
                # Strip system context
                if content.startswith("[SYSTEM CONTEXT"):
                    marker = "User message: "
                    idx = content.find(marker)
                    if idx >= 0:
                        content = content[idx + len(marker):]
                    else:
                        continue
                if isinstance(msg, HM):
                    lines.append(f"User: {content}")
                elif isinstance(msg, AIMessage):
                    lines.append(f"Assistant: {content}")
            messages_text = "\n".join(lines)

        if not messages_text:
            AgentLogger.info(f"🧠 No messages in {session_id[:12]}, skipping")
            return

        # 2. Get existing knowledge for contradiction detection
        existing_text = _format_existing_knowledge(db, user_id)

        # 3. Extract knowledge via LLM
        result = extract_knowledge(messages_text, existing_text, route_data)

        # 4. Merge into user_knowledge
        knowledge_items = result.get("knowledge", [])
        if knowledge_items:
            merge_knowledge(db, user_id, knowledge_items, session_id)
            AgentLogger.info(
                f"🧠 Merged {len(knowledge_items)} knowledge items for user {user_id[:12]}..."
            )
        else:
            AgentLogger.info(f"🧠 No knowledge extracted from {session_id[:12]}")

        # 5. Mark conversation as summarized
        from backend.models.conversation import Conversation
        conv = db.query(Conversation).filter(Conversation.session_id == session_id).first()
        if conv:
            conv.is_summarized = True
            db.commit()

    except Exception as e:
        AgentLogger.error(f"🧠 Summarization failed for {session_id[:12]}: {e}")
        db.rollback()
