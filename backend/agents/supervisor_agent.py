"""
Supervisor Agent — Stateful multi-node LangGraph agent.

Uses PostgreSQL checkpointing to persist full state (messages, route data,
disambiguation candidates) between conversation turns via thread_id.

LLM Strategy:
  - Gemini 3 Flash (via Kie API): All reasoning — intent detection,
    disambiguation, conversation, route Q&A. Capability docs
    (backend/agents/docs/*.md) are loaded once and injected as a compact
    summary so the supervisor knows what each sub-agent can do.

Graph structure:
  router → routing_node | search_node | conversation_node | disambiguation_node → END
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from backend.models.state import SupervisorState
from backend.agents.routing_engine import routing_engine
from backend.agents.search_agent import run_search_agent
from backend.tools.location_search_tool import search_locations, format_distance
from backend.agents.capabilities import CAPABILITY_SUMMARY
from backend.config import KIE_API_KEY, KIE_BASE_URL
from backend.utils.logger import AgentLogger
from backend.utils.route_context import build_route_context
import json
import requests


# ──────────────────────────────────────────────
# LLM Helpers
# ──────────────────────────────────────────────

def call_gemini_api(messages, purpose: str = "", reasoning_effort: str = "high"):
    """
    Call Gemini 3 Flash via Kie API.

    Used for ALL LLM tasks: intent detection, disambiguation, conversation,
    route Q&A. The `reasoning_effort` parameter can be tuned per call-site:
      - "low":  fast, for simple classification / chat
      - "high": slower but better reasoning, for disambiguation / route Q&A

    (Kie API only supports "low" and "high" — no "medium".)
    """
    if reasoning_effort not in ("low", "high"):
        reasoning_effort = "high"

    headers = {
        "Authorization": f"Bearer {KIE_API_KEY}",
        "Content-Type": "application/json"
    }

    formatted_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "user"
        formatted_messages.append({
            "role": role,
            "content": [{"type": "text", "text": msg.content}]
        })

    payload = {
        "messages": formatted_messages,
        "reasoning_effort": reasoning_effort,
    }

    total_chars = sum(len(msg.content) for msg in messages)
    AgentLogger.api_call(
        f"Kie AI (Gemini 3 Flash | effort={reasoning_effort})",
        f"{KIE_BASE_URL}/gemini-3-flash/v1/chat/completions",
        model="gemini-3-flash",
        payload_size=total_chars,
    )

    response = requests.post(
        f"{KIE_BASE_URL}/gemini-3-flash/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=90,
    )
    data = response.json()

    if "choices" in data and data["choices"]:
        content = data["choices"][0].get("message", {}).get("content")
        if content:
            AgentLogger.api_response("Gemini 3 Flash", response.status_code, content[:200])
            return content
    if "data" in data and data["data"]:
        return str(data["data"])

    AgentLogger.error(f"Gemini API error (HTTP {response.status_code}): {json.dumps(data, indent=2)}")
    AgentLogger.error(f"Payload sent: messages={len(formatted_messages)}, effort={reasoning_effort}")
    raise Exception(f"Could not extract Gemini response. Keys: {list(data.keys())}")


# ──────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────

def _extract_pois_from_messages(messages) -> list:
    """Extract POI data from search agent ToolMessages for map markers."""
    from langchain_core.messages import ToolMessage
    pois = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                content = msg.content
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content
                
                # Handle list of POIs directly
                if isinstance(data, list):
                    for poi in data:
                        if isinstance(poi, dict) and "lat" in poi and "lng" in poi:
                            pois.append({
                                "name": poi.get("name", "Unnamed"),
                                "type": poi.get("type", "poi"),
                                "lat": poi["lat"],
                                "lng": poi["lng"],
                                "distance_km": poi.get("distance_km")
                            })
                # Handle dict with pois key
                elif isinstance(data, dict) and "pois" in data:
                    for poi in data["pois"]:
                        if isinstance(poi, dict) and "lat" in poi and "lng" in poi:
                            pois.append({
                                "name": poi.get("name", "Unnamed"),
                                "type": poi.get("type", "poi"),
                                "lat": poi["lat"],
                                "lng": poi["lng"],
                                "distance_km": poi.get("distance_km")
                            })
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
    return pois

def is_coordinates(location: str) -> bool:
    """Check if location string is GPS coordinates (lat,lng)."""
    try:
        parts = location.split(',')
        if len(parts) == 2:
            lat = float(parts[0].strip())
            lng = float(parts[1].strip())
            return -90 <= lat <= 90 and -180 <= lng <= 180
    except (ValueError, AttributeError):
        pass
    return False


def normalize_vehicle(raw: str) -> str:
    """Normalize free-form vehicle descriptions to routing profiles."""
    if not raw:
        return "car"
    value = str(raw).strip().lower()
    if value in {"walk", "walking", "on foot", "foot", "pedestrian"}:
        return "foot"
    if value in {"bike", "bicycle", "cycling", "cycle"}:
        return "bike"
    return "car"


def format_location_options(query: str, candidates: list) -> str:
    """Format location candidates as a conversational, voice-friendly message."""
    n = len(candidates)
    if n == 0:
        return f"I couldn't find any locations for '{query}'."

    # Build a flowing paragraph, closest first
    parts = [f"I found {n} places matching '{query}'."]

    for i, c in enumerate(candidates):
        name = c.get('name', 'Unnamed')
        address = c.get('address', '')
        dist = c.get('distance_text', '')

        # First candidate gets emphasis
        if i == 0:
            loc_desc = f"The closest is {name}"
        elif i <= 2:
            loc_desc = f"There's also {name}"
        else:
            # Summarize remaining
            remaining = n - i
            parts.append(f"Plus {remaining} more option{'s' if remaining > 1 else ''}.")
            break

        # Add address context (city/area only, not full address)
        if address:
            loc_desc += f" in {address}"
        if dist:
            loc_desc += f", about {dist} away"
        loc_desc += "."
        parts.append(loc_desc)

    parts.append("Which one did you mean? Just say a number or the area name.")
    return " ".join(parts)


def format_candidates_for_llm(candidates: list) -> str:
    """Format candidates as structured text for the LLM to reason about."""
    lines = []
    for c in candidates:
        dist = f" ({c.get('distance_text', '')})" if c.get('distance_text') else ""
        lines.append(f"  {c['id']}. {c['name']} — {c['address']}{dist}")
        lines.append(f"     Coordinates: ({c['coordinates']['lat']}, {c['coordinates']['lng']})")
        if c.get('city'):
            lines.append(f"     City: {c['city']}")
        if c.get('state'):
            lines.append(f"     State: {c['state']}")
    return "\n".join(lines)


def format_conversation_history(messages) -> str:
    """Format recent message history as context for the LLM."""
    history_lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            history_lines.append(f"Assistant: {content}")
    return "\n".join(history_lines) if history_lines else "(No prior conversation)"


def build_route_response(route_data: dict, prefix: str = "", alternatives: list = None) -> str:
    """Build a conversational, voice-friendly route response.

    Returns a plain-text message with no emojis, markdown, or visual references.
    The frontend still shows route cards, map markers and turn-by-turn panels
    via the structured route_data — this text is purely conversational.
    """
    dist = route_data.get('distance_km', '?')
    mins = route_data.get('time_minutes', '?')
    origin = route_data.get('from', '?')
    dest = route_data.get('to', '?')

    # Round minutes to nearest integer for natural speech
    if isinstance(mins, (int, float)):
        mins = round(mins)

    parts = []
    if prefix:
        parts.append(prefix.rstrip())

    parts.append(f"Your route from {origin} to {dest} is about {dist} km, roughly {mins} minutes.")

    if alternatives:
        n = len(alternatives)
        parts.append(f"I also found {n} alternate route{'s' if n > 1 else ''} if you're interested.")

    parts.append("Feel free to ask me anything about this route, like highway details, road conditions, or turn-by-turn directions.")
    return " ".join(parts)


# ──────────────────────────────────────────────
# Graph Nodes
# ──────────────────────────────────────────────

def _build_search_candidates(pois: list, user_location: dict = None) -> list:
    """Convert POI search results into disambiguation-compatible candidates."""
    candidates = []
    for idx, poi in enumerate(pois, 1):
        candidate = {
            "id": idx,
            "name": poi.get("name", "Unnamed"),
            "address": poi.get("address", ""),
            "coordinates": {"lat": poi["lat"], "lng": poi["lng"]},
            "type": poi.get("type", "poi"),
        }
        if poi.get("distance_km") is not None:
            candidate["distance_km"] = poi["distance_km"]
            candidate["distance_text"] = format_distance(poi["distance_km"])
        candidates.append(candidate)
    return candidates


def router_node(state: SupervisorState):
    """
    Main routing node — determines which handler to invoke.
    Uses GPT-5-2 for intent detection, informed by
    capability docs loaded from backend/agents/docs/.
    """
    AgentLogger.node_enter("router", f"Message: \"{state['messages'][-1].content[:80]}\"")

    messages = state["messages"]
    user_message = messages[-1].content
    pending_candidates = state.get("pending_candidates")
    user_location = state.get("location")

    # Show conversation context
    context_messages = messages[:-1]
    AgentLogger.conversation_context(context_messages)

    # ── Fast-path: disambiguation in progress ──
    if pending_candidates and pending_candidates.get("candidates"):
        num = len(pending_candidates["candidates"])
        AgentLogger.node_route("router", "disambiguation_node",
                               f"{num} pending candidates — GPT will interpret user response")
        return {"current_intent": "disambiguation"}

    # ── Fast-path: previous search results exist and user wants to pick/route ──
    # Convert search_results into pending_candidates so the disambiguation
    # node handles selection properly (supports "first one", "number 3",
    # "Chauhan Hospital", etc.) instead of brittle keyword matching.
    search_results = state.get("search_results")
    if search_results and len(search_results) > 0:
        msg_lower = user_message.lower()
        selection_signals = [
            "nearest", "closest", "first", "sabse pass", "subse pass",
            "pass wala", "paas wala", "pehla", "1st", "le jao", "lejao",
            "le chalo", "lechalo", "navigate", "take me", "number", "no.",
            "#", "wala", "2nd", "3rd", "second", "third", "doosra", "teesra",
        ]
        if any(kw in msg_lower for kw in selection_signals):
            candidates = _build_search_candidates(search_results, user_location)
            if candidates:
                AgentLogger.info(f"Promoting {len(candidates)} search results to disambiguation")
                AgentLogger.node_route("router", "disambiguation_node",
                                       "User selecting from search results")
                return {
                    "current_intent": "disambiguation",
                    "pending_candidates": {
                        "candidates": candidates,
                        "context": {
                            "location_a": f"{user_location['lat']},{user_location['lng']}" if user_location else "",
                            "location_b": "",
                            "ambiguous_field": "location_b",
                            "origin": "search",
                        },
                    },
                    "location_candidates": candidates,
                }

    # ── LLM-based intent detection (GPT-5-2) ──
    history_messages = messages[-21:-1] if len(messages) > 1 else []
    history_text = format_conversation_history(history_messages)

    has_route = bool(state.get("route_data"))
    route_note = ""
    if has_route:
        rd = state.get("route_data", {})
        route_note = (
            f'5. There is an ACTIVE ROUTE from {rd.get("from", "?")} to '
            f'{rd.get("to", "?")}. Questions about it → route_question.'
        )

    # ── Build system prompt from capability docs ──
    route_question_mapping = (
        '- "route_question" → route_question_node (ONLY when an active route exists)'
        if has_route else ""
    )
    route_question_intent = ' | "route_question"' if has_route else ""

    system_prompt = f"""You are the supervisor of a navigation assistant. Your ONLY job is to classify the user's message and route it to the correct sub-agent. You do NOT answer the user directly.

## Sub-agent capabilities (loaded from docs):

{CAPABILITY_SUMMARY}

## Intent → sub-agent mapping:
- "routing" → routing_node
- "search" → search_node
- "conversation" → conversation_node
{route_question_mapping}

## Critical routing rules:
1. GENERIC CATEGORY + proximity words ("nearest", "closest", "nearby", "near me", "pass mein", "sabse pass") → intent = **search**, NOT routing. Always.
   - "nearest hospital" → search (poi_type: "hospital")
   - "take me to the nearest petrol pump" → search (poi_type: "fuel") — find it first, route later
   - "sabse pass wala ATM" → search (poi_type: "atm")
2. SPECIFIC NAMED PLACE → intent = **routing**
   - "take me to Apollo Hospital Ahmedabad" → routing (location_b: "Apollo Hospital Ahmedabad")
   - "Delhi to Mumbai" → routing
3. Hindi/Hinglish follows the same rules.
4. If the user references a POI from a PREVIOUS search result by its specific name, that IS routing.
{route_note}"""

    user_prompt = f"""Conversation history:
{history_text}

Current user message: "{user_message}"

Respond ONLY with valid JSON (no markdown, no explanation):
{{
    "intent": "routing" | "search" | "conversation"{route_question_intent},
    "location_a": "start location or empty string",
    "location_b": "specific destination or empty string",
    "poi_type": "standard POI type or empty string (hospital, fuel, restaurant, cafe, atm, parking, hotel, pharmacy, supermarket, charging_station)",
    "vehicle": "car" | "bike" | "foot" | "",
    "avoid": [],
    "clarification_needed": false,
    "clarification_message": ""
}}"""

    AgentLogger.llm_prompt("Intent Detection (Gemini)", user_prompt, num_context_messages=len(history_messages))

    try:
        intent_response = call_gemini_api(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            purpose="intent_detection",
            reasoning_effort="low",
        )
        AgentLogger.llm_response("Intent Detection", intent_response)

        response_text = intent_response.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            intent_data = json.loads(response_text[json_start:json_end])
        else:
            raise json.JSONDecodeError("No JSON found", response_text, 0)

        AgentLogger.llm_parsed_intent(intent_data)
        intent = intent_data.get("intent", "conversation")

        if intent_data.get("clarification_needed"):
            AgentLogger.node_exit("router", "clarification")
            return {
                "messages": [AIMessage(content=intent_data["clarification_message"])],
                "current_intent": "clarification"
            }

        result = {"current_intent": intent}

        if intent == "routing":
            location_a = intent_data.get("location_a", "")
            location_b = intent_data.get("location_b", "")
            result["_routing_params"] = {"location_a": location_a, "location_b": location_b}

            vehicle_raw = intent_data.get("vehicle", "") or ""
            avoid_raw = intent_data.get("avoid", []) or []
            prefs = {
                "vehicle": normalize_vehicle(vehicle_raw),
                "avoid": avoid_raw,
            }
            result["routing_preferences"] = prefs
            AgentLogger.state_update("routing_preferences", prefs)
            AgentLogger.node_route("router", "routing_node", f"{location_a or '(GPS)'} → {location_b}")

        elif intent == "search":
            result["_search_params"] = {"poi_type": intent_data.get("poi_type", "")}
            AgentLogger.node_route("router", "search_node", f"POI: {intent_data.get('poi_type', '')}")

        elif intent == "route_question":
            AgentLogger.node_route("router", "route_question_node", "Route Q&A")

        else:
            AgentLogger.node_route("router", "conversation_node", "General conversation")

        AgentLogger.node_exit("router", intent)
        return result

    except json.JSONDecodeError as e:
        AgentLogger.error(f"Intent parse failed: {str(e)}")
        return {
            "messages": [AIMessage(content=f"Intent detection failed due to invalid JSON from the intent model: {e}")],
            "current_intent": "error",
        }
    except Exception as e:
        AgentLogger.error(f"Router error: {str(e)}")
        return {
            "messages": [AIMessage(content=f"Router error: {e}")],
            "current_intent": "error"
        }


def routing_node(state: SupervisorState):
    """Handle routing requests — geocode, disambiguate, or compute route."""
    AgentLogger.node_enter("routing_node")
    
    user_location = state.get("location")
    routing_prefs = state.get("routing_preferences") or {}
    routing_params = state.get("_routing_params", {})
    location_a = routing_params.get("location_a", "")
    location_b = routing_params.get("location_b", "")
    
    # Only use GPS as fallback if location_a is truly empty AND not coming from disambiguation
    # Check if we have pending_candidates context that might have preserved location_a
    pending_candidates = state.get("pending_candidates")
    if pending_candidates and pending_candidates.get("context"):
        context = pending_candidates["context"]
        # If location_a was preserved in context, use it
        if context.get("location_a") and not location_a:
            location_a = context["location_a"]
            AgentLogger.info(f"Restored location_a from disambiguation context: {location_a}")
    
    # Only default to GPS if location_a is still empty
    if not location_a and user_location:
        location_a = f"{user_location['lat']},{user_location['lng']}"
        AgentLogger.info("Using user GPS as start location")
    
    if not location_a or not location_b:
        AgentLogger.node_exit("routing_node", "clarification")
        return {
            "messages": [AIMessage(content="I need both a starting location and destination. Could you provide both?")],
            "current_intent": "clarification"
        }
    
    AgentLogger.info(f"Route request: {location_a} → {location_b}")
    
    # Search location_a (if not coordinates)
    if not is_coordinates(location_a):
        AgentLogger.tool_call("search_locations", {"query": location_a, "limit": 5})
        search_result_a = search_locations.invoke({"query": location_a, "user_location": user_location, "limit": 5})
        AgentLogger.tool_result("search_locations", search_result_a)
        
        if search_result_a.get("needs_disambiguation"):
            candidates = search_result_a["locations"]
            AgentLogger.disambiguation_candidates(location_a, len(candidates), candidates)
            response = format_location_options(location_a, candidates)
            AgentLogger.node_exit("routing_node", "disambiguation")
            return {
                "messages": [AIMessage(content=response)],
                "current_intent": "disambiguation",
                "pending_candidates": {
                    "candidates": candidates,
                    "context": {"location_a": location_a, "location_b": location_b, "ambiguous_field": "location_a"}
                },
                "location_candidates": candidates
            }
        
        if search_result_a.get("found") and search_result_a["locations"]:
            loc_a = search_result_a["locations"][0]
            location_a = f"{loc_a['coordinates']['lat']},{loc_a['coordinates']['lng']}"
            AgentLogger.info(f"Resolved start → ({location_a})")
    
    # Search location_b
    AgentLogger.tool_call("search_locations", {"query": location_b, "limit": 5})
    search_result_b = search_locations.invoke({"query": location_b, "user_location": user_location, "limit": 5})
    AgentLogger.tool_result("search_locations", search_result_b)
    
    if search_result_b.get("needs_disambiguation"):
        candidates = search_result_b["locations"]
        AgentLogger.disambiguation_candidates(location_b, len(candidates), candidates)
        response = format_location_options(location_b, candidates)
        AgentLogger.node_exit("routing_node", "disambiguation")
        return {
            "messages": [AIMessage(content=response)],
            "current_intent": "disambiguation",
            "pending_candidates": {
                "candidates": candidates,
                "context": {"location_a": location_a, "location_b": location_b, "ambiguous_field": "location_b"}
            },
            "location_candidates": candidates
        }
    
    if search_result_b.get("found") and search_result_b["locations"]:
        loc_b = search_result_b["locations"][0]
        location_b = f"{loc_b['coordinates']['lat']},{loc_b['coordinates']['lng']}"
        AgentLogger.info(f"Resolved destination → ({location_b})")
    
    # Compute route
    vehicle = normalize_vehicle(routing_prefs.get("vehicle"))
    AgentLogger.tool_call("routing_engine", {"from": location_a, "to": location_b, "vehicle": vehicle})
    try:
        route_data = routing_engine(location_a, location_b, vehicle=vehicle)
        AgentLogger.state_update("route_data", route_data)
        alternatives = route_data.get("alternative_routes", [])
        
        # Build route context for conversational Q&A
        route_context = build_route_context(route_data)
        AgentLogger.info(f"Built route context ({len(route_context)} chars)")
        
        # Build conversational response (no emojis/markdown)
        response = build_route_response(route_data, prefix="Got it!", alternatives=alternatives)
        
        AgentLogger.agent_response(response)
        AgentLogger.node_exit("routing_node", "routing")
        return {
            "messages": [AIMessage(content=response)],
            "route_data": route_data,
            "route_context": route_context,
            "alternative_routes": alternatives,
            "current_intent": "routing"
        }
    except Exception as e:
        AgentLogger.error(f"Routing failed: {str(e)}")
        return {
            "messages": [AIMessage(content="Sorry, I couldn't find a route. Please check the location names and try again.")],
            "current_intent": "error"
        }


def search_node(state: SupervisorState):
    """Handle POI search requests — extract POI markers for the map."""
    AgentLogger.node_enter("search_node")

    user_message = state["messages"][-1].content
    search_params = state.get("_search_params", {})
    poi_type = search_params.get("poi_type", "")
    route_data = state.get("route_data")
    location = state.get("location")

    AgentLogger.info(f"Searching for: {poi_type}")

    try:
        result = run_search_agent(user_message, route_data=route_data, location=location)
        agent_response = result["messages"][-1].content

        pois = _extract_pois_from_messages(result.get("messages", []))
        if pois:
            AgentLogger.info(f"Extracted {len(pois)} POIs for map markers")

        # Append a hint so the user knows they can pick one
        if pois and len(pois) > 1:
            agent_response += "\n\nWant me to navigate to any of these? Just tell me which one."

        AgentLogger.agent_response(agent_response)
        AgentLogger.node_exit("search_node", "search")
        return {
            "messages": [AIMessage(content=agent_response)],
            "current_intent": "search",
            "search_results": pois,
        }
    except Exception as e:
        AgentLogger.error(f"Search failed: {str(e)}")
        return {
            "messages": [AIMessage(content="Sorry, I couldn't complete the search. Please try again.")],
            "current_intent": "error"
        }


def conversation_node(state: SupervisorState):
    """Handle general conversation — uses GPT-5-2 for quality responses."""
    AgentLogger.node_enter("conversation_node")
    
    messages = state["messages"]
    user_message = messages[-1].content
    context_messages = messages[-11:-1] if len(messages) > 1 else []
    history_text = format_conversation_history(context_messages)
    
    route_hint = ""
    if state.get("route_data"):
        rd = state["route_data"]
        route_hint = f"\n\nNote: There is an active route from {rd.get('from', '?')} to {rd.get('to', '?')} ({rd.get('distance_km', '?')} km). If the user asks about it, suggest they can ask questions about the route details."
    
    conversation_prompt = f"""You are a friendly navigation assistant named Nav AI.

Your capabilities:
- Find routes between any two locations worldwide
- Search for places like gas stations, restaurants, hotels, parking, etc.
- Provide navigation assistance and travel information
- Answer detailed questions about active routes (highways, lanes, turns, road types){route_hint}

Conversation history:
{history_text}

Current user message: {user_message}

IMPORTANT response formatting rules:
- Respond in short, natural spoken sentences as if you were a friendly co-pilot.
- NEVER use emojis, markdown formatting (no **bold**, no bullet points, no numbered lists).
- NEVER reference visual elements like "shown on the map" or "click here".
- Keep responses concise and conversational."""
    
    AgentLogger.llm_prompt("Conversation (Gemini)", conversation_prompt, num_context_messages=len(context_messages))
    
    try:
        response_text = call_gemini_api([HumanMessage(content=conversation_prompt)], purpose="conversation", reasoning_effort="low")
        AgentLogger.llm_response("Conversation", response_text)
        AgentLogger.agent_response(response_text)
        AgentLogger.node_exit("conversation_node", "conversation")
        return {
            "messages": [AIMessage(content=response_text)],
            "current_intent": "conversation"
        }
    except Exception as e:
        AgentLogger.error(f"Conversation error: {str(e)}")
        return {
            "messages": [AIMessage(content="I'm here to help with navigation! Ask me for directions or to find places nearby.")],
            "current_intent": "error"
        }


def route_question_node(state: SupervisorState):
    """
    Handle questions about the active route — uses GPT-5-2 with the full
    route context document to provide data-backed answers.
    """
    AgentLogger.node_enter("route_question_node")
    
    messages = state["messages"]
    user_message = messages[-1].content
    route_context = state.get("route_context", "")
    route_data = state.get("route_data", {})
    
    # If no route context, rebuild it from route_data
    if not route_context and route_data:
        route_context = build_route_context(route_data)
        AgentLogger.info(f"Rebuilt route context ({len(route_context)} chars)")
    
    if not route_context:
        AgentLogger.node_exit("route_question_node", "no_route")
        return {
            "messages": [AIMessage(content="I don't have an active route to answer questions about. Would you like me to find a route first?")],
            "current_intent": "conversation"
        }
    
    # Build conversation context
    context_messages = messages[-6:-1] if len(messages) > 1 else []
    history_text = format_conversation_history(context_messages)
    
    route_qa_prompt = f"""You are a navigation assistant with detailed knowledge of the user's current route.

Below is the COMPLETE route data. Use ONLY this data to answer the user's question.
Do NOT make up information not present in the data. If the data doesn't contain what they're asking about, say so.

--- ROUTE DATA ---
{route_context}
--- END ROUTE DATA ---

Conversation history:
{history_text}

User's question: "{user_message}"

IMPORTANT response formatting rules:
- Answer in plain conversational sentences, like a helpful navigator sitting next to the user.
- NEVER use emojis or markdown formatting (no **bold**, no bullet points, no numbered lists).
- Use commas and natural sentence flow instead of lists.
- Be precise with numbers and road names from the data but keep the tone natural."""
    
    AgentLogger.llm_prompt("Route Q&A (Gemini)", route_qa_prompt, num_context_messages=len(context_messages))
    
    try:
        response_text = call_gemini_api([HumanMessage(content=route_qa_prompt)], purpose="route_question", reasoning_effort="high")
        AgentLogger.llm_response("Route Q&A", response_text)
        AgentLogger.agent_response(response_text)
        AgentLogger.node_exit("route_question_node", "route_question")
        return {
            "messages": [AIMessage(content=response_text)],
            "current_intent": "route_question"
        }
    except Exception as e:
        AgentLogger.error(f"Route Q&A error: {str(e)}")
        return {
            "messages": [AIMessage(content="I had trouble analyzing the route data. Could you try rephrasing your question?")],
            "current_intent": "error"
        }


def disambiguation_node(state: SupervisorState):
    """
    LLM-powered disambiguation — uses GPT-5-2 to understand user responses naturally.
    
    GPT can handle:
    - Direct selections: "2", "the first one", "KK Nagar in Chennai"
    - Follow-up questions: "is there any in ahmedabad?", "which is closest?"
    - Re-search requests: "search for KK Nagar in Gujarat instead"
    - Abandoning: "never mind", "let's go somewhere else"
    """
    AgentLogger.node_enter("disambiguation_node")
    
    messages = state["messages"]
    user_message = messages[-1].content
    user_location = state.get("location")
    pending_candidates = state.get("pending_candidates", {})
    candidates = pending_candidates.get("candidates", [])
    context = pending_candidates.get("context", {})
    
    AgentLogger.info(f"User response: \"{user_message}\"")
    AgentLogger.info(f"Candidates: {len(candidates)}")
    
    # Format candidates for GPT
    candidates_text = format_candidates_for_llm(candidates)
    
    # Build conversation context
    recent_messages = messages[-6:-1] if len(messages) > 1 else []
    history_text = format_conversation_history(recent_messages)
    
    # Build user location context
    user_location_context = ""
    if user_location and user_location.get("lat") and user_location.get("lng"):
        user_location_context = f"\n\nIMPORTANT — The user is currently located at GPS coordinates ({user_location['lat']}, {user_location['lng']}). When they select a location or ask about options, STRONGLY prefer locations that are geographically close to them (same country/region). Do NOT select locations in a different country unless the user explicitly asks for it."
    
    origin = context.get("origin", "geocoding")
    origin_note = ""
    if origin == "search":
        origin_note = """
NOTE: These candidates came from a nearby POI search (OpenStreetMap), so they are real local results sorted by distance. The user likely wants to navigate to one of them. If they say "nearest", "first", "sabse pass wala", "1", etc., select ID 1 (the closest).
"""
    else:
        origin_note = """
NOTE: These candidates came from geocoding (place name search), which can return globally spread results. Check distances carefully.
"""

    disambiguation_prompt = f"""You are a navigation assistant helping a user choose a location.{user_location_context}
{origin_note}
The user was searching for a place and I found multiple locations. Here are the candidates:

{candidates_text}

Recent conversation:
{history_text}

The user just said: "{user_message}"

Analyze the user's response and determine their intent. Respond with ONLY valid JSON:

{{
    "action": "select" | "question" | "re_search" | "abandon",
    "selected_id": <MUST be an exact ID number from the candidates list above (1-{len(candidates)}). Only set when action is "select". null otherwise>,
    "answer": "<natural language response — plain conversational text, NO emojis, NO markdown, NO bullet points>",
    "new_search_query": "<new search query if action is re_search, null otherwise>"
}}

CRITICAL: When action is "select", the "selected_id" MUST exactly match one of the candidate IDs listed above. Double-check that the ID corresponds to the correct location name and address before responding.
RESPONSE FORMAT: The "answer" field must be written as natural spoken English. Never use emojis, markdown, or bullet points. Keep it brief and conversational, like a co-pilot talking to the driver.

Rules:
- "select": User is picking a specific location (by number, name, description, city, or other identifier). The selected_id MUST match the candidate's ID number.
- "question": User is asking about the candidates (e.g. "which is closest?", "is there one in X city?"). Answer helpfully based on the candidate data. If they ask about a city/area not in the list, tell them and suggest re-searching.
- "re_search": User wants to search differently (e.g. "search in Ahmedabad instead", "find one near me"). USE THIS when:
  * The user says "nearest to me" / "closest" / "pass wala" BUT all candidates are very far (>50 km away from the user) — suggest a POI search instead
  * The user explicitly asks to search differently
  * None of the candidates seem to match what the user actually wants
- "abandon": User wants to do something else entirely (e.g. "never mind", "take me to Delhi instead")

DISTANCE SANITY CHECK (for geocoding results only): If the user asks for the "nearest" or "closest" option, and even the closest candidate is hundreds of km away, that means the geocoding search failed to find local results. In this case, use "re_search" and suggest the user try a POI search (e.g., "hospital near me") rather than selecting a faraway location. A "nearest hospital" should be within ~10 km, not 1000+ km. This does NOT apply to POI search results, which are already local.

For "select": Confirm the selection with the EXACT name and address from the candidates list.
For "question": Answer their question, then ask which location they'd like."""

    AgentLogger.llm_prompt("Disambiguation (Gemini)", disambiguation_prompt, num_context_messages=len(recent_messages))
    
    try:
        gemini_response = call_gemini_api([HumanMessage(content=disambiguation_prompt)], purpose="disambiguation", reasoning_effort="high")
        AgentLogger.llm_response("Disambiguation", gemini_response)
        
        response_text = gemini_response.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            decision = json.loads(response_text[json_start:json_end])
        else:
            raise json.JSONDecodeError("No JSON found", response_text, 0)
        
        action = decision.get("action", "question")
        answer = decision.get("answer", "")
        
        AgentLogger.info(f"GPT decision: action={action}")
        
        # ── ACTION: SELECT ──
        if action == "select":
            selected_id = decision.get("selected_id")
            if selected_id and 1 <= selected_id <= len(candidates):
                # Safe lookup: find by ID field, not array index
                selected = None
                for c in candidates:
                    if c.get("id") == selected_id:
                        selected = c
                        break
                if selected is None:
                    # Fallback to array index
                    selected = candidates[selected_id - 1]
                
                AgentLogger.disambiguation_selected(selected)
                
                # Build route with selected location
                location_a = context.get("location_a", "")
                location_b = context.get("location_b", "")
                
                selected_coords = f"{selected['coordinates']['lat']},{selected['coordinates']['lng']}"
                if context.get("ambiguous_field") == "location_a":
                    location_a = selected_coords
                else:
                    location_b = selected_coords
                
                # Only use GPS if location_a is EMPTY (not just non-coordinates)
                # If location_a has a value (place name), keep it - routing_node will geocode it
                if not location_a and user_location:
                    location_a = f"{user_location['lat']},{user_location['lng']}"
                
                routing_prefs = state.get("routing_preferences") or {}
                vehicle = normalize_vehicle(routing_prefs.get("vehicle"))
                AgentLogger.tool_call("routing_engine", {"from": location_a, "to": location_b, "vehicle": vehicle})
                
                try:
                    route_data = routing_engine(location_a, location_b, vehicle=vehicle)
                    AgentLogger.state_update("route_data", route_data)
                    AgentLogger.state_update("pending_candidates", "cleared")
                    alternatives = route_data.get("alternative_routes", [])
                    
                    # Build route context for conversational Q&A
                    route_context = build_route_context(route_data)
                    AgentLogger.info(f"Built route context ({len(route_context)} chars)")
                    
                    response = build_route_response(route_data, prefix=answer, alternatives=alternatives)
                    
                    AgentLogger.agent_response(response)
                    AgentLogger.node_exit("disambiguation_node", "routing")
                    
                    return {
                        "messages": [AIMessage(content=response)],
                        "route_data": route_data,
                        "route_context": route_context,
                        "alternative_routes": alternatives,
                        "current_intent": "routing",
                        "pending_candidates": None,
                        "location_candidates": None
                    }
                except Exception as e:
                    AgentLogger.error(f"Routing after selection failed: {str(e)}")
                    return {
                        "messages": [AIMessage(content=f"{answer}\n\nHowever, I couldn't calculate the route. Please try again.")],
                        "current_intent": "error",
                        "pending_candidates": None,
                        "location_candidates": None
                    }
            else:
                # GPT said select but invalid ID — treat as question
                AgentLogger.error(f"Invalid selected_id: {selected_id}")
                action = "question"
        
        # ── ACTION: QUESTION ──
        if action == "question":
            AgentLogger.agent_response(answer)
            AgentLogger.node_exit("disambiguation_node", "disambiguation")
            # Keep pending_candidates — user hasn't chosen yet
            return {
                "messages": [AIMessage(content=answer)],
                "current_intent": "disambiguation"
            }
        
        # ── ACTION: RE-SEARCH ──
        if action == "re_search":
            new_query = decision.get("new_search_query", "")
            if new_query:
                AgentLogger.info(f"Re-searching with: \"{new_query}\"")
                AgentLogger.tool_call("search_locations", {"query": new_query, "limit": 5})
                
                search_result = search_locations.invoke({
                    "query": new_query,
                    "user_location": user_location,
                    "limit": 5
                })
                AgentLogger.tool_result("search_locations", search_result)
                
                if search_result.get("found") and search_result["locations"]:
                    new_candidates = search_result["locations"]
                    
                    if len(new_candidates) == 1 or not search_result.get("needs_disambiguation"):
                        # Single clear result — route directly
                        chosen = new_candidates[0]
                        location_a = context.get("location_a", "")
                        location_b = context.get("location_b", "")
                        
                        selected_coords = f"{chosen['coordinates']['lat']},{chosen['coordinates']['lng']}"
                        if context.get("ambiguous_field") == "location_a":
                            location_a = selected_coords
                        else:
                            location_b = selected_coords
                        
                        # Only use GPS if location_a is EMPTY (not just non-coordinates)
                        if not location_a and user_location:
                            location_a = f"{user_location['lat']},{user_location['lng']}"
                        
                        try:
                            routing_prefs = state.get("routing_preferences") or {}
                            vehicle = normalize_vehicle(routing_prefs.get("vehicle"))
                            route_data = routing_engine(location_a, location_b, vehicle=vehicle)
                            alternatives = route_data.get("alternative_routes", [])
                            prefix = f"Found it! {chosen['name']} in {chosen['address']}."
                            response = build_route_response(route_data, prefix=prefix, alternatives=alternatives)
                            
                            AgentLogger.agent_response(response)
                            return {
                                "messages": [AIMessage(content=response)],
                                "route_data": route_data,
                                "alternative_routes": alternatives,
                                "current_intent": "routing",
                                "pending_candidates": None,
                                "location_candidates": None
                            }
                        except Exception as e:
                            AgentLogger.error(f"Route after re-search failed: {str(e)}")
                    
                    # Multiple results — update candidates
                    response_msg = format_location_options(new_query, new_candidates)
                    AgentLogger.disambiguation_candidates(new_query, len(new_candidates), new_candidates)
                    
                    return {
                        "messages": [AIMessage(content=f"{answer}\n\n{response_msg}")],
                        "current_intent": "disambiguation",
                        "pending_candidates": {
                            "candidates": new_candidates,
                            "context": context  # Keep original routing context
                        },
                        "location_candidates": new_candidates
                    }
                else:
                    return {
                        "messages": [AIMessage(content=f"I couldn't find any results for '{new_query}'. Would you like to choose from the original options?")],
                        "current_intent": "disambiguation"
                    }
            else:
                return {
                    "messages": [AIMessage(content=answer)],
                    "current_intent": "disambiguation"
                }
        
        # ── ACTION: ABANDON ──
        if action == "abandon":
            AgentLogger.info("User abandoned disambiguation")
            AgentLogger.node_exit("disambiguation_node", "abandoned")
            return {
                "messages": [AIMessage(content=answer)],
                "current_intent": "conversation",
                "pending_candidates": None,
                "location_candidates": None
            }
        
        # Fallback
        return {
            "messages": [AIMessage(content=answer or "Could you clarify which location you'd like?")],
            "current_intent": "disambiguation"
        }
    
    except Exception as e:
        AgentLogger.error(f"Disambiguation error: {str(e)}")
        return {
            "messages": [AIMessage(content="I had trouble understanding that. Could you pick a number from the list, or tell me what you'd like to do?")],
            "current_intent": "disambiguation"
        }


# ──────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────

def _route_by_intent(state: SupervisorState) -> str:
    intent = state.get("current_intent", "conversation")
    if intent == "routing":
        return "routing_node"
    elif intent == "search":
        return "search_node"
    elif intent == "disambiguation":
        return "disambiguation_node"
    elif intent == "route_question":
        return "route_question_node"
    elif intent in ("error", "clarification"):
        return END
    else:
        return "conversation_node"


def create_supervisor_agent(checkpointer=None):
    """Create the stateful supervisor agent graph."""
    workflow = StateGraph(SupervisorState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("routing_node", routing_node)
    workflow.add_node("search_node", search_node)
    workflow.add_node("conversation_node", conversation_node)
    workflow.add_node("disambiguation_node", disambiguation_node)
    workflow.add_node("route_question_node", route_question_node)
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges("router", _route_by_intent, {
        "routing_node": "routing_node",
        "search_node": "search_node",
        "conversation_node": "conversation_node",
        "disambiguation_node": "disambiguation_node",
        "route_question_node": "route_question_node",
        END: END
    })
    
    workflow.add_edge("routing_node", END)
    workflow.add_edge("search_node", END)
    workflow.add_edge("conversation_node", END)
    workflow.add_edge("disambiguation_node", END)
    workflow.add_edge("route_question_node", END)
    
    return workflow.compile(checkpointer=checkpointer)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def run_supervisor(user_message, session_id, checkpointer=None, location=None, user_id=None, knowledge_context=None):
    """Run the supervisor agent. State persisted via checkpointer + thread_id."""
    
    AgentLogger._print_box("🚀 NEW REQUEST", AgentLogger.Colors.HEADER)
    AgentLogger.info(f"Session: {session_id[:12]}...")
    AgentLogger.info(f"Message: \"{user_message}\"")
    if location:
        AgentLogger.info(f"GPS: ({location.get('lat')}, {location.get('lng')})")
    if knowledge_context:
        AgentLogger.info(f"🧠 Knowledge context: {len(knowledge_context)} chars injected")
    
    agent = create_supervisor_agent(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": session_id}}
    
    # If knowledge context exists, prepend it to the user message as system context
    enriched_message = user_message
    if knowledge_context:
        enriched_message = f"[SYSTEM CONTEXT — User Knowledge Base]\n{knowledge_context}\n[END SYSTEM CONTEXT]\n\nUser message: {user_message}"
    
    input_state = {
        "messages": [HumanMessage(content=enriched_message)],
        "location": location or {},
    }
    
    result = agent.invoke(input_state, config=config)
    
    AgentLogger._print_section("📋 REQUEST COMPLETE", AgentLogger.Colors.GREEN)
    AgentLogger.info(f"Intent: {result.get('current_intent')}")
    AgentLogger.info(f"Response: {len(result['messages'][-1].content)} chars")
    if result.get("route_data"):
        rd = result["route_data"]
        AgentLogger.info(f"Route: {rd.get('distance_km')} km, {rd.get('time_minutes')} min")
    if result.get("location_candidates"):
        AgentLogger.info(f"Candidates: {len(result['location_candidates'])}")
    if result.get("alternative_routes"):
        AgentLogger.info(f"Alternative routes: {len(result['alternative_routes'])}")
    AgentLogger.separator()
    
    return {
        "message": result["messages"][-1].content,
        "route_data": result.get("route_data"),
        "intent": result.get("current_intent"),
        "location_candidates": result.get("location_candidates"),
        "search_results": result.get("search_results"),
        "alternative_routes": result.get("alternative_routes")
    }
