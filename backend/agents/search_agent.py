"""
Search Agent — POI discovery with tool-calling.

Uses Gemini 3 Flash (via Kie API) to understand search queries and invoke
the appropriate OSM Overpass tools. Falls back to direct tool results
if the LLM summary call fails.
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from backend.models.state import SearchAgentState
from backend.tools.osm_search_tool import search_poi_along_route, search_poi_nearby
from backend.config import KIE_API_KEY, KIE_BASE_URL
from backend.utils.logger import AgentLogger
import json


def create_search_agent():
    """Create and return the search agent graph."""
    
    llm = ChatOpenAI(
        model="gemini-3-flash",
        api_key=KIE_API_KEY,
        base_url=f"{KIE_BASE_URL}/gemini-3-flash/v1",
        temperature=0,
    )
    
    tools = [search_poi_along_route, search_poi_nearby]
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: SearchAgentState):
        """Main agent reasoning node."""
        AgentLogger.agent_thinking()
        messages = state["messages"]
        
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            AgentLogger.error(f"LLM call failed in search agent: {str(e)}")
            
            # Check if we already have tool results we can summarize
            tool_results = [m for m in messages if isinstance(m, ToolMessage)]
            if tool_results:
                # We have tool results but LLM failed to summarize — format them ourselves
                summary = _format_tool_results(tool_results)
                return {"messages": [AIMessage(content=summary)]}
            
            # No tool results yet and LLM failed — return error message
            return {"messages": [AIMessage(content="I encountered an error while searching. Please try again.")]}
    
    def should_continue(state: SearchAgentState):
        """Determine if we should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    # Build graph
    workflow = StateGraph(SearchAgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


def _format_tool_results(tool_messages) -> str:
    """Format raw tool results into a conversational response when LLM summarization fails."""
    all_pois = []

    for msg in tool_messages:
        try:
            content = msg.content
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content

            if isinstance(data, dict) and "pois" in data:
                all_pois.extend(data["pois"])
            elif isinstance(data, list):
                all_pois.extend(data)
        except (json.JSONDecodeError, TypeError):
            pass

    if not all_pois:
        return "I searched but couldn't find any results nearby. Try a different search or location."

    parts = [f"I found {len(all_pois)} places nearby."]
    for i, poi in enumerate(all_pois[:5], 1):
        name = poi.get("name", "Unnamed")
        dist = poi.get("distance_km", "")
        dist_str = f", about {dist} km away" if dist else ""
        address = poi.get("address", "")
        addr_str = f" on {address}" if address else ""
        parts.append(f"{i} is {name}{addr_str}{dist_str}.")

    if len(all_pois) > 5:
        parts.append(f"There are {len(all_pois) - 5} more as well.")

    return " ".join(parts)


def run_search_agent(query: str, route_data: dict = None, location: dict = None):
    """
    Run the search agent to find POIs.
    
    Args:
        query: User's search query
        route_data: Optional route data with polyline
        location: Optional current location {"lat": x, "lng": y}
    
    Returns:
        Search results with POI information
    """
    
    AgentLogger.search_start(query)
    
    agent = create_search_agent()
    
    system_prompt = """You are a search assistant that finds Points of Interest (POIs) for a navigation app.

## YOUR TOOLS:
1. **search_poi_nearby** — Search for POIs near a GPS coordinate. Use this when:
   - User wants to find places near their current location
   - No active route exists
   - User says "near me", "nearby", "closest", "nearest", "pass mein"

2. **search_poi_along_route** — Search for POIs along an active navigation route. Use this when:
   - There IS an active route AND user wants POIs along it (e.g., "gas stations on the way")
   - User says "on the way", "along the route", "en route", "raste mein"

## POI TYPE MAPPING (user query → poi_type parameter):
- hospital, clinic, medical → "hospital"
- petrol pump, gas station, fuel → "fuel"  
- EV charger, charging → "charging_station"
- restaurant, food, khana → "restaurant"
- cafe, coffee → "cafe"
- ATM, cash → "atm"
- parking → "parking"
- hotel, lodge, stay → "hotel"
- pharmacy, medical store, dawai → "pharmacy"
- supermarket, grocery → "supermarket"

## RULES:
- ALWAYS use the user's current location coordinates when available.
- Default radius: 5000 meters. Increase to 10000 if few results.
- After getting results, summarize them clearly with names and distances, sorted by distance (nearest first).
- If no results found, suggest increasing radius or trying a different POI type.

## RESPONSE FORMAT (critical):
- Write your response as natural conversational text, like a co-pilot talking to the driver.
- NEVER use emojis or markdown formatting (no **bold**, no bullet points, no numbered lists).
- Mention the closest result first with its distance.
- Keep it brief and natural. Example: 'I found 3 hospitals nearby. The closest is City Hospital, about 1.2 km away.'"""
    
    context = ""
    if route_data:
        context += f"\nRoute available with {len(route_data.get('polyline', []))} points."
    if location:
        context += f"\nCurrent location: {location['lat']}, {location['lng']}"
    
    initial_state = {
        "messages": [
            SystemMessage(content=system_prompt + context),
            HumanMessage(content=query)
        ],
        "route_data": route_data or {},
        "location": location or {},
        "search_results": []
    }
    
    result = agent.invoke(initial_state)
    
    final_message = result["messages"][-1].content
    AgentLogger.agent_response(final_message)
    AgentLogger.separator()
    
    return result
