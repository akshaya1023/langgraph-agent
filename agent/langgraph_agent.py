"""
LangGraph agent that properly uses gateway via MCP protocol.
"""

import os
import json
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp.client.session import ClientSession
from mcp.client.sse import SSEClientSession

NOVA_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

GATEWAY_URL = os.environ.get(
    "GATEWAY_URL",
    "https://hospital-agent-gateway-4wonadpgog.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
)

llm = init_chat_model(
    NOVA_MODEL_ID,
    model_provider="bedrock_converse",
    region_name=AWS_REGION,
    temperature=0.3,
)


def _call_gateway_tool(tool_name: str, tool_input: dict) -> str:
    """Call a tool through the gateway MCP server synchronously."""
    import asyncio

    async def call_async():
        try:
            async with SSEClientSession(GATEWAY_URL) as session:
                result = await session.call_tool(tool_name, tool_input)
                if result.content:
                    return json.dumps({"success": True, "result": result.content[0].text})
                return json.dumps({"success": False, "error": "No response from tool"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(call_async())
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def calculate_invoice(services: list, discount_percent: float = 0, insurance_covered: bool = False) -> str:
    """Calculate a hospital invoice based on services, discount, and insurance coverage."""
    tool_input = {
        "services": services,
        "discount_percent": discount_percent,
        "insurance_covered": insurance_covered
    }
    return _call_gateway_tool("calculate_invoice", tool_input)


@tool
def get_hr_info(department: str) -> str:
    """Get HR information about hospital staff and schedules."""
    tool_input = {"department": department}
    return _call_gateway_tool("get_hr_info", tool_input)


tools = [calculate_invoice, get_hr_info]
llm_with_tools = llm.bind_tools(tools)


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = (
    "You are a helpful hospital assistant. You have access to tools to calculate invoices "
    "and get HR information. Use the available tools to answer questions about hospital services, "
    "pricing, and staff. Keep answers clear and concise."
)


def call_model(state: GraphState) -> GraphState:
    """Call the LLM with tool support."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_use_tools(state: GraphState) -> str:
    """Route to tools if needed."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


graph_builder = StateGraph(GraphState)
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "call_model")
graph_builder.add_conditional_edges(
    "call_model",
    should_use_tools,
    {"tools": "tools", "end": END}
)
graph_builder.add_edge("tools", "call_model")

graph = graph_builder.compile()


app = BedrockAgentCoreApp()


@app.entrypoint
def agent_invocation(payload: dict, context=None) -> dict:
    """Agent entrypoint."""
    user_prompt = payload.get("prompt", "Hello!")
    result = graph.invoke({"messages": [HumanMessage(content=user_prompt)]})
    answer = result["messages"][-1].content
    return {"result": answer}


if __name__ == "__main__":
    app.run()
