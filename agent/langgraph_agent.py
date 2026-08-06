"""
LangGraph "Hello World" agent powered by Amazon Bedrock Nova,
packaged to run on Amazon Bedrock AgentCore Runtime.

Flow:
  request -> BedrockAgentCoreApp.entrypoint -> LangGraph graph.invoke() -> Nova model -> response

Local test:
  python langgraph_agent.py
  curl -X POST http://localhost:8080/invocations \
       -H "Content-Type: application/json" \
       -d '{"prompt": "Hello, who are you?"}'
"""

import os
import json
from typing import Annotated, TypedDict
import asyncio
import httpx

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mcp.client.stdio import StdioClientSession
from mcp.client.sse import SSEClientSession

from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ---------------------------------------------------------------------------
# 1. Configure the Amazon Nova model via Bedrock Converse API
# ---------------------------------------------------------------------------
NOVA_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

GATEWAY_URL = os.environ.get(
    "GATEWAY_URL",
    "https://hospital-agent-gateway-4wonadpgog.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
)

# Initialize MCP client for gateway
async def get_gateway_tools():
    """Connect to gateway MCP server and get available tools"""
    async with SSEClientSession(GATEWAY_URL) as session:
        tools_list = await session.list_tools()
        return tools_list.tools


# Create tool wrappers for gateway tools
@tool
def calculate_invoice(services: list, discount_percent: float = 0, insurance_covered: bool = False) -> str:
    """Calculate a hospital invoice based on services, discount, and insurance coverage."""
    try:
        payload = {
            "services": services,
            "discount_percent": discount_percent,
            "insurance_covered": insurance_covered
        }
        async def invoke():
            async with SSEClientSession(GATEWAY_URL) as session:
                result = await session.call_tool("calculate_invoice", payload)
                return result.content[0].text if result.content else ""

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(invoke())
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_hr_info(department: str) -> str:
    """Get HR information about hospital staff and schedules."""
    try:
        payload = {"department": department}
        async def invoke():
            async with SSEClientSession(GATEWAY_URL) as session:
                result = await session.call_tool("get_hr_info", payload)
                return result.content[0].text if result.content else ""

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(invoke())
    except Exception as e:
        return f"Error: {str(e)}"


tools = [calculate_invoice, get_hr_info]

llm = init_chat_model(
    NOVA_MODEL_ID,
    model_provider="bedrock_converse",
    region_name=AWS_REGION,
    temperature=0.3,
)

llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 2. Define LangGraph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = (
    "You are a helpful hospital assistant running on Amazon Bedrock AgentCore. "
    "You have access to tools to calculate invoices and get HR information. "
    "Use the available tools to answer questions about hospital services, pricing, and staff. "
    "Keep answers clear and concise."
)


def call_model(state: GraphState) -> GraphState:
    """LangGraph node: send the conversation to Nova with tool support."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_use_tools(state: GraphState) -> str:
    """Route to tools if the model wants to call them."""
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


# ---------------------------------------------------------------------------
# 3. Wire the graph up to Bedrock AgentCore Runtime
# ---------------------------------------------------------------------------
# BedrockAgentCoreApp provides the /invocations and /ping endpoints AgentCore
# Runtime expects, listening on 0.0.0.0:8080 by default.
app = BedrockAgentCoreApp()


@app.entrypoint
def agent_invocation(payload: dict, context=None) -> dict:
    """
    Entrypoint called by AgentCore Runtime for every InvokeAgentRuntime request.

    Expected payload: {"prompt": "<user text>"}
    Returns: {"result": "<agent text response>"}
    """
    user_prompt = payload.get("prompt", "Hello!")

    result = graph.invoke({"messages": [HumanMessage(content=user_prompt)]})
    answer = result["messages"][-1].content

    return {"result": answer}


if __name__ == "__main__":
    # Runs the AgentCore-compatible HTTP server locally on 0.0.0.0:8080
    # for testing before pushing to AgentCore Runtime.
    app.run()
