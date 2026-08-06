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
import requests
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ---------------------------------------------------------------------------
# 1. Configure the Amazon Nova model via Bedrock Converse API
# ---------------------------------------------------------------------------
# Use a cross-region inference profile id if your account requires one, e.g.
# "us.amazon.nova-lite-v1:0". Override via env var for flexibility across
# environments (dev/stage/prod) without touching code.
NOVA_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

GATEWAY_URL = os.environ.get(
    "GATEWAY_URL",
    "https://hospital-agent-gateway-4wonadpgog.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
)

# Define tools first, before LLM initialization
@tool
def calculate_invoice(services: list, discount_percent: float = 0, insurance_covered: bool = False) -> str:
    """Calculate a hospital invoice based on services, discount, and insurance coverage.

    Args:
        services: List of hospital services (e.g., ["general consultation", "blood test"])
        discount_percent: Discount percentage (0-100)
        insurance_covered: Whether insurance covers the bill
    """
    try:
        payload = {
            "services": services,
            "discount_percent": discount_percent,
            "insurance_covered": insurance_covered
        }
        response = requests.post(
            f"{GATEWAY_URL}/calculate_invoice",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return json.dumps(response.json())
        return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling invoice service: {str(e)}"


@tool
def get_hr_info(department: str) -> str:
    """Get HR information about hospital staff and schedules.

    Args:
        department: Hospital department (e.g., "cardiology", "emergency")
    """
    try:
        payload = {"department": department}
        response = requests.post(
            f"{GATEWAY_URL}/get_hr_info",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return json.dumps(response.json())
        return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling HR service: {str(e)}"


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
