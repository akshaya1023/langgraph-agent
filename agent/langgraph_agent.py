"""
LangGraph agent that uses gateway tools via HTTP requests.
"""

import os
import json
import httpx
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from bedrock_agentcore.runtime import BedrockAgentCoreApp

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


@tool
def calculate_invoice(services: list, discount_percent: float = 0, insurance_covered: bool = False) -> str:
    """Calculate a hospital invoice based on services, discount, and insurance coverage."""
    try:
        payload = {
            "services": services,
            "discount_percent": discount_percent,
            "insurance_covered": insurance_covered
        }
        response = httpx.post(
            f"{GATEWAY_URL}/calculate_invoice",
            json=payload,
            timeout=10.0
        )
        if response.status_code == 200:
            return json.dumps(response.json())
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_hr_info(department: str) -> str:
    """Get HR information about hospital staff and schedules."""
    try:
        payload = {"department": department}
        response = httpx.post(
            f"{GATEWAY_URL}/get_hr_info",
            json=payload,
            timeout=10.0
        )
        if response.status_code == 200:
            return json.dumps(response.json())
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


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
