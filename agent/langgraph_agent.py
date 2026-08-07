"""
LangGraph agent that uses gateway tools via MCP SDK v2.0.0.
"""

import os
import json
import requests
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from bedrock_agentcore.runtime import BedrockAgentCoreApp
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

NOVA_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

GATEWAY_URL = os.environ.get(
    "GATEWAY_URL",
    "https://hospital-agent-gateway-4wonadpgog.gateway.bedrock-agentcore.us-east-1.amazonaws.com"
)

llm = init_chat_model(
    NOVA_MODEL_ID,
    model_provider="bedrock_converse",
    region_name=AWS_REGION,
    temperature=0.3,
)


def call_gateway_tool(tool_name: str, tool_input: dict) -> str:
    """Call a tool through the Bedrock AgentCore Gateway via JSON-RPC HTTP POST."""
    try:
        # Create AWS credentials for signing the request
        session = boto3.Session()
        credentials = session.get_credentials()

        # Gateway endpoint
        url = f"{GATEWAY_URL}/mcp"

        # JSON-RPC payload
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_input
            },
            "id": 1
        }

        # Sign the request with AWS SigV4
        request = AWSRequest(method='POST', url=url, data=json.dumps(payload))
        SigV4Auth(credentials, 'bedrock-agentcore', AWS_REGION).add_auth(request)

        # Make the HTTP POST call
        response = requests.post(url, data=json.dumps(payload), headers=dict(request.headers))

        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                return json.dumps({"success": False, "error": result["error"].get("message", "Unknown error")})
            elif "result" in result:
                return json.dumps({"success": True, "result": result["result"]})
            else:
                return json.dumps({"success": False, "error": "Unexpected response format"})
        else:
            return json.dumps({"success": False, "error": f"HTTP {response.status_code}: {response.text}"})

    except Exception as e:
        import traceback
        return json.dumps({"success": False, "error": f"{type(e).__name__}: {str(e)}", "traceback": traceback.format_exc()})


@tool
def calculate_invoice(services: list, discount_percent: float = 0, insurance_covered: bool = False) -> str:
    """Calculate a hospital invoice based on services, discount, and insurance coverage."""
    try:
        tool_input = {
            "services": services,
            "discount_percent": discount_percent,
            "insurance_covered": insurance_covered
        }
        result = call_gateway_tool("target-invoiceprocessor___calculate_invoice", tool_input)
        result_obj = json.loads(result)
        if result_obj.get("success"):
            return str(result_obj.get("result", "No result"))
        else:
            return f"ERROR: {result_obj.get('error', 'Unknown error')}"
    except Exception as e:
        import traceback
        return f"EXCEPTION: {type(e).__name__}: {str(e)}"


@tool
def get_hr_info(employee_id: str, info_type: str = "schedule") -> str:
    """Get HR information for an employee including schedule, benefits, and HR details."""
    try:
        tool_input = {
            "employeeId": employee_id,
            "infoType": info_type
        }
        result = call_gateway_tool("target-hrassitant___get_hr_info", tool_input)
        result_obj = json.loads(result)
        if result_obj.get("success"):
            return str(result_obj.get("result", "No result"))
        else:
            return f"ERROR: {result_obj.get('error', 'Unknown error')}"
    except Exception as e:
        import traceback
        return f"EXCEPTION: {type(e).__name__}: {str(e)}"


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
