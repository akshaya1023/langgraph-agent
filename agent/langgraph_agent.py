"""
LangGraph agent that uses gateway tools via MCP SDK v2.0.0.
"""

import os
import json
import asyncio
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp import ClientSession
from mcp.client.sse import sse_client
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

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


async def call_gateway_tool_async(tool_name: str, tool_input: dict) -> str:
    """Call a tool through the gateway MCP server via SSE with AWS authentication."""
    try:
        # Create AWS credentials for signing the request
        session = boto3.Session()
        credentials = session.get_credentials()

        # Sign the gateway URL for authentication
        request = AWSRequest(method='GET', url=GATEWAY_URL)
        SigV4Auth(credentials, 'bedrock-agentcore', AWS_REGION).add_auth(request)

        # Extract signed headers
        headers = dict(request.headers)

        async with sse_client(GATEWAY_URL, headers=headers) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name=tool_name, arguments=tool_input)

                if result.content:
                    response_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                    return json.dumps({"success": True, "result": response_text})
                else:
                    return json.dumps({"success": False, "error": "No response from tool"})
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return json.dumps({"success": False, "error": f"{type(e).__name__}: {str(e)}", "details": error_details})


def call_gateway_tool_sync(tool_name: str, tool_input: dict) -> str:
    """Synchronous wrapper for calling gateway tools."""
    try:
        # Check if there's already a running event loop
        try:
            loop = asyncio.get_running_loop()
            is_running = True
        except RuntimeError:
            is_running = False

        if is_running:
            # We're in async context, use thread pool to run the async function
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, call_gateway_tool_async(tool_name, tool_input))
                return future.result()
        else:
            # No running loop, we can create one
            return asyncio.run(call_gateway_tool_async(tool_name, tool_input))
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
        result = call_gateway_tool_sync("calculate_invoice", tool_input)
        result_obj = json.loads(result)
        if result_obj.get("success"):
            return result_obj.get("result", "No result")
        else:
            return f"ERROR: {result_obj.get('error', 'Unknown error')}"
    except Exception as e:
        import traceback
        return f"EXCEPTION: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


@tool
def get_hr_info(employee_id: str, info_type: str = "schedule") -> str:
    """Get HR information for an employee including schedule, benefits, and HR details."""
    try:
        tool_input = {
            "employeeId": employee_id,
            "infoType": info_type
        }
        result = call_gateway_tool_sync("get_hr_info", tool_input)
        result_obj = json.loads(result)
        if result_obj.get("success"):
            return result_obj.get("result", "No result")
        else:
            return f"ERROR: {result_obj.get('error', 'Unknown error')}"
    except Exception as e:
        import traceback
        return f"EXCEPTION: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


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
