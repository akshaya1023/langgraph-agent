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
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ---------------------------------------------------------------------------
# 1. Configure the Amazon Nova model via Bedrock Converse API
# ---------------------------------------------------------------------------
# Use a cross-region inference profile id if your account requires one, e.g.
# "us.amazon.nova-lite-v1:0". Override via env var for flexibility across
# environments (dev/stage/prod) without touching code.
NOVA_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

llm = init_chat_model(
    NOVA_MODEL_ID,
    model_provider="bedrock_converse",
    region_name=AWS_REGION,
    temperature=0.3,
)


# ---------------------------------------------------------------------------
# 2. Define LangGraph state + a single "hello world" node
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = (
    "You are a friendly hello-world assistant running on Amazon Bedrock "
    "AgentCore, backed by an Amazon Nova model. Keep answers short and clear."
)


def call_model(state: GraphState) -> GraphState:
    """Single LangGraph node: send the conversation so far to Nova."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


graph_builder = StateGraph(GraphState)
graph_builder.add_node("call_model", call_model)
graph_builder.add_edge(START, "call_model")
graph_builder.add_edge("call_model", END)

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
