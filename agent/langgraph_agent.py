import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from tools import (
    calculate_treatment_invoice,
    get_staff_schedule,
)

# ------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------

NOVA_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "amazon.nova-lite-v1:0"
)

AWS_REGION = os.environ.get(
    "AWS_REGION",
    "us-east-1"
)

llm = init_chat_model(
    NOVA_MODEL_ID,
    model_provider="bedrock_converse",
    region_name=AWS_REGION,
    temperature=0.3,
)

# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------

tools = [
    calculate_treatment_invoice,
    get_staff_schedule,
]

# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

agent = create_react_agent(
    llm,
    tools
)

# ------------------------------------------------------------------
# AgentCore Runtime
# ------------------------------------------------------------------

app = BedrockAgentCoreApp()


@app.entrypoint
def agent_invocation(payload: dict, context=None):

    user_prompt = payload.get("prompt", "Hello!")

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content=user_prompt)
            ]
        }
    )

    answer = response["messages"][-1].content

    return {
        "result": answer
    }


if __name__ == "__main__":
    app.run()