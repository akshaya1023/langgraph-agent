"""
Smoke test: invokes the deployed AgentCore Runtime once and prints the result.

Required env vars:
  AWS_REGION
  AGENT_RUNTIME_ARN
"""
import json
import os

import boto3

REGION = os.environ["AWS_REGION"]
RUNTIME_ARN = os.environ["AGENT_RUNTIME_ARN"]

client = boto3.client("bedrock-agentcore", region_name=REGION)

response = client.invoke_agent_runtime(
    agentRuntimeArn=RUNTIME_ARN,
    payload=json.dumps({"prompt": "Say hello and tell me what model powers you."}).encode("utf-8"),
)

body = response["response"].read().decode("utf-8")
print("Agent response:")
print(body)
