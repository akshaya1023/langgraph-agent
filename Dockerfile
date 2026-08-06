# AgentCore Runtime REQUIRES an ARM64 (AWS Graviton) image.
# --platform is also enforced at build time in the GitHub Actions workflow
# via `docker buildx build --platform linux/arm64`.
FROM --platform=linux/arm64 python:3.12-slim

# Non-root user (best practice for AgentCore Runtime containers)
RUN useradd -m -u 1000 agentuser

WORKDIR /app

# Install dependencies first for better layer caching
COPY agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent source
COPY agent/langgraph_agent.py ./langgraph_agent.py
COPY agent/tools.py ./tools.py

# AgentCore passes model/region config as env vars at deploy time; these are
# just safe defaults for local `docker run` testing.
ENV AWS_REGION=us-east-1
ENV BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
ENV PYTHONUNBUFFERED=1

USER agentuser

# AgentCore Runtime service contract: host 0.0.0.0, port 8080,
# GET /ping and POST /invocations (both provided by BedrockAgentCoreApp).
EXPOSE 8080

CMD ["python", "langgraph_agent.py"]
