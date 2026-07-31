# LangGraph "Hello World" Agent on Amazon Bedrock AgentCore Runtime

A minimal LangGraph agent, backed by an **Amazon Nova** model via Bedrock, packaged
as an ARM64 container and deployed to **Amazon Bedrock AgentCore Runtime** using
native **CloudFormation** (`AWS::BedrockAgentCore::Runtime`) — no custom Python
deploy script required.

```
agentcore-langgraph-nova/
├── agent/
│   ├── langgraph_agent.py            # LangGraph graph + BedrockAgentCoreApp entrypoint
│   └── requirements.txt
├── Dockerfile                        # ARM64 image matching AgentCore's container contract
├── cloudformation/
│   ├── bootstrap.yaml                # ONE-TIME: ECR repo + AgentCore execution role
│   └── agentcore-runtime.yaml        # EVERY RUN: creates/updates the AgentCore Runtime
├── scripts/
│   └── test_agent.py                 # optional manual smoke test (CLI does this in the pipeline)
├── iam/
│   ├── github-oidc-trust-policy.json
│   └── github-oidc-permissions-policy.json
└── .github/workflows/
    └── deploy-agentcore.yml          # build -> push to ECR -> deploy CFN stack
```

## Why the order is "image first, then runtime"

`AWS::BedrockAgentCore::Runtime`'s `ContainerUri` property must point at an
image that **already exists** in ECR — CloudFormation can't create the Runtime
against an image that isn't there yet. So the actual sequence is:

1. **Once, manually**: deploy `cloudformation/bootstrap.yaml` → creates the ECR
   repository and the IAM execution role AgentCore assumes at run time.
2. **Every pipeline run**: build the Docker image → push it to that ECR repo →
   `aws cloudformation deploy` on `cloudformation/agentcore-runtime.yaml`.

That last step is what actually creates/updates the Runtime, and it's
idempotent:
- **First run** → stack doesn't exist → CloudFormation **creates** the
  `AWS::BedrockAgentCore::Runtime` resource pointing at that first image.
- **Every run after** → stack exists → CloudFormation **updates** it in place
  (`ContainerUri` is a no-interruption update per AWS's docs) to point at the
  new image tag.

No `deploy_agentcore.py`, no boto3 create-or-update branching logic — one
`aws cloudformation deploy` command handles both cases.

## One-time setup

### 1. Enable model access
In the Bedrock console, enable access to the Amazon Nova models (e.g.
`amazon.nova-lite-v1:0`) in your target region.

### 2. Create the GitHub OIDC provider + deploy role
```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1

aws iam create-role \
  --role-name github-actions-agentcore-deploy-role \
  --assume-role-policy-document file://iam/github-oidc-trust-policy.json

aws iam put-role-policy \
  --role-name github-actions-agentcore-deploy-role \
  --policy-name agentcore-deploy-permissions \
  --policy-document file://iam/github-oidc-permissions-policy.json
```
Edit the placeholders in `iam/github-oidc-trust-policy.json`
(`<YOUR_ACCOUNT_ID>`, `<YOUR_GH_ORG>/<YOUR_GH_REPO>`) before applying.

### 3. Add the GitHub repository secret
| Secret | Value |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | ARN of `github-actions-agentcore-deploy-role` |

### 4. Deploy the bootstrap stack once (can also be a one-off manual step, or the first workflow run does it automatically — it's in the pipeline too)
```bash
aws cloudformation deploy \
  --template-file cloudformation/bootstrap.yaml \
  --stack-name langgraph-nova-agent-bootstrap \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides RepositoryName=langgraph-nova-agent
```

### 5. Push
```bash
git add .
git commit -m "Deploy LangGraph Nova hello-world agent"
git push origin main
```
Watch the **Actions** tab. The pipeline:
1. Deploys/confirms the bootstrap stack (ECR repo + execution role).
2. Builds the ARM64 image and pushes it to ECR.
3. Waits for the Amazon Inspector scan.
4. Runs `aws cloudformation deploy` on `agentcore-runtime.yaml` — creating the
   Runtime on the first run, updating it on every run after.
5. Reads the deployed `AgentRuntimeArn` from stack outputs and invokes it once
   as a smoke test.

You can also invoke it yourself anytime:
```bash
aws bedrock-agentcore invoke-agent-runtime \
  --agent-runtime-arn <ARN-from-stack-outputs> \
  --payload '{"prompt": "Hello!"}' \
  --region us-east-1 \
  output.json
```

## Local test before pushing
```bash
cd agent
pip install -r requirements.txt
python langgraph_agent.py
# in another terminal
curl -X POST http://localhost:8080/invocations \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, who are you?"}'
```

## Notes / things to double-check against your account
- `AWS::BedrockAgentCore::Runtime` is a **recently added** native CloudFormation
  resource type. Property names (`AgentRuntimeArtifact`, `NetworkConfiguration`,
  `ProtocolConfiguration`, etc.) match AWS's current CloudFormation Template
  Reference as of this writing — re-check the reference page before relying on
  it in production, since new services evolve quickly.
- AgentCore Runtime **only supports ARM64 images** today — the workflow builds
  with `--platform linux/arm64` via Buildx/QEMU emulation on the standard
  `ubuntu-latest` runner.
- Swap `amazon.nova-lite-v1:0` for `amazon.nova-pro-v1:0` /
  `amazon.nova-micro-v1:0` (or a `us.amazon.nova-*` cross-region inference
  profile id) via the `BedrockModelId` CloudFormation parameter / `BEDROCK_MODEL_ID`
  workflow env var — no code changes needed.
- To run the Runtime inside a VPC instead of `PUBLIC` mode, extend
  `cloudformation/agentcore-runtime.yaml` with a `VpcConfig` (`Subnets` +
  `SecurityGroups`) under `NetworkConfiguration.NetworkModeConfig` and pass
  `NetworkMode=VPC`.
- The Inspector "fail on critical/high vulnerabilities" gate is not enforced
  by default in the workflow; add a hard check once you have a clean baseline
  image.
