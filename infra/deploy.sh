#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Project A (RAG Pipeline) Full Deployment Script
#
# What this script does, in order:
#   1.  Validates prerequisites (AWS CLI, Docker, required env vars)
#   2.  Stores API keys in AWS Secrets Manager
#   3.  Deploys the shared base stack (VPC, Aurora, Redis, ECS cluster, ECR)
#   4.  Builds the Docker image and pushes it to ECR
#   5.  Deploys the service stack (ALB, ECS service, auto-scaling)
#   6.  Runs DB setup as a one-off ECS task (creates pgvector, tables, indexes)
#   7.  Runs corpus ingestion as a one-off ECS task
#   8.  Waits for the ECS service to be stable
#   9.  Prints the live URL
#
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   export COHERE_API_KEY="..."
#   export LANGFUSE_PUBLIC_KEY="pk-lf-..."
#   export LANGFUSE_SECRET_KEY="sk-lf-..."
#   bash infra/deploy.sh
#
# Teardown (delete everything):
#   bash infra/deploy.sh --teardown
#
# Redeploy only (skip base stack + DB setup, just push new image):
#   bash infra/deploy.sh --redeploy
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REGION="ap-south-1"
STACK_BASE="acmera-base"
STACK_SERVICE="acmera-service-a"
ECR_REPO_NAME="acmera-project-a"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

step()  { echo -e "\n${CYAN}${BOLD}[$(date +%H:%M:%S)]${RESET} ${BOLD}$*${RESET}"; }
ok()    { echo -e "  ${GREEN}✓${RESET} $*"; }
warn()  { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
die()   { echo -e "\n${RED}✗ ERROR:${RESET} $*" >&2; exit 1; }

# ── Mode flags ────────────────────────────────────────────────────────────────
TEARDOWN=false
REDEPLOY=false
for arg in "$@"; do
  case $arg in
    --teardown) TEARDOWN=true ;;
    --redeploy) REDEPLOY=true ;;
  esac
done

# ─────────────────────────────────────────────────────────────────────────────
# TEARDOWN MODE — delete all stacks and clean up
# ─────────────────────────────────────────────────────────────────────────────
if $TEARDOWN; then
  step "Tearing down Project A service stack..."
  aws cloudformation delete-stack --stack-name "${STACK_SERVICE}" --region "${REGION}" 2>/dev/null || warn "Service stack not found"
  aws cloudformation wait stack-delete-complete --stack-name "${STACK_SERVICE}" --region "${REGION}" 2>/dev/null && ok "Service stack deleted"

  # Only delete base stack if project-b service stack is also gone
  if ! aws cloudformation describe-stacks --stack-name "acmera-service-b" --region "${REGION}" &>/dev/null; then
    step "No project-b stack found — deleting base stack too..."
    aws cloudformation delete-stack --stack-name "${STACK_BASE}" --region "${REGION}" 2>/dev/null || warn "Base stack not found"
    aws cloudformation wait stack-delete-complete --stack-name "${STACK_BASE}" --region "${REGION}" 2>/dev/null && ok "Base stack deleted"
  else
    warn "Skipping base stack deletion — project-b service stack still exists."
    warn "Run project-b's deploy.sh --teardown first, then re-run this."
  fi

  echo -e "\n${GREEN}${BOLD}Teardown complete.${RESET}"
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────────────
step "Checking prerequisites..."

command -v aws    &>/dev/null || die "AWS CLI not found. Install: https://aws.amazon.com/cli/"
command -v docker &>/dev/null || die "Docker not found. Install: https://docs.docker.com/get-docker/"

if ! $REDEPLOY; then
  for var in OPENAI_API_KEY COHERE_API_KEY LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
    [[ -z "${!var:-}" ]] && die "Missing required environment variable: ${var}"
  done
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "${REGION}")
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"
ok "AWS account: ${AWS_ACCOUNT_ID}"
ok "ECR target:  ${ECR_URI}:${IMAGE_TAG}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — STORE SECRETS IN SECRETS MANAGER
# DB password MUST be stored before base stack deploys — Aurora references it
# via a CloudFormation dynamic reference: {{resolve:secretsmanager:/acmera/db-password}}
# ─────────────────────────────────────────────────────────────────────────────
if ! $REDEPLOY; then
  # SSM Parameter Store — SecureString type encrypts the value at rest.
  # --overwrite makes this idempotent: creates on first run, updates on re-runs.
  # ECS reads these by name at task startup (no ARN needed for SSM).
  step "Storing secrets in SSM Parameter Store..."

  store_param() {
    local name="/acmera/${1}"
    local value="${2}"
    aws ssm put-parameter \
      --name      "${name}" \
      --value     "${value}" \
      --type      SecureString \
      --overwrite \
      --region    "${REGION}" >/dev/null
    ok "Stored: ${name}"
  }

  # Generate a random DB password on first run; keep existing on re-runs
  # so Aurora doesn't trigger a password-change update unnecessarily.
  if ! aws ssm get-parameter --name "/acmera/db-password" --region "${REGION}" &>/dev/null; then
    DB_PASSWORD=$(openssl rand -base64 32 | tr -dc 'A-Za-z0-9' | head -c 32)
    store_param "db-password" "${DB_PASSWORD}"
  else
    ok "Kept existing: /acmera/db-password"
  fi

  store_param "openai-api-key"      "${OPENAI_API_KEY}"
  store_param "cohere-api-key"      "${COHERE_API_KEY}"
  store_param "langfuse-public-key" "${LANGFUSE_PUBLIC_KEY}"
  store_param "langfuse-secret-key" "${LANGFUSE_SECRET_KEY}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DEPLOY BASE STACK (idempotent — safe to run even if already deployed)
# ─────────────────────────────────────────────────────────────────────────────
if ! $REDEPLOY; then
  step "Deploying base infrastructure stack (${STACK_BASE})..."
  aws cloudformation deploy \
    --template-file "${SCRIPT_DIR}/base-stack.yaml" \
    --stack-name    "${STACK_BASE}" \
    --capabilities  CAPABILITY_IAM CAPABILITY_NAMED_IAM \
    --region        "${REGION}" \
    --no-fail-on-empty-changeset
  ok "Base stack deployed"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD AND PUSH DOCKER IMAGE
# ─────────────────────────────────────────────────────────────────────────────
step "Authenticating with ECR..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" >/dev/null
ok "ECR login successful"

step "Building Docker image..."
docker build \
  --platform linux/amd64 \
  -t "${ECR_REPO_NAME}:${IMAGE_TAG}" \
  "${PROJECT_ROOT}"
ok "Image built"

step "Pushing image to ECR..."
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"
ok "Image pushed: ${ECR_URI}:${IMAGE_TAG}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — DEPLOY SERVICE STACK
# ─────────────────────────────────────────────────────────────────────────────
step "Deploying service stack (${STACK_SERVICE})..."
aws cloudformation deploy \
  --template-file    "${SCRIPT_DIR}/service-a.yaml" \
  --stack-name       "${STACK_SERVICE}" \
  --parameter-overrides \
    "ImageUri=${ECR_URI}:${IMAGE_TAG}" \
  --capabilities     CAPABILITY_IAM \
  --region           "${REGION}" \
  --no-fail-on-empty-changeset
ok "Service stack deployed"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b — FORCE NEW ECS DEPLOYMENT
# CloudFormation won't redeploy ECS if the ImageUri tag didn't change (e.g.
# always using :latest). Force a new deployment so ECS pulls the fresh image.
# ─────────────────────────────────────────────────────────────────────────────
step "Forcing new ECS deployment (to pick up latest image)..."

ECS_CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_BASE}" \
  --query      'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
  --output     text --region "${REGION}")

ECS_SERVICE=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_SERVICE}" \
  --query      'Stacks[0].Outputs[?OutputKey==`ECSServiceName`].OutputValue' \
  --output     text --region "${REGION}")

aws ecs update-service \
  --cluster            "${ECS_CLUSTER}" \
  --service            "${ECS_SERVICE}" \
  --force-new-deployment \
  --region             "${REGION}" >/dev/null
ok "New deployment triggered"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a one-off ECS task with a command override and wait for it
# ─────────────────────────────────────────────────────────────────────────────
run_ecs_task() {
  local description="$1"
  local cmd_json="$2"   # JSON array: '["python","-m","scripts.setup_rds"]'

  step "Running ECS task: ${description}..."

  ECS_CLUSTER=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_BASE}" \
    --query      'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
    --output     text --region "${REGION}")

  TASK_DEF_ARN=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_SERVICE}" \
    --query      'Stacks[0].Outputs[?OutputKey==`TaskDefinitionArn`].OutputValue' \
    --output     text --region "${REGION}")

  PRIVATE_SUBNET=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_BASE}" \
    --query      'Stacks[0].Outputs[?OutputKey==`PrivateSubnet1Id`].OutputValue' \
    --output     text --region "${REGION}")

  ECS_SG=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_BASE}" \
    --query      'Stacks[0].Outputs[?OutputKey==`ECSSecurityGroupId`].OutputValue' \
    --output     text --region "${REGION}")

  TASK_ARN=$(aws ecs run-task \
    --cluster            "${ECS_CLUSTER}" \
    --task-definition    "${TASK_DEF_ARN}" \
    --launch-type        FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[${PRIVATE_SUBNET}],securityGroups=[${ECS_SG}],assignPublicIp=DISABLED}" \
    --overrides          "{\"containerOverrides\":[{\"name\":\"project-a\",\"command\":${cmd_json}}]}" \
    --query              'tasks[0].taskArn' \
    --output             text \
    --region             "${REGION}")

  echo "  Task ARN: ${TASK_ARN}"
  echo "  Waiting for task to complete..."

  aws ecs wait tasks-stopped \
    --cluster "${ECS_CLUSTER}" \
    --tasks   "${TASK_ARN}" \
    --region  "${REGION}"

  EXIT_CODE=$(aws ecs describe-tasks \
    --cluster "${ECS_CLUSTER}" \
    --tasks   "${TASK_ARN}" \
    --query   'tasks[0].containers[0].exitCode' \
    --output  text \
    --region  "${REGION}")

  if [[ "${EXIT_CODE}" == "0" ]]; then
    ok "${description} succeeded"
  else
    die "${description} failed with exit code ${EXIT_CODE}. Check CloudWatch logs: /ecs/acmera-project-a"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DB SETUP (only on first deploy, skip on --redeploy)
# ─────────────────────────────────────────────────────────────────────────────
if ! $REDEPLOY; then
  run_ecs_task "Database setup (pgvector extension + tables)" \
    '["python","-m","scripts.setup_rds"]'

  run_ecs_task "Corpus ingestion (embed + insert all .md files)" \
    '["python","-m","scripts.ingest_to_rds"]'
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — WAIT FOR SERVICE TO BE STABLE
# ─────────────────────────────────────────────────────────────────────────────
step "Waiting for ECS service to be stable..."

ECS_CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_BASE}" \
  --query      'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
  --output     text --region "${REGION}")

ECS_SERVICE=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_SERVICE}" \
  --query      'Stacks[0].Outputs[?OutputKey==`ECSServiceName`].OutputValue' \
  --output     text --region "${REGION}")

aws ecs wait services-stable \
  --cluster  "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" \
  --region   "${REGION}"
ok "Service is stable and healthy"

# ─────────────────────────────────────────────────────────────────────────────
# DONE — print the URL
# ─────────────────────────────────────────────────────────────────────────────
SERVICE_URL=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_SERVICE}" \
  --query      'Stacks[0].Outputs[?OutputKey==`ServiceURL`].OutputValue' \
  --output     text --region "${REGION}")

echo ""
echo -e "${GREEN}${BOLD}╔════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}${BOLD}║  Project A deployed successfully!       ║${RESET}"
echo -e "${GREEN}${BOLD}╠════════════════════════════════════════╣${RESET}"
echo -e "${GREEN}${BOLD}║${RESET}  URL: ${BOLD}${SERVICE_URL}${RESET}"
echo -e "${GREEN}${BOLD}╚════════════════════════════════════════╝${RESET}"
echo ""
echo "  CloudWatch logs: aws logs tail /ecs/acmera-project-a --follow --region ${REGION}"
echo "  To teardown:     bash infra/deploy.sh --teardown"
echo "  To redeploy:     bash infra/deploy.sh --redeploy"
