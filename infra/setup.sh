#!/usr/bin/env bash
#
# One-time AWS setup for context-mcp. Creates everything .github/workflows/
# deploy.yml assumes already exists; after this runs, pushes to main deploy
# on their own.
#
#   ./infra/setup.sh            # dry run — prints every action, changes nothing
#   ./infra/setup.sh --apply    # actually creates resources
#
# Idempotent: every step checks for an existing resource first, so re-running
# after a partial failure repairs rather than duplicates or errors out.
#
# What gets created, in your account:
#   - Secret   context-mcp/credentials    (one JSON blob: Voyage + Chroma + token)
#   - IAM role context-mcp-lambda-role    (Lambda execution, logs, read that secret)
#   - Lambda   context-mcp                (python3.13, arm64, 512MB, SnapStart)
#   - Alias    live                       (what clients point at)
#   - Function URL on the alias           (public hostname, see SECURITY below)
#   - IAM OIDC provider for GitHub Actions (only if not already present)
#   - IAM role context-mcp-deploy-role    (assumed by CI to deploy)
#
# WHY SECRETS MANAGER RATHER THAN FUNCTION ENVIRONMENT VARIABLES:
# Reading a Lambda's configuration returns its environment variables in
# plaintext, and the CI deploy role must be able to read configuration to poll
# SnapStart status after publishing. Credentials in the environment would
# therefore be readable by anything that can run the deploy workflow. Kept in
# Secrets Manager, they are fetched at runtime by the function's own execution
# role, and CI can read all the configuration it likes without learning
# anything. One secret holding a JSON object, not five secrets — Secrets
# Manager bills per secret per month.
#
# SECURITY — read before running:
# The Function URL is created with AuthType=NONE. That is deliberate: AWS_IAM
# auth requires SigV4-signed requests, which MCP clients do not produce, so it
# would make the endpoint unusable. The URL is public and unguessable, not
# private. What actually protects it:
#
#   /mcp       OAuth via Cognito (both claude.ai and Claude Code get their own
#              app client), plus — while MCP_ALLOW_STATIC_TOKEN is on — the
#              static AUTH_TOKEN as a fallback.
#   /map*      AUTH_TOKEN only. These routes bypass MCP-level auth by design
#              and serve the entire store to a browser, so they need their own
#              guard; there is no OAuth flow for a static page.
#
# If AUTH_TOKEN is ever empty the map guard becomes a no-op and those pages are
# world-readable. Once Claude Code's OAuth is confirmed working, set
# MCP_ALLOW_STATIC_TOKEN=0 so the shared password no longer grants access to
# the store itself.
#
# There is deliberately NO IP allowlist. Anthropic publishes outbound ranges
# (160.79.104.0/21) which would cover claude.ai, but Claude Code runs on your
# own machine and connects from your own address — so an Anthropic-only
# allowlist would block half the clients this exists for, and adding your own
# address breaks whenever your ISP changes it. The bearer token is the control.
#
# TEARDOWN (not automated on purpose — deleting the function is destructive):
#   aws lambda delete-function-url-config --function-name context-mcp --qualifier live
#   aws lambda delete-function --function-name context-mcp
#   aws secretsmanager delete-secret --secret-id context-mcp/credentials \
#     --recovery-window-in-days 7
#   aws iam delete-role-policy --role-name context-mcp-deploy-role --policy-name deploy
#   aws iam delete-role --role-name context-mcp-deploy-role
#   aws iam delete-role-policy --role-name context-mcp-lambda-role --policy-name read-secret
#   aws iam detach-role-policy --role-name context-mcp-lambda-role \
#     --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
#   aws iam delete-role --role-name context-mcp-lambda-role

set -euo pipefail

# --- Config -----------------------------------------------------------------
FUNCTION_NAME="context-mcp"
ALIAS="live"
REGION="${AWS_REGION:-eu-west-1}"
RUNTIME="python3.13"
ARCH="arm64"
MEMORY_MB=512          # measured: more memory does not meaningfully cut init time
TIMEOUT_S=30
HANDLER="mcp_server.server.handler"
SECRET_NAME="context-mcp/credentials"
LAMBDA_ROLE="context-mcp-lambda-role"
DEPLOY_ROLE="context-mcp-deploy-role"
GITHUB_REPO="Anyth3nG/claude-context-mcp"

# Cognito — the OAuth authorization server claude.ai authenticates against.
COGNITO_POOL_NAME="context-mcp"
COGNITO_CLIENT_NAME="claude-ai-connector"
COGNITO_CODE_CLIENT_NAME="claude-code-cli"
# Claude Code calls back on a loopback port. Cognito matches redirect URIs
# exactly, including the port, so the port has to be pinned on both sides —
# hence --callback-port when adding the server.
CLAUDE_CODE_CALLBACK_PORT="8765"
COGNITO_RESOURCE_ID="context-mcp"
COGNITO_SCOPE="access"
# Fixed callback claude.ai redirects to after login. If Anthropic ever changes
# it, the connector fails with a redirect_mismatch error from Cognito.
CLAUDE_AI_CALLBACK="https://claude.ai/api/mcp/auth_callback"
# Login identity for the Cognito pool. Read from .env, never hardcoded — this
# repository is PUBLIC, and a committed email address is harvested by scrapers.
# Set COGNITO_USER_EMAIL in .env (gitignored).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/.build"
ZIP_PATH="$REPO_ROOT/.build.zip"

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
run()  {
  if [ "$APPLY" = 1 ]; then "$@"
  else info "[dry-run] $*"
  fi
}

# --- Preflight --------------------------------------------------------------
say "Preflight"
command -v aws     >/dev/null || { echo "aws CLI not found"; exit 1; }
command -v zip     >/dev/null || { echo "zip not found"; exit 1; }
command -v pip     >/dev/null || { echo "pip not found"; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
info "account: $ACCOUNT_ID   region: $REGION"

# Hosted-UI domain prefix must be globally unique across every AWS account.
# Derived from the account id rather than randomised so re-runs stay idempotent.
COGNITO_DOMAIN="context-mcp-${ACCOUNT_ID}"

# Credentials are read from .env (gitignored) and handed to AWS over the API,
# so they are never typed on a command line that lands in shell history.
[ -f "$REPO_ROOT/.env" ] || { echo ".env not found at $REPO_ROOT/.env"; exit 1; }
set -a; . "$REPO_ROOT/.env"; set +a

for v in VOYAGE_API_KEY CHROMA_TENANT CHROMA_DATABASE CHROMA_API_KEY COGNITO_USER_EMAIL; do
  [ -n "${!v:-}" ] || { echo "Missing $v in .env"; exit 1; }
done
info "voyage + chroma credentials present"
info "cognito login identity: $COGNITO_USER_EMAIL"

# AUTH_TOKEN is the only thing standing between the Function URL and the open
# internet. Generate one if absent rather than letting the deploy proceed
# unprotected.
if [ -z "${AUTH_TOKEN:-}" ]; then
  AUTH_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  say "Generated AUTH_TOKEN (appending to .env — it is gitignored)"
  info "AUTH_TOKEN=$AUTH_TOKEN"
  if [ "$APPLY" = 1 ]; then
    printf '\n# Bearer token for the deployed Function URL (generated by infra/setup.sh)\nAUTH_TOKEN=%s\n' \
      "$AUTH_TOKEN" >> "$REPO_ROOT/.env"
  else
    info "[dry-run] would append AUTH_TOKEN to .env"
  fi
else
  info "AUTH_TOKEN already set in .env"
fi

# --- Secret -----------------------------------------------------------------
say "Secret: $SECRET_NAME"
# Built with python3 rather than string interpolation so any character in a
# key (quotes, backslashes) is escaped correctly instead of producing invalid
# JSON that only fails at runtime.
SECRET_JSON="$(VOYAGE_API_KEY="$VOYAGE_API_KEY" CHROMA_TENANT="$CHROMA_TENANT" \
  CHROMA_DATABASE="$CHROMA_DATABASE" CHROMA_API_KEY="$CHROMA_API_KEY" \
  AUTH_TOKEN="$AUTH_TOKEN" python3 -c '
import json, os
print(json.dumps({k: os.environ[k] for k in
  ("VOYAGE_API_KEY","CHROMA_TENANT","CHROMA_DATABASE","CHROMA_API_KEY","AUTH_TOKEN")}))')"

if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
  info "already exists — updating its value to match .env"
  if [ "$APPLY" = 1 ]; then
    aws secretsmanager put-secret-value --secret-id "$SECRET_NAME" \
      --secret-string "$SECRET_JSON" --no-cli-pager >/dev/null
  else
    info "[dry-run] would put-secret-value (5 keys, values not shown)"
  fi
else
  if [ "$APPLY" = 1 ]; then
    aws secretsmanager create-secret --name "$SECRET_NAME" \
      --description "context-mcp: Voyage + Chroma Cloud credentials and bearer token" \
      --secret-string "$SECRET_JSON" --no-cli-pager >/dev/null
  else
    info "[dry-run] would create-secret with 5 keys (values not shown)"
  fi
fi
SECRET_ARN="arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:${SECRET_NAME}"

# --- Cognito (the OAuth authorization server) -------------------------------
# claude.ai authenticates custom connectors by running an OAuth 2.1 flow, and
# on accounts without the request-header beta there is no way to give it a
# static token. So it needs a real authorization server. Cognito is used rather
# than writing one, and its free tier covers far more than one user.
#
# Note Cognito does NOT support dynamic client registration, which is exactly
# why claude.ai exposes "OAuth Client ID / Client Secret" fields — you paste the
# values printed at the end into those. It is also why Claude Code keeps using
# the static bearer token: it registers dynamically and calls back on ephemeral
# loopback ports, neither of which Cognito can accommodate.
say "Cognito user pool: $COGNITO_POOL_NAME"
POOL_ID="$(aws cognito-idp list-user-pools --max-results 60 \
  --query "UserPools[?Name=='${COGNITO_POOL_NAME}'].Id | [0]" --output text 2>/dev/null || echo None)"

if [ "$POOL_ID" != "None" ] && [ -n "$POOL_ID" ]; then
  info "already exists: $POOL_ID"
elif [ "$APPLY" = 1 ]; then
  POOL_ID="$(aws cognito-idp create-user-pool --pool-name "$COGNITO_POOL_NAME" \
    --auto-verified-attributes email --username-attributes email \
    --policies '{"PasswordPolicy":{"MinimumLength":12,"RequireUppercase":true,"RequireLowercase":true,"RequireNumbers":true,"RequireSymbols":false}}' \
    --query 'UserPool.Id' --output text)"
  info "created: $POOL_ID"
else
  info "[dry-run] would create user pool"
  POOL_ID="eu-west-1_DRYRUN"
fi

# The hosted-UI domain is what serves the actual login page. Without it, the
# discovery document has no authorization_endpoint and the whole flow is dead.
# Prefix must be globally unique across all AWS accounts; the account id makes
# it unique and, unlike a random suffix, keeps re-runs idempotent.
say "Cognito hosted-UI domain: $COGNITO_DOMAIN"
if aws cognito-idp describe-user-pool-domain --domain "$COGNITO_DOMAIN" \
     --query 'DomainDescription.UserPoolId' --output text 2>/dev/null | grep -q '^eu-'; then
  info "already exists"
else
  run aws cognito-idp create-user-pool-domain --domain "$COGNITO_DOMAIN" \
    --user-pool-id "$POOL_ID"
fi

# A resource server gives us a custom scope. Its identifier is a plain string,
# deliberately NOT the Function URL — that hostname doesn't exist yet at this
# point, and baking it in would force another ordering dependency.
say "Cognito resource server: $COGNITO_RESOURCE_ID (scope: $COGNITO_SCOPE)"
run aws cognito-idp create-resource-server --user-pool-id "$POOL_ID" \
  --identifier "$COGNITO_RESOURCE_ID" --name "context-mcp" \
  --scopes "ScopeName=${COGNITO_SCOPE},ScopeDescription=Access the context store" \
  --no-cli-pager 2>/dev/null || info "resource server already exists (or dry run)"

FULL_SCOPE="${COGNITO_RESOURCE_ID}/${COGNITO_SCOPE}"

say "Cognito app client for claude.ai"
CLIENT_ID="$(aws cognito-idp list-user-pool-clients --user-pool-id "$POOL_ID" --max-results 60 \
  --query "UserPoolClients[?ClientName=='${COGNITO_CLIENT_NAME}'].ClientId | [0]" --output text 2>/dev/null || echo None)"
if [ "$CLIENT_ID" != "None" ] && [ -n "$CLIENT_ID" ]; then
  info "already exists: $CLIENT_ID"
elif [ "$APPLY" = 1 ]; then
  # generate-secret because claude.ai asks for a client SECRET as well as an id.
  CLIENT_ID="$(aws cognito-idp create-user-pool-client --user-pool-id "$POOL_ID" \
    --client-name "$COGNITO_CLIENT_NAME" --generate-secret \
    --allowed-o-auth-flows code --allowed-o-auth-flows-user-pool-client \
    --allowed-o-auth-scopes openid "$FULL_SCOPE" \
    --callback-urls "$CLAUDE_AI_CALLBACK" \
    --supported-identity-providers COGNITO \
    --query 'UserPoolClient.ClientId' --output text)"
  info "created: $CLIENT_ID"
else
  info "[dry-run] would create app client with callback $CLAUDE_AI_CALLBACK"
  CLIENT_ID="dryrunclientid"
fi

if [ "$APPLY" = 1 ]; then
  CLIENT_SECRET="$(aws cognito-idp describe-user-pool-client --user-pool-id "$POOL_ID" \
    --client-id "$CLIENT_ID" --query 'UserPoolClient.ClientSecret' --output text)"
else
  CLIENT_SECRET="(dry run)"
fi

# Separate app client for Claude Code. PUBLIC (no --generate-secret): a CLI on
# your laptop cannot keep a secret, so it uses PKCE instead — which is why this
# client is configured differently from the claude.ai one rather than shared.
# Giving Claude Code a real OAuth identity is what makes it possible to stop
# accepting the static bearer token on the MCP endpoint at all.
say "Cognito app client for Claude Code (public, PKCE)"
CODE_CLIENT_ID="$(aws cognito-idp list-user-pool-clients --user-pool-id "$POOL_ID" --max-results 60 \
  --query "UserPoolClients[?ClientName=='${COGNITO_CODE_CLIENT_NAME}'].ClientId | [0]" --output text 2>/dev/null || echo None)"
if [ "$CODE_CLIENT_ID" != "None" ] && [ -n "$CODE_CLIENT_ID" ]; then
  info "already exists: $CODE_CLIENT_ID"
elif [ "$APPLY" = 1 ]; then
  CODE_CLIENT_ID="$(aws cognito-idp create-user-pool-client --user-pool-id "$POOL_ID" \
    --client-name "$COGNITO_CODE_CLIENT_NAME" \
    --allowed-o-auth-flows code --allowed-o-auth-flows-user-pool-client \
    --allowed-o-auth-scopes openid "$FULL_SCOPE" \
    --callback-urls "http://localhost:${CLAUDE_CODE_CALLBACK_PORT}/callback" \
    --supported-identity-providers COGNITO \
    --query 'UserPoolClient.ClientId' --output text)"
  info "created: $CODE_CLIENT_ID"
else
  info "[dry-run] would create public client, callback http://localhost:${CLAUDE_CODE_CALLBACK_PORT}/callback"
  CODE_CLIENT_ID="dryruncodeclientid"
fi

# Both clients are accepted by the token verifier.
ALLOWED_CLIENT_IDS="${CLIENT_ID},${CODE_CLIENT_ID}"

# One user — you. Cognito needs an actual identity to authenticate against;
# self-signup stays off so nobody else can create an account against this pool.
say "Cognito user: $COGNITO_USER_EMAIL"
if [ "$APPLY" = 1 ]; then
  if aws cognito-idp admin-get-user --user-pool-id "$POOL_ID" \
       --username "$COGNITO_USER_EMAIL" >/dev/null 2>&1; then
    info "already exists"
  else
    TEMP_PW="$(python3 -c 'import secrets,string; a=string.ascii_letters+string.digits; print("Aa1"+"".join(secrets.choice(a) for _ in range(17)))')"
    aws cognito-idp admin-create-user --user-pool-id "$POOL_ID" \
      --username "$COGNITO_USER_EMAIL" --message-action SUPPRESS \
      --user-attributes Name=email,Value="$COGNITO_USER_EMAIL" Name=email_verified,Value=true \
      --no-cli-pager >/dev/null
    # Permanent, so the first login isn't a forced password-change screen in
    # the middle of an OAuth redirect — which is confusing and easy to fail.
    aws cognito-idp admin-set-user-password --user-pool-id "$POOL_ID" \
      --username "$COGNITO_USER_EMAIL" --password "$TEMP_PW" --permanent
    info "created with password: $TEMP_PW"
    info "SAVE THIS — it is not stored anywhere and is what you type when claude.ai redirects you."
  fi
else
  info "[dry-run] would create user and set a generated permanent password"
fi

# --- Build the deployment bundle -------------------------------------------
# Same steps as the CI workflow, so what you test here is what CI ships.
say "Building deployment bundle ($ARCH)"
if [ "$APPLY" = 1 ]; then
  rm -rf "$BUILD_DIR" "$ZIP_PATH"; mkdir -p "$BUILD_DIR"
  pip install -q --target "$BUILD_DIR" \
    --platform manylinux2014_aarch64 --python-version 3.13 \
    --implementation cp --only-binary=:all: \
    -r "$REPO_ROOT/requirements-lambda.txt"
  cp -r "$REPO_ROOT/mcp_server" "$REPO_ROOT/shared" "$BUILD_DIR/"
  # __pycache__ is safe to drop; .dist-info is NOT (importlib.metadata needs it).
  find "$BUILD_DIR" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
  ( cd "$BUILD_DIR" && zip -qr "$ZIP_PATH" . )
  UNZIPPED_MB=$(du -sm "$BUILD_DIR" | cut -f1)
  info "unzipped ${UNZIPPED_MB}MB (limit 250) / zipped $(du -m "$ZIP_PATH" | cut -f1)MB (limit 50)"
  [ "$UNZIPPED_MB" -lt 240 ] || { echo "bundle too large for zip deploy"; exit 1; }
else
  info "[dry-run] would pip install -r requirements-lambda.txt --platform manylinux2014_aarch64"
  info "[dry-run] would zip mcp_server/ + shared/ + deps  (~108MB unzipped, ~33MB zipped)"
fi

# --- Lambda execution role --------------------------------------------------
say "Lambda execution role: $LAMBDA_ROLE"
if aws iam get-role --role-name "$LAMBDA_ROLE" >/dev/null 2>&1; then
  info "already exists"
else
  run aws iam create-role --role-name "$LAMBDA_ROLE" \
    --description "Execution role for the context-mcp Lambda" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' --no-cli-pager
  run aws iam attach-role-policy --role-name "$LAMBDA_ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  # IAM is eventually consistent; creating a Lambda with a role that has not
  # propagated yet fails with InvalidParameterValueException.
  info "waiting 10s for IAM propagation"
  [ "$APPLY" = 1 ] && sleep 10
fi
LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE}"

# Read access to exactly one secret, by name. The trailing -?????? matches the
# random suffix AWS appends to every secret ARN.
say "Granting $LAMBDA_ROLE read access to $SECRET_NAME"
run aws iam put-role-policy --role-name "$LAMBDA_ROLE" --policy-name read-secret \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{
    \"Effect\":\"Allow\",
    \"Action\":[\"secretsmanager:GetSecretValue\"],
    \"Resource\":\"${SECRET_ARN}-??????\"}]}" --no-cli-pager

# --- Lambda function --------------------------------------------------------
# Only non-secret configuration lives here: which secret to read, and which
# hostname to accept. MCP_ALLOWED_HOST is a placeholder now and corrected in a
# second pass below, because the Function URL hostname is generated by AWS and
# cannot be known until the URL exists. It must be the EXACT hostname — "*" is
# compared literally by the MCP host validator and rejects everything with 421.
ENV_VARS="Variables={CONTEXT_MCP_SECRET_ID=$SECRET_NAME,MCP_ALLOWED_HOST=placeholder.invalid,COGNITO_REGION=$REGION,COGNITO_USER_POOL_ID=$POOL_ID,COGNITO_ALLOWED_CLIENT_IDS=$ALLOWED_CLIENT_IDS,MCP_REQUIRED_SCOPES=$FULL_SCOPE}"

say "Lambda function: $FUNCTION_NAME"
if aws lambda get-function --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  info "already exists — updating code"
  run aws lambda update-function-code --function-name "$FUNCTION_NAME" \
    --zip-file "fileb://$ZIP_PATH" --no-cli-pager
  run aws lambda wait function-updated --function-name "$FUNCTION_NAME"
else
  run aws lambda create-function --function-name "$FUNCTION_NAME" \
    --runtime "$RUNTIME" --architectures "$ARCH" \
    --role "$LAMBDA_ROLE_ARN" --handler "$HANDLER" \
    --zip-file "fileb://$ZIP_PATH" \
    --timeout "$TIMEOUT_S" --memory-size "$MEMORY_MB" \
    --environment "$ENV_VARS" \
    --snap-start ApplyOn=PublishedVersions \
    --description "context-mcp — cross-machine context store over MCP" \
    --no-cli-pager
  run aws lambda wait function-active --function-name "$FUNCTION_NAME"
fi

# SnapStart is what takes cold start from ~5.3s to ~1.5s. It applies ONLY to
# published versions, never $LATEST — which is why everything below publishes
# a version and points an alias at it.
say "Ensuring SnapStart is enabled"
run aws lambda update-function-configuration --function-name "$FUNCTION_NAME" \
  --snap-start ApplyOn=PublishedVersions --no-cli-pager
run aws lambda wait function-updated --function-name "$FUNCTION_NAME"

# --- Publish + alias --------------------------------------------------------
wait_version_active() {
  local ver="$1"
  for i in $(seq 1 40); do
    local state
    state="$(aws lambda get-function-configuration --function-name "$FUNCTION_NAME" \
      --qualifier "$ver" --query State --output text)"
    info "  version $ver state=$state"
    [ "$state" = "Active" ] && return 0
    [ "$state" = "Failed" ] && { echo "version $ver failed"; exit 1; }
    sleep 15
  done
  echo "timed out waiting for version $ver"; exit 1
}

say "Publishing version and creating alias '$ALIAS'"
if [ "$APPLY" = 1 ]; then
  VERSION="$(aws lambda publish-version --function-name "$FUNCTION_NAME" --query Version --output text)"
  info "published version $VERSION"
  wait_version_active "$VERSION"
  if aws lambda get-alias --function-name "$FUNCTION_NAME" --name "$ALIAS" >/dev/null 2>&1; then
    aws lambda update-alias --function-name "$FUNCTION_NAME" --name "$ALIAS" \
      --function-version "$VERSION" --no-cli-pager >/dev/null
  else
    aws lambda create-alias --function-name "$FUNCTION_NAME" --name "$ALIAS" \
      --function-version "$VERSION" --no-cli-pager >/dev/null
  fi
  info "alias '$ALIAS' -> version $VERSION"
else
  info "[dry-run] would publish a version, wait for SnapStart, and point '$ALIAS' at it"
fi

# --- Function URL (on the alias, so traffic gets SnapStart) -----------------
say "Function URL on alias '$ALIAS'"
if [ "$APPLY" = 1 ]; then
  if aws lambda get-function-url-config --function-name "$FUNCTION_NAME" \
       --qualifier "$ALIAS" >/dev/null 2>&1; then
    FUNCTION_URL="$(aws lambda get-function-url-config --function-name "$FUNCTION_NAME" \
      --qualifier "$ALIAS" --query FunctionUrl --output text)"
    info "already exists"
  else
    FUNCTION_URL="$(aws lambda create-function-url-config --function-name "$FUNCTION_NAME" \
      --qualifier "$ALIAS" --auth-type NONE --query FunctionUrl --output text)"
    # Without this, the URL returns 403 — the URL config alone does not grant invoke.
    aws lambda add-permission --function-name "$FUNCTION_NAME" --qualifier "$ALIAS" \
      --statement-id FunctionURLAllowPublicAccess --action lambda:InvokeFunctionUrl \
      --principal '*' --function-url-auth-type NONE --no-cli-pager >/dev/null
  fi
  info "url: $FUNCTION_URL"
  URL_HOST="$(echo "$FUNCTION_URL" | sed -E 's#^https?://##; s#/$##')"

  # Second pass: now that the hostname exists, write it into the config and
  # republish, because environment variables are part of a version's immutable
  # config — the already-published version still carries the placeholder.
  # (Credentials do NOT need this treatment: they live in Secrets Manager and
  # are read at runtime, so rotating one never requires a redeploy.)
  say "Setting MCP_ALLOWED_HOST=$URL_HOST and republishing"
  aws lambda update-function-configuration --function-name "$FUNCTION_NAME" \
    --environment "Variables={CONTEXT_MCP_SECRET_ID=$SECRET_NAME,MCP_ALLOWED_HOST=$URL_HOST,COGNITO_REGION=$REGION,COGNITO_USER_POOL_ID=$POOL_ID,COGNITO_ALLOWED_CLIENT_IDS=$ALLOWED_CLIENT_IDS,MCP_REQUIRED_SCOPES=$FULL_SCOPE}" \
    --no-cli-pager >/dev/null
  aws lambda wait function-updated --function-name "$FUNCTION_NAME"
  VERSION="$(aws lambda publish-version --function-name "$FUNCTION_NAME" --query Version --output text)"
  wait_version_active "$VERSION"
  aws lambda update-alias --function-name "$FUNCTION_NAME" --name "$ALIAS" \
    --function-version "$VERSION" --no-cli-pager >/dev/null
  info "alias '$ALIAS' -> version $VERSION (with real MCP_ALLOWED_HOST)"
else
  info "[dry-run] would create a Function URL (AuthType=NONE, bearer-protected),"
  info "[dry-run] then set MCP_ALLOWED_HOST to its hostname and republish,"
  info "[dry-run] because env vars are baked into a published version."
fi

# --- GitHub Actions OIDC ----------------------------------------------------
say "GitHub Actions OIDC provider"
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
  info "already exists"
else
  run aws iam create-open-id-connect-provider \
    --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
    --no-cli-pager
fi

say "Deploy role: $DEPLOY_ROLE"
# Trust is scoped to this repo specifically. Without the sub condition, ANY
# GitHub repo in the world could assume this role.
TRUST=$(cat <<JSON
{"Version":"2012-10-17","Statement":[{
  "Effect":"Allow",
  "Principal":{"Federated":"$OIDC_ARN"},
  "Action":"sts:AssumeRoleWithWebIdentity",
  "Condition":{
    "StringEquals":{"token.actions.githubusercontent.com:aud":"sts.amazonaws.com"},
    "StringLike":{"token.actions.githubusercontent.com:sub":"repo:${GITHUB_REPO}:*"}
  }}]}
JSON
)
if aws iam get-role --role-name "$DEPLOY_ROLE" >/dev/null 2>&1; then
  info "already exists — refreshing trust policy"
  run aws iam update-assume-role-policy --role-name "$DEPLOY_ROLE" \
    --policy-document "$TRUST" --no-cli-pager
else
  run aws iam create-role --role-name "$DEPLOY_ROLE" \
    --description "Assumed by GitHub Actions to deploy context-mcp" \
    --assume-role-policy-document "$TRUST" --no-cli-pager
fi

# Least privilege: enough to ship code and move the alias, nothing more.
# Configuration read is included because the workflow polls SnapStart status —
# harmless now that the function's environment holds no credentials, which is
# precisely why they were moved to Secrets Manager. Note there is no
# secretsmanager permission here at all, so CI cannot read the secret.
DEPLOY_POLICY=$(cat <<JSON
{"Version":"2012-10-17","Statement":[{
  "Effect":"Allow",
  "Action":["lambda:UpdateFunctionCode","lambda:PublishVersion","lambda:UpdateAlias",
            "lambda:CreateAlias","lambda:GetAlias","lambda:GetFunction",
            "lambda:GetFunctionConfiguration"],
  "Resource":["arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}",
              "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}:*"]
}]}
JSON
)
run aws iam put-role-policy --role-name "$DEPLOY_ROLE" \
  --policy-name deploy --policy-document "$DEPLOY_POLICY" --no-cli-pager

# --- Done -------------------------------------------------------------------
say "Done"
if [ "$APPLY" = 1 ]; then
  cat <<EOF

  Function URL : $FUNCTION_URL
  Alias        : $ALIAS -> version $VERSION
  Secret       : $SECRET_NAME
  Bearer token : $AUTH_TOKEN   (also in .env and in the secret)

  Cognito:
    User pool         : $POOL_ID
    Login as          : $COGNITO_USER_EMAIL
    claude.ai client  : $CLIENT_ID
    claude.ai secret  : $CLIENT_SECRET
    Claude Code client: $CODE_CLIENT_ID  (public, PKCE, no secret)
    Scope             : $FULL_SCOPE

  Next steps, by hand:

  0a. Add the connector on claude.ai:
        URL: ${FUNCTION_URL}mcp
        Advanced settings -> OAuth Client ID / Client Secret = the pair above.
      claude.ai redirects you to a Cognito login page; sign in as
      $COGNITO_USER_EMAIL with the password printed earlier.

  0b. Add it to Claude Code with real OAuth (not the static token):
        claude mcp add --transport http context-mcp ${FUNCTION_URL}mcp \\
          --client-id $CODE_CLIENT_ID \\
          --callback-port $CLAUDE_CODE_CALLBACK_PORT

      The port must stay $CLAUDE_CODE_CALLBACK_PORT — Cognito matches redirect
      URIs exactly, port included.

      Known bug to watch for: some Claude Code versions write the client id as
      "clientId" while the OAuth code reads "client_id", so the flow ignores it
      and fails with "does not support dynamic client registration". If that
      happens, fix the key by hand in ~/.claude.json.

      Fallback if OAuth won't cooperate — the static token still works:
        claude mcp add --transport http context-mcp ${FUNCTION_URL}mcp \\
          --header "Authorization: Bearer $AUTH_TOKEN"

  0c. Once BOTH clients authenticate via OAuth, close the shared-password hole:
        aws lambda update-function-configuration --function-name $FUNCTION_NAME \\
          --environment "Variables={...,MCP_ALLOW_STATIC_TOKEN=0}"
      then publish a version and move the alias. After that the static token
      only opens /map, not the store.

  1. Add this repository secret on GitHub so CI can deploy:
       AWS_DEPLOY_ROLE_ARN = arn:aws:iam::${ACCOUNT_ID}:role/${DEPLOY_ROLE}

  2. Smoke-test the endpoint (expect a JSON-RPC result, not 401/421):
       curl -sS -X POST "${FUNCTION_URL}mcp" \\
         -H "Authorization: Bearer $AUTH_TOKEN" \\
         -H "Content-Type: application/json" \\
         -H "Accept: application/json, text/event-stream" \\
         -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'

     Then confirm auth is actually enforced — this MUST return 401:
       curl -s -o /dev/null -w '%{http_code}\\n' -X POST "${FUNCTION_URL}mcp" \\
         -H "Content-Type: application/json" -d '{}'

  3. Point Claude Code and the claude.ai connector at ${FUNCTION_URL}mcp
     with that bearer token.

  To rotate a credential later: update the secret and the next cold start
  picks it up. No redeploy, no new version.

EOF
else
  cat <<'EOF'

  Dry run only — nothing was created.
  Re-run with --apply once you are happy with the plan above.

EOF
fi
