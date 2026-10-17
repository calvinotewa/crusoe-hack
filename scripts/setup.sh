#!/usr/bin/env bash
# Developer setup: installs tooling, signs you in to Modal, Neon and Crusoe, gives you a
# personal Neon branch (database + bucket), and generates a complete, verified .env.
#
# Safe to re-run: steps that are already done are skipped.
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_FILE=".env"
# Team Neon project "crusoe-hack" (aws-us-east-2). Not a secret; override with --project-id.
PROJECT_ID="${NEON_PROJECT_ID:-patient-art-99467040}"
BRANCH=""
REFRESH_STORAGE=0

usage() {
  cat <<EOF
Usage: scripts/setup.sh [options]

  --project-id ID      Neon project to link (default: \$NEON_PROJECT_ID, else the team project $PROJECT_ID)
  --branch NAME        Your Neon branch (default: dev-<your git email name>)
  --refresh-storage    Issue a new Neon bucket credential even if .env already has one
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --refresh-storage) REFRESH_STORAGE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# --- output helpers ---------------------------------------------------------------------
bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die() { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
confirm() {
  local reply
  read -r -p "  $1 [Y/n] " reply
  [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}
open_url() {
  if have open; then open "$1" >/dev/null 2>&1 || true
  elif have xdg-open; then xdg-open "$1" >/dev/null 2>&1 || true
  fi
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# --- .env helpers (see scripts/dotenv_tool.py) ------------------------------------------
dotenv() { uv run --quiet python scripts/dotenv_tool.py "$@"; }
env_get() { dotenv get "$ENV_FILE" "$1"; }
# Usage: printf '%s' "$value" | env_set KEY   (stdin keeps secrets out of the process list)
env_set() { dotenv set "$ENV_FILE" "$1"; }

# --- 1. tooling -------------------------------------------------------------------------
step "Tooling"

if ! have uv; then
  warn "uv (Python package manager) is not installed."
  confirm "Install uv from astral.sh?" || die "uv is required: https://docs.astral.sh/uv/"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version | awk '{print $2}')"

uv sync --quiet
ok "Python dependencies installed (includes the Modal CLI: uv run modal)"

have npm || die "Node.js/npm is required for the Neon CLI. Install it (e.g. 'brew install node') and re-run."

# The current CLI is the npm package `neon` (v8+). Homebrew's older `neonctl` 2.x also
# installs a `neon` binary but lacks the branch-storage and env commands used below.
neon_major() { neon --version 2>/dev/null | head -1 | sed -E 's/^v?([0-9]+).*/\1/'; }
if ! have neon; then
  confirm "Install the Neon CLI (npm install -g neon)?" || die "The Neon CLI is required."
  npm install -g neon@latest --silent
  hash -r
fi
if [[ "$(neon_major)" =~ ^[0-9]+$ ]] && (( $(neon_major) >= 8 )); then
  NEON=(neon)
  ok "Neon CLI $(neon --version | head -1)"
else
  warn "'neon' on your PATH is $(neon --version 2>/dev/null | head -1) (old neonctl); using npx neon@latest instead."
  warn "To fix permanently: brew uninstall neonctl && npm install -g neon"
  NEON=(npx --yes neon@latest)
fi

# --- 2. .env ----------------------------------------------------------------------------
step "Local .env"
[[ -f "$ENV_FILE" ]] && existed=1 || existed=0
dotenv normalize "$ENV_FILE" .env.example
if (( existed )); then
  ok "Updated $ENV_FILE to the current template (your existing values are kept)"
else
  ok "Generated $ENV_FILE from .env.example"
fi
if [[ -f .env.local ]]; then
  warn ".env.local exists and overrides $ENV_FILE in the app. Delete it unless you need it."
fi

# --- 3. Modal ---------------------------------------------------------------------------
step "Modal"
if uv run --quiet modal token info >/dev/null 2>&1; then
  ok "Already authenticated ($(uv run --quiet modal token info 2>/dev/null | grep -iE 'workspace' | head -1 | xargs || echo 'token found'))"
else
  bold "  Opening a browser to sign in to Modal..."
  uv run modal token new
  ok "Modal token saved to ~/.modal.toml"
fi
dotenv modal-tokens "$ENV_FILE" || die "Could not read the Modal token. Run 'uv run modal token new' and re-run."
ok "Wrote MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to $ENV_FILE"

# --- 4. Neon ----------------------------------------------------------------------------
step "Neon"
if "${NEON[@]}" me >/dev/null 2>&1; then
  ok "Already signed in"
else
  bold "  Opening a browser to sign in to Neon..."
  "${NEON[@]}" login
fi

if [[ -f .neon ]] && grep -q "\"projectId\": *\"$PROJECT_ID\"" .neon 2>/dev/null; then
  ok "Project $PROJECT_ID already linked (.neon)"
else
  "${NEON[@]}" link --project-id "$PROJECT_ID" --yes --no-env-pull >/dev/null ||
    die "Could not link Neon project $PROJECT_ID. Ask the project owner to invite your Neon account."
  ok "Linked Neon project $PROJECT_ID"
fi

if [[ -z "$BRANCH" ]]; then
  who="$(git config user.email 2>/dev/null | cut -d@ -f1)"
  who="${who:-$USER}"
  BRANCH="dev-$(printf '%s' "$who" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//;s/-$//')"
fi
"${NEON[@]}" checkout "$BRANCH" --create --no-env-pull >"$TMP_DIR/checkout.log" 2>&1 ||
  die "Could not check out Neon branch $BRANCH: $(tail -3 "$TMP_DIR/checkout.log")"
ok "Using your personal branch: $BRANCH (a copy-on-write clone; safe to experiment)"

BUCKET="$(env_get S3_BUCKET)"
BUCKET="${BUCKET:-artifacts}"
# Always pass --branch: bucket commands do not reliably follow the branch pinned in .neon.
if "${NEON[@]}" bucket list --branch "$BRANCH" -o json 2>/dev/null | uv run --quiet python -c '
import json, sys
want = sys.argv[1]
def names(node):
    if isinstance(node, dict):
        if node.get("name") == want:
            yield want
        for v in node.values():
            yield from names(v)
    elif isinstance(node, list):
        for v in node:
            yield from names(v)
sys.exit(0 if any(names(json.load(sys.stdin))) else 1)' "$BUCKET"; then
  ok "Bucket '$BUCKET' exists on $BRANCH"
elif "${NEON[@]}" bucket create "$BUCKET" --branch "$BRANCH" >"$TMP_DIR/bucket.log" 2>&1; then
  ok "Created bucket '$BUCKET' on $BRANCH"
else
  die "Could not create bucket '$BUCKET' on $BRANCH ($(tail -1 "$TMP_DIR/bucket.log")). Neon Object Storage (beta) must be available for the project's region (aws-us-east-1/2, eu-central-1, ap-southeast-1)."
fi

NEON_ENV="$TMP_DIR/neon.env"
NEON_KEYS=(DATABASE_URL DATABASE_URL_UNPOOLED NEON_BRANCH)
pull_args=(env pull --branch "$BRANCH" --file "$NEON_ENV" --service postgres)
if [[ "$REFRESH_STORAGE" == 1 || -z "$(env_get AWS_ACCESS_KEY_ID)" || "$(env_get NEON_BRANCH)" != "$BRANCH" ]]; then
  pull_args+=(--service object-storage)   # issues a new bucket credential
  NEON_KEYS+=(AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL_S3 AWS_REGION)
  storage_msg="new bucket credential issued"
else
  storage_msg="kept existing bucket credential; --refresh-storage rotates it"
fi
: > "$NEON_ENV"
"${NEON[@]}" "${pull_args[@]}" >"$TMP_DIR/pull.log" 2>&1 ||
  die "neon env pull failed for branch $BRANCH: $(tail -3 "$TMP_DIR/pull.log")"
for key in "${NEON_KEYS[@]}"; do
  [[ "$key" == NEON_BRANCH || -n "$(dotenv get "$NEON_ENV" "$key")" ]] || die "neon env pull did not return $key for branch $BRANCH."
done
dotenv merge "$ENV_FILE" "$NEON_ENV" "${NEON_KEYS[@]}"
ok "Wrote Neon Postgres and bucket settings to $ENV_FILE ($storage_msg)"

# --- 5. Crusoe --------------------------------------------------------------------------
step "Crusoe"
BASE_URL="$(env_get CRUSOE_BASE_URL)"
BASE_URL="${BASE_URL:-https://api.inference.crusoecloud.com/v1}"

# Prints the HTTP status; the body is written to $TMP_DIR/models.json.
crusoe_check() {
  printf 'Authorization: Bearer %s\n' "$1" |
    curl -s -o "$TMP_DIR/models.json" -w '%{http_code}' -H @- "$BASE_URL/models" || true
}

KEY="$(env_get CRUSOE_API_KEY)"
status="$([[ -n "$KEY" ]] && crusoe_check "$KEY" || echo none)"
if [[ "$status" == 200 ]]; then
  ok "Existing CRUSOE_API_KEY is valid"
else
  [[ "$status" != none ]] && warn "CRUSOE_API_KEY in $ENV_FILE was rejected (HTTP $status)"
  bold "  Create an API key: Crusoe Console → Intelligence Foundry → API Keys → Create API Key"
  echo "  https://console.crusoecloud.com/foundry"
  open_url "https://console.crusoecloud.com/foundry"
  while true; do
    read -r -s -p "  Paste your Crusoe API key (input hidden): " KEY ||
      die "No Crusoe key entered. Re-run scripts/setup.sh in an interactive terminal."
    echo
    [[ -n "$KEY" ]] || continue
    status="$(crusoe_check "$KEY")"
    [[ "$status" == 200 ]] && break
    warn "Key rejected (HTTP $status). Check you copied the whole key and it belongs to the right project."
  done
  printf '%s' "$KEY" | env_set CRUSOE_API_KEY
  ok "CRUSOE_API_KEY saved to $ENV_FILE"
fi
unset KEY

for var in CRUSOE_DEFAULT_MODEL CRUSOE_JUDGE_MODEL; do
  model="$(env_get "$var")"
  [[ -z "$model" ]] && continue
  if uv run --quiet python -c '
import json, sys
ids = {m["id"] for m in json.load(open(sys.argv[1])).get("data", [])}
sys.exit(0 if sys.argv[2] in ids else 1)' "$TMP_DIR/models.json" "$model"; then
    ok "$var=$model is available"
  else
    warn "$var=$model is not in your model list (IDs are case-sensitive; see GET /v1/models)"
  fi
done

# --- 6. Database schema + smoke test ----------------------------------------------------
step "Database migrations"
uv run --quiet alembic upgrade head
ok "Schema is up to date on $BRANCH"

step "Verify .env"
if ! uv run --quiet python scripts/check_env.py; then
  die "$ENV_FILE is not fully working yet. Fix the items above and re-run scripts/setup.sh."
fi

step "Done"
echo "  $ENV_FILE is complete and verified (branch: $BRANCH; git-ignored, chmod 600)."
echo "  Re-check anytime:  uv run python scripts/check_env.py"
echo "  Start the API:  uv run fastapi dev app/main.py   →  http://localhost:8000/docs"
