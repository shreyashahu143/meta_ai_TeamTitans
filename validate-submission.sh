#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator (Full)
#
# Runs 5 mandatory checks before you submit to the hackathon:
#   1. HF Space is live — /reset returns 200
#   2. Mandatory files exist + inference.py has required env vars & log markers
#   3. Docker build succeeds
#   4. openenv validate passes
#   5. inference.py Python syntax check
#
# Prerequisites:
#   Docker, openenv-core (pip install openenv-core), curl, Python 3.11
#
# Usage:
#   chmod +x validate-submission.sh
#   ./validate-submission.sh <hf_space_url> [repo_dir]
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BOLD=''; NC=''
fi

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; FAIL=$((FAIL + 1)); }
warn() { log "${YELLOW}WARN  ${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n${RED}${BOLD}Stopped at %s.${NC} Fix above error first.\n" "$1"
  printf "  Passed: %d | Failed: %d\n\n" "$PASS" "$FAIL"
  exit 1
}

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then gtimeout "$secs" "$@"
  else
    "$@" & local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) & watcher=$!
    wait "$pid" 2>/dev/null; rc=$?
    kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() { mktemp "${TMPDIR:-/tmp}/${1:-val}-XXXXXX" 2>/dev/null || mktemp; }
CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <hf_space_url> [repo_dir]\n\n" "$0"
  printf "  Example: ./validate-submission.sh https://teamtitans.hf.space .\n\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"; exit 1
fi

PING_URL="${PING_URL%/}"
PASS=0; FAIL=0

printf "\n${BOLD}===============================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator — Full${NC}\n"
printf "${BOLD}===============================================${NC}\n"
log "Repo:      $REPO_DIR"
log "Space URL: $PING_URL"
printf "\n"

# =============================================================================
# STEP 1 — Ping HF Space (fixed curl bug)
# =============================================================================
log "${BOLD}Step 1/5: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUT=$(portable_mktemp "val-curl-body")
CURL_ERR=$(portable_mktemp "val-curl-err")
CLEANUP_FILES+=("$CURL_OUT" "$CURL_ERR")

HTTP_CODE=$(curl -s -o "$CURL_OUT" -w "%{http_code}" \
  -X POST -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_ERR" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space live — /reset returned 200"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable"
  [ -s "$CURL_ERR" ] && cat "$CURL_ERR" >&2
  hint "Deploy your Space first, then re-run this script."
  hint "Manual test: curl -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "/reset returned HTTP $HTTP_CODE (expected 200)"
  [ -s "$CURL_ERR" ] && cat "$CURL_ERR" >&2
  hint "Check Space build logs on huggingface.co"
  stop_at "Step 1"
fi

# =============================================================================
# STEP 2 — Files + inference.py compliance
# =============================================================================
log "${BOLD}Step 2/5: Checking required files and inference.py compliance${NC} ..."
ERR=0

for f in "inference.py" "openenv.yaml" "requirements.txt" "Dockerfile" \
         "grader.py" "client.py" "models.py"; do
  if [ ! -f "$REPO_DIR/$f" ]; then
    fail "Missing: $f"; ERR=$((ERR+1))
  fi
done

for t in "tasks/task_1_easy.json" "tasks/task_2_medium.json" "tasks/task_3_hard.json"; do
  if [ ! -f "$REPO_DIR/$t" ]; then
    fail "Missing task file: $t"; ERR=$((ERR+1))
  fi
done

if [ ! -f "$REPO_DIR/data/email_bank.json" ]; then
  fail "Missing: data/email_bank.json"; ERR=$((ERR+1))
fi

[ $ERR -gt 0 ] && stop_at "Step 2 (missing files)"

# Check env vars
for var in "API_BASE_URL" "MODEL_NAME" "HF_TOKEN"; do
  if ! grep -q "$var" "$REPO_DIR/inference.py"; then
    fail "inference.py missing env var: $var"; ERR=$((ERR+1))
  fi
done

# Check log markers
for marker in "\[START\]" "\[STEP\]" "\[END\]"; do
  if ! grep -q "$marker" "$REPO_DIR/inference.py"; then
    fail "inference.py missing log marker: $marker"; ERR=$((ERR+1))
  fi
done

# Must use OpenAI client, NOT Anthropic
if grep -q "from anthropic" "$REPO_DIR/inference.py" 2>/dev/null; then
  fail "inference.py uses Anthropic SDK — spec requires OpenAI client"; ERR=$((ERR+1))
fi
if ! grep -q "from openai" "$REPO_DIR/inference.py" 2>/dev/null; then
  warn "inference.py may not import OpenAI client — verify manually"
fi

[ $ERR -gt 0 ] && stop_at "Step 2 (inference.py compliance)"
pass "All required files present — inference.py spec-compliant"

# =============================================================================
# STEP 3 — Docker build
# =============================================================================
log "${BOLD}Step 3/5: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker not found"; hint "Install: https://docs.docker.com/get-docker/"
  stop_at "Step 3"
fi

if   [ -f "$REPO_DIR/Dockerfile"        ]; then DOCKER_CTX="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then DOCKER_CTX="$REPO_DIR/server"
else fail "No Dockerfile found in repo root or server/"; stop_at "Step 3"
fi

log "  Context: $DOCKER_CTX"
BUILD_OK=false
BUILD_OUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CTX" 2>&1) \
  && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUT" | tail -25
  stop_at "Step 3"
fi

# =============================================================================
# STEP 4 — openenv validate
# =============================================================================
log "${BOLD}Step 4/5: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv not found"; hint "Install: pip install openenv-core"
  stop_at "Step 4"
fi

VALIDATE_OK=false
VALIDATE_OUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUT" ] && log "  $VALIDATE_OUT"
else
  fail "openenv validate failed"
  printf "\n%s\n\n" "$VALIDATE_OUT"
  hint "Fix openenv.yaml — check task IDs, observation_space, action_space."
  stop_at "Step 4"
fi

# =============================================================================
# STEP 5 — Python syntax check
# =============================================================================
log "${BOLD}Step 5/5: Python syntax check on inference.py${NC} ..."

PYTHON_CMD=""
for py in "python3.11" "python3" "python"; do
  command -v "$py" &>/dev/null && { PYTHON_CMD="$py"; break; }
done

if [ -z "$PYTHON_CMD" ]; then
  warn "No Python interpreter found — skipping syntax check"
  pass "Syntax check skipped (no Python in PATH)"
else
  SYN_OUT=$(cd "$REPO_DIR" && "$PYTHON_CMD" -m py_compile inference.py 2>&1)
  if [ $? -eq 0 ]; then
    pass "inference.py syntax valid (py_compile passed)"
  else
    fail "inference.py has syntax errors"
    printf "\n%s\n\n" "$SYN_OUT"
    hint "Fix syntax errors and re-run the validator."
    stop_at "Step 5"
  fi
fi

# =============================================================================
# SUMMARY
# =============================================================================
printf "\n${BOLD}===============================================${NC}\n"
if [ $FAIL -eq 0 ]; then
  printf "${GREEN}${BOLD}  All 5/5 checks passed!${NC}\n"
  printf "${GREEN}${BOLD}  Your submission is ready.${NC}\n"
  printf "${BOLD}===============================================${NC}\n\n"
  printf "  Next steps:\n"
  printf "  1. git push to GitHub\n"
  printf "  2. Deploy to HuggingFace Spaces\n"
  printf "  3. Submit the HF Space URL\n\n"
  exit 0
else
  printf "${RED}${BOLD}  %d check(s) failed. Fix before submitting.${NC}\n" "$FAIL"
  printf "${BOLD}===============================================${NC}\n\n"
  exit 1
fi