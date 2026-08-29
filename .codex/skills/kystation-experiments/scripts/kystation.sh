#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${SSBENCH_REMOTE_HOST:-kystation}"
REMOTE_DIR="${SSBENCH_REMOTE_DIR:-/home/monkey/apps/ada-ssdataagent}"
REMOTE_UV="${SSBENCH_REMOTE_UV:-/home/monkey/.local/bin/uv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

usage() {
  printf '%s\n' \
    "Usage: kystation.sh status" \
    "       kystation.sh sync-code" \
    "       kystation.sh setup" \
    "       kystation.sh run -- <command> [args...]" \
    "       kystation.sh pull <remote-relative-path> [local-destination]" \
    "       kystation.sh sync-private"
}

require_clean_main() {
  local branch status
  branch="$(git -C "$LOCAL_ROOT" branch --show-current)"
  if [[ "$branch" != "main" ]]; then
    printf 'Refusing remote sync: local branch is %s, expected main.\n' "$branch" >&2
    exit 2
  fi

  status="$(git -C "$LOCAL_ROOT" status --porcelain --untracked-files=normal)"
  if [[ -n "$status" ]]; then
    printf '%s\n' "Refusing remote sync: the local working tree is not clean." >&2
    printf '%s\n' "$status" >&2
    exit 2
  fi
}

sync_code() {
  local local_sha remote_sha
  require_clean_main
  git -C "$LOCAL_ROOT" push origin main
  ssh "$REMOTE_HOST" git -C "$REMOTE_DIR" pull --ff-only origin main

  local_sha="$(git -C "$LOCAL_ROOT" rev-parse HEAD)"
  remote_sha="$(ssh "$REMOTE_HOST" git -C "$REMOTE_DIR" rev-parse HEAD)"
  if [[ "$local_sha" != "$remote_sha" ]]; then
    printf 'SHA mismatch: local=%s remote=%s\n' "$local_sha" "$remote_sha" >&2
    exit 3
  fi
  printf 'Code synchronized at %s\n' "$local_sha"
}

setup_remote() {
  local quoted_dir quoted_uv
  printf -v quoted_dir '%q' "$REMOTE_DIR"
  printf -v quoted_uv '%q' "$REMOTE_UV"
  ssh "$REMOTE_HOST" "cd $quoted_dir && $quoted_uv sync --frozen"
}

run_remote() {
  local quoted_dir quoted_uv remote_command arg
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  if [[ "$#" -eq 0 ]]; then
    usage >&2
    exit 2
  fi

  sync_code
  setup_remote

  printf -v quoted_dir '%q' "$REMOTE_DIR"
  printf -v quoted_uv '%q' "$REMOTE_UV"
  remote_command="$quoted_uv run"
  for arg in "$@"; do
    printf -v remote_command '%s %q' "$remote_command" "$arg"
  done
  ssh "$REMOTE_HOST" "cd $quoted_dir && $remote_command"
}

show_status() {
  local local_sha remote_sha
  local_sha="$(git -C "$LOCAL_ROOT" rev-parse HEAD)"
  remote_sha="$(ssh "$REMOTE_HOST" git -C "$REMOTE_DIR" rev-parse HEAD)"
  printf 'Local:  %s\nRemote: %s\n' "$local_sha" "$remote_sha"
  git -C "$LOCAL_ROOT" status --short --branch
  ssh "$REMOTE_HOST" git -C "$REMOTE_DIR" status --short --branch
  ssh "$REMOTE_HOST" du -sh "$REMOTE_DIR/data/real_data" "$REMOTE_DIR/runs" 2>/dev/null || true
}

pull_result() {
  local remote_path="${1:-}" local_destination="${2:-$LOCAL_ROOT/runs/remote/}"
  if [[ -z "$remote_path" || "$remote_path" == /* || "$remote_path" == *..* ]]; then
    printf '%s\n' "Remote result path must be a safe path relative to the remote repository." >&2
    exit 2
  fi
  mkdir -p "$local_destination"
  scp -r "$REMOTE_HOST:$REMOTE_DIR/$remote_path" "$local_destination"
}

sync_private() {
  if [[ ! -f "$LOCAL_ROOT/.env" || ! -d "$LOCAL_ROOT/data/real_data" ]]; then
    printf '%s\n' "Expected local .env and data/real_data/ before private sync." >&2
    exit 2
  fi
  ssh "$REMOTE_HOST" mkdir -p "$REMOTE_DIR/data"
  scp "$LOCAL_ROOT/.env" "$REMOTE_HOST:$REMOTE_DIR/.env"
  ssh "$REMOTE_HOST" chmod 600 "$REMOTE_DIR/.env"
  scp -r "$LOCAL_ROOT/data/real_data" "$REMOTE_HOST:$REMOTE_DIR/data/"
  printf '%s\n' "Private environment and source data synchronized."
}

case "${1:-}" in
  status)
    show_status
    ;;
  sync-code)
    sync_code
    ;;
  setup)
    setup_remote
    ;;
  run)
    shift
    run_remote "$@"
    ;;
  pull)
    shift
    pull_result "$@"
    ;;
  sync-private)
    sync_private
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
