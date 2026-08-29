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
    "       kystation.sh run -- <short-command> [args...]" \
    "       kystation.sh start <job-name> -- <long-command> [args...]" \
    "       kystation.sh job-status <job-name> [log-lines]" \
    "       kystation.sh jobs" \
    "       kystation.sh pull <remote-relative-path> [local-destination]" \
    "       kystation.sh sync-private"
}

validate_job_name() {
  local job_name="${1:-}"
  if [[ -z "$job_name" || ! "$job_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf '%s\n' "Job name must use letters, digits, dots, underscores, or hyphens." >&2
    exit 2
  fi
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
  local local_sha remote_sha sync_tmp bundle_path remote_bundle
  require_clean_main
  git -C "$LOCAL_ROOT" push origin main

  local_sha="$(git -C "$LOCAL_ROOT" rev-parse HEAD)"
  remote_sha="$(ssh "$REMOTE_HOST" git -C "$REMOTE_DIR" rev-parse HEAD)"
  if [[ "$local_sha" != "$remote_sha" ]]; then
    printf '%s\n' "Remote SHA differs; synchronizing a verified git bundle." >&2
    sync_tmp="$(mktemp -d "${TMPDIR:-/tmp}/ssbench-sync.XXXXXX")"
    bundle_path="$sync_tmp/main.bundle"
    remote_bundle="/tmp/ssbench-main-$local_sha.bundle"
    git -C "$LOCAL_ROOT" bundle create "$bundle_path" main
    scp "$bundle_path" "$REMOTE_HOST:$remote_bundle"
    ssh "$REMOTE_HOST" \
      "git -C '$REMOTE_DIR' fetch '$remote_bundle' main && git -C '$REMOTE_DIR' merge --ff-only FETCH_HEAD"
    rm -rf "$sync_tmp"
  fi

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

start_remote() {
  local job_name="${1:-}" session jobs_dir log_path exit_path
  local quoted_dir quoted_uv quoted_log quoted_exit quoted_session
  local remote_command job_command quoted_job_command arg
  validate_job_name "$job_name"
  shift
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  if [[ "$#" -eq 0 ]]; then
    usage >&2
    exit 2
  fi

  sync_code
  setup_remote

  session="ssb-$job_name"
  jobs_dir="$REMOTE_DIR/runs/remote_jobs"
  log_path="$jobs_dir/$job_name.log"
  exit_path="$jobs_dir/$job_name.exit"

  if ssh "$REMOTE_HOST" tmux has-session -t "$session" 2>/dev/null \
      || ssh "$REMOTE_HOST" test -e "$log_path" \
      || ssh "$REMOTE_HOST" test -e "$exit_path"; then
    printf 'Refusing to reuse job name %s; its session or artifacts already exist.\n' "$job_name" >&2
    exit 4
  fi

  ssh "$REMOTE_HOST" mkdir -p "$jobs_dir"
  printf -v quoted_dir '%q' "$REMOTE_DIR"
  printf -v quoted_uv '%q' "$REMOTE_UV"
  printf -v quoted_log '%q' "$log_path"
  printf -v quoted_exit '%q' "$exit_path"
  printf -v quoted_session '%q' "$session"

  remote_command="$quoted_uv run"
  for arg in "$@"; do
    printf -v remote_command '%s %q' "$remote_command" "$arg"
  done
  job_command="cd $quoted_dir && $remote_command > $quoted_log 2>&1; status=\$?; printf '%s\\n' \"\$status\" > $quoted_exit; exit \"\$status\""
  printf -v quoted_job_command '%q' "$job_command"

  ssh "$REMOTE_HOST" "tmux new-session -d -s $quoted_session $quoted_job_command"
  if ! ssh "$REMOTE_HOST" tmux has-session -t "$session" 2>/dev/null \
      && ! ssh "$REMOTE_HOST" test -f "$exit_path"; then
    printf 'tmux job %s did not start and produced no exit record.\n' "$job_name" >&2
    exit 5
  fi

  printf 'Started tmux session %s\nLog: %s\nStatus: kystation.sh job-status %s\n' \
    "$session" "$log_path" "$job_name"
}

show_job_status() {
  local job_name="${1:-}" lines="${2:-40}" session log_path exit_path state exit_code
  validate_job_name "$job_name"
  if [[ ! "$lines" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' "Log line count must be a positive integer." >&2
    exit 2
  fi

  session="ssb-$job_name"
  log_path="$REMOTE_DIR/runs/remote_jobs/$job_name.log"
  exit_path="$REMOTE_DIR/runs/remote_jobs/$job_name.exit"

  if ssh "$REMOTE_HOST" tmux has-session -t "$session" 2>/dev/null; then
    state="running"
  elif ssh "$REMOTE_HOST" test -f "$exit_path"; then
    exit_code="$(ssh "$REMOTE_HOST" cat "$exit_path")"
    state="finished (exit=$exit_code)"
  else
    state="not found"
  fi
  printf 'Job: %s\nSession: %s\nState: %s\n' "$job_name" "$session" "$state"
  if ssh "$REMOTE_HOST" test -f "$log_path"; then
    printf '%s\n' "--- log tail ---"
    ssh "$REMOTE_HOST" tail -n "$lines" "$log_path"
  fi
}

list_jobs() {
  printf '%s\n' "--- active tmux sessions ---"
  ssh "$REMOTE_HOST" "tmux list-sessions -F '#{session_name} windows=#{session_windows} attached=#{session_attached}' 2>/dev/null || true"
  printf '%s\n' "--- recorded exits ---"
  ssh "$REMOTE_HOST" "find '$REMOTE_DIR/runs/remote_jobs' -maxdepth 1 -type f -name '*.exit' -print 2>/dev/null | sort || true"
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
  start)
    shift
    start_remote "$@"
    ;;
  job-status)
    shift
    show_job_status "$@"
    ;;
  jobs)
    list_jobs
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
