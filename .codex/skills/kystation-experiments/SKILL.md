---
name: kystation-experiments
description: Run, resume, evaluate, or inspect Ada SSDataBench experiments on the kystation server while keeping the local repository as the code-authoring source. Use for experiment execution in this project; do not run data or model experiments on the local Mac.
---

# Kystation Experiments

Keep code authoring local and execute experiments on the maintained server copy.

## Fixed environment

- SSH host: `kystation`
- Remote repository: `/home/monkey/apps/ada-ssdataagent`
- Remote `uv`: `/home/monkey/.local/bin/uv`
- Code source of truth: committed `main` from `git@github.com:houx15/ada-ssdataagent.git`
- Private inputs: remote `.env` and `data/real_data/`; both stay outside Git

Use [scripts/kystation.sh](scripts/kystation.sh) for the repeated operations.

## Required workflow

1. Edit code locally and perform only lightweight static checks locally.
2. Before an experiment, inspect Git status. Remote execution uses committed `main`; never silently run stale code or copy an uncommitted working tree over the server checkout.
3. If the task authorizes code changes, commit and push them. If committing is not authorized and the relevant code is dirty, stop and ask the user how to proceed.
4. Use `kystation.sh run` only for short smoke checks. Start every formal experiment or other long-running task with `kystation.sh start <job-name> -- ...`; it runs in a detached `tmux` session so the job survives SSH or Codex disconnection. Both paths require a clean local tree, push `main`, pull it on the server with `--ff-only`, verify matching SHAs, run `uv sync --frozen`, and execute through remote `uv run`.
5. For a long task, record the tmux session name and inspect it with `job-status`. Confirm the saved exit code and expected artifact paths before declaring completion. Experiment outputs normally remain under remote `runs/`.
6. Pull back only the specific result directory needed for local inspection. Do not copy the complete remote `runs/` tree by default.

Short smoke-check example:

```bash
bash .codex/skills/kystation-experiments/scripts/kystation.sh run -- \
  python scripts/simulate.py --dataset cfps --method direct --n 5
```

Long experiment example:

```bash
bash .codex/skills/kystation-experiments/scripts/kystation.sh start cfps-direct-001 -- \
  python scripts/simulate.py --dataset cfps --method direct --n 1000

bash .codex/skills/kystation-experiments/scripts/kystation.sh job-status cfps-direct-001
```

Long-task sessions use the `ssb-<job-name>` tmux namespace. Logs and exit codes are stored under remote `runs/remote_jobs/`. Do not reuse an existing job name; use a new descriptive name so prior logs remain intact.

Useful operations:

```bash
bash .codex/skills/kystation-experiments/scripts/kystation.sh status
bash .codex/skills/kystation-experiments/scripts/kystation.sh jobs
bash .codex/skills/kystation-experiments/scripts/kystation.sh sync-code
bash .codex/skills/kystation-experiments/scripts/kystation.sh setup
bash .codex/skills/kystation-experiments/scripts/kystation.sh pull \
  runs/cfps/direct/<run_id> runs/remote/
```

## Private files

Use `sync-private` only for initial deployment or when the user explicitly asks to refresh `.env` or source data. It uploads local `.env` and `data/real_data/` through SCP and sets the remote `.env` mode to `600`.

Never display secret values, add private inputs to Git, force-push, reset or clean the remote checkout, or overwrite a divergent remote branch. Stop on dirty or divergent code state and report the exact condition.
