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
4. Run the experiment with `kystation.sh run`. It requires a clean local tree, pushes `main`, pulls it on the server with `--ff-only`, verifies matching SHAs, runs `uv sync --frozen`, and then executes the requested command through remote `uv run`.
5. Check the remote exit status and expected artifact paths. Experiment outputs normally remain under remote `runs/`.
6. Pull back only the specific result directory needed for local inspection. Do not copy the complete remote `runs/` tree by default.

Example:

```bash
bash .codex/skills/kystation-experiments/scripts/kystation.sh run -- \
  python scripts/simulate.py --dataset cfps --method direct --n 5
```

Useful operations:

```bash
bash .codex/skills/kystation-experiments/scripts/kystation.sh status
bash .codex/skills/kystation-experiments/scripts/kystation.sh sync-code
bash .codex/skills/kystation-experiments/scripts/kystation.sh setup
bash .codex/skills/kystation-experiments/scripts/kystation.sh pull \
  runs/cfps/direct/<run_id> runs/remote/
```

## Private files

Use `sync-private` only for initial deployment or when the user explicitly asks to refresh `.env` or source data. It uploads local `.env` and `data/real_data/` through SCP and sets the remote `.env` mode to `600`.

Never display secret values, add private inputs to Git, force-push, reset or clean the remote checkout, or overwrite a divergent remote branch. Stop on dirty or divergent code state and report the exact condition.
