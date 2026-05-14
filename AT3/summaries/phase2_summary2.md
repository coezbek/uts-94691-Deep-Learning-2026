# research-loop — playbook for multi-hour unattended experiment loops

How to orchestrate a sequence of Python-notebook experiments unattended
via a cron heartbeat plus harness-tracked background tasks.

## Headline

* One `/loop 5m <prompt>` invocation drives the whole session; the cron is
  a 5-min **safety net**, not the primary trigger.
* The **primary wake signal** is the harness's `<task-notification>` when
  a background task exits.
* One GPU = strict serial queue. Non-GPU work (edits, docs, prefetches)
  runs in parallel inside the same turn.
* Every experiment honours an artifact contract: `metrics.json`,
  `progress.log`, `history.csv`, `*_predictions.csv`, `best.pt` under
  `output/.../<run_name>/`.
* **Queue empty ≠ loop done.** When the linear queue dries up, pivot
  laterally before stopping.

## The three signals

| Signal | Cadence | Use |
|---|---|---|
| Cron heartbeat | every 5 min | safety net — check progress, decide, yield |
| Harness completion notification | per background task | primary wake — read `metrics.json`, launch next |
| Direct tool output | on demand | tail logs between events |

Cron firing during work is harmless: look at progress, yield.

## Launch recipe

Experiments live as Python files in jupytext "percent" format. Run them as
plain Python for the loop. nbconvert produces an `.ipynb` with output
cells; that matters only for final submission, not for execution.

```
Bash(
  command="HSA_OVERRIDE_GFX_VERSION=11.5.1 .venv/bin/python -u phase2_eXX.py",
  run_in_background=True,
  timeout=1800000,
)
```

Three rules:

1. **Never combine `&` with `run_in_background=True`.** The Bash command
   exits with code 1 while the Python process continues; the harness
   reports the run as failed when it succeeded.
2. **Don't write PID files.** The harness tracks the process; use the
   returned task ID to tail output and identify the completion event.
3. **Always `-u`** (unbuffered stdout). Without it, `tail` shows stale
   state.

Project-specific requirements on every launch:

* **CWD = project root** (`assignment3_notebook/`). Scripts do
  `sys.path.insert(0, "output")` relative to CWD.
* **`HSA_OVERRIDE_GFX_VERSION=11.5.1`** inline; don't rely on shell rc.
* **Absolute interpreter path** (`.venv/bin/python`). The Bash tool
  restarts the shell between calls — earlier `source activate` doesn't
  persist.
* A harmless `(null): No such file or directory` line appears in every
  script's stdout from a venv-startup oddity. Ignore it.

## The artifact contract

Every experiment writes to `output/phase2_results/<run_name>/`:

* `progress.log` — line-buffered; tail this for live status.
* **`metrics.json`** — final numbers. The scaling primitive: anything
  that emits this file is automatically picked up by downstream tools.
* `history.csv` — per-epoch curves.
* `val_predictions.csv` / `test_predictions.csv` — raw captions for
  qualitative inspection.
* `best.pt` — checkpoint.

Adding a new experiment is "copy an existing `.py`, edit the config,
syntax-check, launch". Everything downstream is free.

## Per-cycle rhythm

On every cron firing or completion notification:

1. Task running? Tail its output; yield if nothing to do.
2. Task just completed? Read `metrics.json`, append a row to the
   summary table, write a short analysis paragraph, launch the next item.
3. Queue has items? Launch next. `TaskUpdate` to `in_progress`.
4. Queue empty? **Do not declare done.** See "Queue runs dry" below.

## Three modes — rotate them across firings

| Mode | Uses GPU? | Examples |
|---|---|---|
| Plan / write | no | design next experiment, syntax-check, update ensemble configs |
| Launch / monitor | yes | start background task, tail log, yield |
| Document / synthesise | no | update summary, re-rank table, write per-experiment paragraph |

Plan and document parallelise with monitor. Efficient rhythm: launch →
yield, on next firing write docs while GPU works → next yield, plan
next launch while GPU works → next yield, launch on completion.

**Hour-mark checkpoint.** Every ~hour, re-read your own summary doc.
Catches duplicate rank numbers, stale "still to come" sections, and
similar drift before it compounds.

## Two persistence layers

* **`TaskCreate` / `TaskUpdate`** for in-conversation orchestration state.
  Create one task per planned experiment up front. Mark `in_progress` on
  launch, `completed` on result-in-hand. Add new tasks for follow-ups
  that emerge mid-loop. Delete tasks that became invalid.
* **A markdown summary doc** (e.g. `phase3_summary.md`) for results that
  survive the session. **Update as results land, never batch at the end.**
  If the session terminates unexpectedly the doc should reflect actual
  state.

## When the queue runs dry

"Queue empty" is a transition state, not a terminal state. Before
stopping, do at least one of:

* **Read predictions.** Open the current best model's
  `val_predictions.csv`. Length distribution, vocabulary gaps,
  systematic image-type failures. Qualitative analysis almost always
  generates the next targeted experiment.
* **Re-read literature notes** from earlier in the session. Find
  techniques named but never tested *on the current best model* —
  a common gap is "tried in an earlier phase on a weaker baseline,
  never re-evaluated on the new winner".
* **Re-read the prior summary's "did not sweep" list.** Most summaries
  end with explicit untouched axes (tokenizer, weight decay, dropout,
  LR schedule). That list *is* the queue when the obvious one empties.
* **Try a different research direction**: distillation, test-time
  augmentation, larger encoder variants. Usually no new infrastructure,
  just a new script.
* **Ask the user.** If the above genuinely yield nothing, the loop is
  done — confirm with a focused question before stopping.

Failure mode this section exists to prevent: declaring complete when
only the linear queue is empty, while obvious lateral moves remain.

## Pre-launch checklist

1. **Syntax-check** the script:
   `python -c "import ast; ast.parse(open('foo.py').read())"`.
   Saves a 30-second-in failure.
2. **Verify upstream artifacts**: feature caches, parent checkpoints
   (for warm-starts), corpus DF files. Catch missing-input errors
   before launch, not 30 seconds into a 20-min run.
3. **Launch** with `Bash(run_in_background=True)`. No `&`, no PID files.
   Note the task ID.
4. **`TaskUpdate`** the planned experiment to `in_progress`.
5. **Yield** the turn. Harness wakes you on completion or cron.

## Stopping the loop

The cron is in-memory and dies with the session. Two ways:

* `CronDelete <id>` to halt firings immediately.
* Let the session end — in-memory crons die with the process.

Before stopping, confirm the work is actually done in the
lateral-exploration sense (above), not just the linear-queue sense. Each
no-op heartbeat costs tokens; over hours the total is non-trivial.

## Rules to encode

1. Cron is the safety net. Harness notifications are the primary wake.
2. `run_in_background=True` *or* `&`, never both.
3. `-u`, CWD = project root, absolute venv path, env vars inline.
4. Every experiment writes `metrics.json` (the artifact contract).
5. One GPU = one job. Non-GPU work parallelises in the same turn.
6. Update the summary as results land. Don't batch.
7. Rotate plan / monitor / document across firings.
8. Queue empty ≠ loop done. Pivot: predictions, literature, "did not
   sweep" lists, ask the user.
9. Stop the cron once truly done; don't burn tokens on no-op heartbeats.
