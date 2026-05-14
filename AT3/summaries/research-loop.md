# research-loop.md — how to run a long unattended experiment loop

Instructions for orchestrating a multi-hour sequence of ML experiments
unattended, using a cron heartbeat plus harness-tracked background tasks.
Specific to this project's Python-notebook layout, but the pattern
generalises.

## Loop architecture

The user starts the loop once with `/loop 5m <prompt>` and walks away. From
that single invocation, three asynchronous signals drive everything:

1. **The cron heartbeat** fires every 5 minutes and re-enters the prompt.
   Treat each firing as a safety net, not a primary trigger: check whether
   the current background task is still running, tail its progress log,
   decide on action, yield.
2. **The harness completion notification** (`<task-notification>`) arrives
   the moment a background process exits. This is the *primary* wake
   signal — "run finished, read the result, launch the next thing".
3. **Direct tool output** when you actively check between events.

The cron firing during work is harmless: just look at progress and yield.
The architecture works because the cron is the safety net, not the driver.

## Launching a script

The project's experiments are Python files in jupytext "percent" format
(`# %% [markdown]` and `# %%` cell markers). Run them as plain Python — no
need for nbconvert during the loop. nbconvert produces an .ipynb with
output cells, which matters only for final submission, not for the loop.

Canonical launch:

```python
Bash(
    command="HSA_OVERRIDE_GFX_VERSION=11.5.1 "
            ".venv/bin/python -u phase2_eXX_*.py",
    run_in_background=True,
    timeout=1800000,
)
```

Three rules that prevent the most common failure mode:

1. **Never combine `&` with `run_in_background: True`.** Pick one. The
   harness handles backgrounding when `run_in_background: True`; adding
   `&` produces a Bash command that exits with code 1 while the Python
   process continues running, and the harness reports the run as failed
   when it actually succeeded.
2. **Don't write PID files yourself.** The harness tracks the process.
   Note the task ID it returns; you'll use that to read the output and
   identify the completion notification.
3. **Always pass `-u`** (unbuffered stdout). Without it, `print()` and
   line-logged progress lag in the per-task output file and `tail`
   shows stale state.

After launching, the harness returns a task ID and an output path like
`/tmp/claude-*/tasks/<task-id>.output`. Tail that file during the next
cron firings to read live progress.

## Working directory and environment

These project-specific requirements matter on every launch:

- **CWD must be the project root** (`assignment3_notebook/`). Every notebook
  starts with `sys.path.insert(0, str(Path("output").resolve()))`, which
  resolves relative to CWD. Running from anywhere else breaks the
  `exp_runner` import.
- **`HSA_OVERRIDE_GFX_VERSION=11.5.1`** is required for the AMD iGPU.
  Export it inline on every launch; don't rely on the shell rc.
- **Use the absolute interpreter path** (`.venv/bin/python`). The Bash tool
  restarts the shell between calls, so a `source .venv/bin/activate` from
  an earlier turn doesn't carry over.
- A `(null): No such file or directory` line appears in every script's
  stdout, from a venv-startup oddity. It's harmless. Ignore unless
  debugging.

## The artifact contract

Every experiment script writes to `output/phase2_results/<run_name>/`:

- `progress.log` — line-buffered training/eval progress; tail this for
  live status without touching the harness output file.
- `metrics.json` — final BLEU/CIDEr/etc numbers. **This is the contract
  that makes the loop scalable.** Any script that emits `metrics.json` is
  automatically picked up by downstream tools and can be cited in the
  running summary.
- `history.csv` — per-epoch train/val curves for plots.
- `val_predictions.csv` / `test_predictions.csv` — raw per-image captions
  for qualitative inspection.
- `best.pt` — the trained checkpoint.

To add a new experiment: copy an existing `.py`, edit the config and
hyperparameter cells, syntax-check, launch. Everything downstream is free
because of the artifact contract.

## GPU is the serialising resource

A single GPU = strict serial queue. Never run two GPU-using tasks
concurrently — the memory headroom doesn't allow it and the time saved
by interleaving is eaten by thrashing.

Non-GPU work *can* run in parallel and should: edit the next notebook,
update the summary doc, syntax-check the next script, prefetch
pretrained weights (HuggingFace cache priming). Dispatch all of these
from the same conversation turn as the launch, before yielding. This
keeps the GPU utilised while you're "free" between firings.

## Per-cycle rhythm

On every cron firing or completion notification:

1. **If a background task is running**: tail its output file, note
   progress, decide whether anything needs doing. If not, yield.
2. **If a task just completed**: read `metrics.json`, append a row to
   the running summary table, write a paragraph of analysis, then
   launch the next item from the queue.
3. **If the queue has items waiting**: launch the next one. Update
   the task tracker (`TaskUpdate` to `in_progress`).
4. **If the queue is empty**: do *not* declare the loop complete. See
   "When the queue runs dry" below.

## Task tracking and the running summary

Maintain two persistence layers throughout the session:

- **`TaskCreate` / `TaskUpdate`** for in-conversation orchestration state.
  Create one task per planned experiment up front, mark `in_progress` on
  launch, `completed` on result-in-hand. Add new tasks for follow-ups
  that emerge during the loop. Delete tasks that become invalid
  mid-session (e.g. a planned follow-up that depends on a result that
  came back negative).
- **A markdown summary doc** (e.g. `phase3_summary.md`) for results that
  survive the session. Update it after every completed experiment with:
  one row in the headline table, a per-experiment write-up paragraph,
  and any new cross-cutting lesson.

**Update the doc as results land, never batch at the end.** If the
session terminates unexpectedly, the doc should reflect the actual
state of completed work.

## Mode discipline between launches

Three modes happen in this loop. Recognise which one you're in:

- **Plan / write** — designing the next experiment, syntax-checking,
  updating ensemble configs to incorporate just-finished members.
- **Launch / monitor** — kicking off a background task, checking
  progress, yielding.
- **Document / synthesise** — updating the summary, re-ranking the
  headline table, writing per-experiment paragraphs.

Plan and document are non-GPU; they run in parallel with monitor.
Efficient rhythm: launch (monitor) → yield, on next cron fire write
docs (document) while the GPU works → next yield, plan the next launch
(plan) while the GPU works → next yield, launch on completion.

Hour-mark checkpoint: roughly every hour, re-read the summary doc you
are writing. Catches creeping inconsistencies (duplicate rank numbers,
stale "still to come" sections) before they compound.

## When the queue runs dry

**"Queue empty" is a transition state, not a terminal state.** When the
linear queue of planned experiments dries up, do not stop the loop. Pivot
to one of the lateral moves below:

- **Read predictions.** Open the current best model's `val_predictions.csv`.
  Look for systematic failure patterns: caption length distribution,
  vocabulary gaps, image types that consistently fail. Qualitative analysis
  almost always generates the next targeted experiment.
- **Re-read the literature notes.** If the session began with a research
  pass, return to it. Find techniques that were named but never tested
  *on the current best model*. Common case: a trick was tested in an
  earlier phase on a weaker baseline and didn't move the needle, but
  was never re-evaluated on the new winner.
- **Re-read the prior summary's "did not sweep" notes.** Many summaries
  end with an explicit list of untouched hyperparameter axes (tokenizer,
  weight decay, dropout, LR schedule). When the obvious queue is empty,
  this list is the next thing to read.
- **Different research directions.** Knowledge distillation, test-time
  augmentation, larger encoder variants — none require new infrastructure,
  just a new script.
- **Ask the user.** If the above genuinely yield nothing new, the loop
  *is* done and a confirming question is appropriate before stopping.

The rule: **prefer lateral exploration over declaring complete**. Declaring
complete after the linear queue empties is the failure mode this section
exists to prevent.

## Launch checklist

Before every launch:

1. **Syntax-check** the script:
   `python -c "import ast; ast.parse(open('foo.py').read())"`
   Saves the awkward "experiment failed at line N" 30 seconds in.
2. **Verify upstream artifacts exist**: feature caches, parent checkpoints
   (for warm-starts), corpus DF files. Catch missing-input errors before
   launch, not 30 seconds into a 20-min run.
3. **Launch** with `Bash(run_in_background=True)`, no `&`, no PID files.
   Note the task ID returned.
4. **Update the task tracker** to `in_progress`.
5. **Yield** the turn. The harness will wake you on completion or cron.

Between launches, do non-GPU work (docs, next-script editing). Don't
poll the GPU task on a short interval — the harness notification is the
primary wake.

## Stopping the loop

The loop's cron is in-memory and dies with the session. Two ways to stop:

- **`CronDelete <id>`** to halt firings immediately.
- **Let the session end** — in-memory crons die with the process.

Before stopping, confirm the work is actually done in the
lateral-exploration sense (see "When the queue runs dry"), not just the
linear-queue sense. Don't leave a cron firing on no-op heartbeats once
the work is genuinely complete — each firing costs tokens, and over
hours the total is non-trivial. But also don't stop early.

## Recap of the rules

In one place, the rules to encode for next time:

1. Cron is the safety net. Harness notifications are the primary wake.
2. `run_in_background: True` *or* `&`, never both.
3. `-u` for unbuffered output; CWD = project root; absolute venv path;
   project-specific env vars set inline on every launch.
4. Every experiment honours the `metrics.json` artifact contract.
5. One GPU = one job. Non-GPU work parallelises in the same turn.
6. Update the summary doc as results land. Don't batch.
7. Three modes (plan / monitor / document); rotate them across cron
   firings to keep the GPU and yourself both utilised.
8. Queue empty ≠ loop done. Pivot laterally: read predictions, re-read
   literature, re-read prior "did not sweep" lists, ask the user.
9. Stop the cron when truly done; don't burn tokens on no-op heartbeats.
