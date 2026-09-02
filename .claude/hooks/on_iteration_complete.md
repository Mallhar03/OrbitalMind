# Hook: On Iteration Complete

When verify passes for an iteration, do these steps before closing the session.

## Step 1 — Update current iteration
- Open memory/current_iteration.md
- Mark current iteration as COMPLETE
- Set next iteration number as IN PROGRESS
- Write what the next action is

## Step 2 — Log completion
- Open memory/what_is_done.md
- Add completed module with:
  - Iteration number
  - Module name and file path
  - What the test checks
  - RMSE or metric value achieved

## Step 3 — Run full test suite
- Run pytest tests/ for ALL iterations so far
- Confirm nothing previously passing is now broken
- If anything broke, fix it before proceeding

## Step 4 — Commit to git
- git add .
- git commit -m "Iteration [N] complete: [module name] — [metric achieved]"

## Step 5 — Report
- Print a one-paragraph summary of what was built
- Print current RMSE at all horizons achieved so far
- Print what iteration N+1 will build

## Step 6 — Deck compliance audit (MANDATORY, do not skip)

An iteration is not finished when pytest passes. It is finished when the work
still matches what latest.pdf promises the judges.

- Invoke the `audit-stage-lead` agent, naming the stage that just completed.
- It runs ONLY that stage's auditor (S1 ingest ... S6 postprocess). Never a
  whole-project sweep: findings about later stages are worthless while an
  earlier stage is unproven.
- A FAIL does NOT stop the work and does NOT go back to Mallhar for a decision.
  Fix what the auditor found, re-run it, and repeat until it signals green.
  Record every round — all corrections go into the single final report.
- Never clear a finding by weakening the measurement. See the integrity charter
  in .claude/audit/AUDIT_CONTRACT.md.
- A FAIL blocks every later stage. Fix the earliest failing gate first.
- Read its BOTTOM LINE and every P0 and P1 finding.
- A P0 finding blocks the iteration. P0 means the code or the report is telling
  someone something untrue. Fix it before starting iteration N+1.
- P1 findings (promised to judges, not built) go into memory/what_is_done.md as
  known gaps with the claim ID, so the next session inherits them.
- Do NOT let the audit edit code. It reports; fixing is a separate decision.

The report lands in memory/ppt_audit/S<n>-LATEST.md, with dated copies alongside
so drift over time is visible. Stage IDs and their code mapping are in
.claude/audit/STAGES.md.

## Step 7 — STOP for Mallhar's review

The iteration is not closed and the next phase does not begin until the auditor
has signalled green AND Mallhar has reviewed the audit report. Hand him the four-part report, say what the next
phase would be, and wait. Do not start the next iteration on your own judgement.
