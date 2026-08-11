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
