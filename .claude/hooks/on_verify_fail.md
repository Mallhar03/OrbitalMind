# Hook: On Verify Fail

When pytest fails or RMSE/Shapiro-Wilk conditions are not met, do this exactly.

## Step 1 — Do not guess
- Read the full error message
- Identify the single root cause
- Do not fix multiple things at once

## Step 2 — Log the failure
- Open memory/what_failed.md
- Add a new entry with:
  - Iteration number
  - What was attempted
  - Exact error message
  - Root cause identified

## Step 3 — Fix only the root cause
- Write the smallest possible change that addresses the root cause
- Do not refactor anything else while fixing

## Step 4 — Re-run verify immediately
- Run pytest tests/ again
- Run the RMSE check again
- Do not move on until both pass

## Step 5 — If same error appears twice
- Stop
- Read memory/decisions.md
- Check if a decision was already made about this
- If no prior decision exists, try a fundamentally different approach
- Log the new approach in memory/decisions.md before trying it
