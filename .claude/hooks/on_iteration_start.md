# Hook: On Iteration Start

Run this checklist at the start of every iteration before writing any code.

## Step 1 — Read state
- Read memory/current_iteration.md
- Confirm which iteration number you are on
- Confirm which module you are building

## Step 2 — Check what is done
- Read memory/what_is_done.md
- Confirm all previous iterations are marked complete
- If a previous iteration is not marked complete, stop and complete it first

## Step 3 — Check what failed before
- Read memory/what_failed.md
- If this module has a previous failure entry, read it fully
- Do NOT repeat the same approach that failed

## Step 4 — Load constraints
- Read memory/never_do.md
- Load all hard rules before writing a single line

## Step 5 — Load the skill
- Read the relevant skill file from skills/ for this iteration
- Follow it exactly — do not invent a different approach

## Step 6 — Confirm test file exists
- Check that the test file for this iteration exists in tests/
- If it does not exist, create it BEFORE writing the implementation
- Tests first. Always.
