# Hard Constraints — Never Violate These

These rules exist because we already thought through them.
Do not re-argue them. Do not find exceptions. Just follow them.

## Data rules
- NEVER use sequence_length less than 96 steps (= 24 hours of history)
- NEVER mix GEO and MEO satellites in the same model training run
- NEVER skip IOD jump correction in preprocessing
- NEVER feed raw unnormalised data into a neural network

## Model rules
- NEVER call an iteration complete without running pytest
- NEVER move to the next iteration if any test in tests/ is failing
- NEVER use fewer than 5 training days (train on days 1-5, validate on 6-7)
- NEVER hardcode a file path — always use relative paths or config

## Code rules
- NEVER write implementation before writing the test
- NEVER modify a completed module without re-running its test
- NEVER use random operations without seed=42
- NEVER skip the docstring on any function

## Memory rules
- NEVER delete or overwrite entries in what_failed.md
- NEVER mark an iteration complete in current_iteration.md
  without first running the full pytest suite
- NEVER start a new session without reading current_iteration.md first

## Pipeline rules
- NEVER let the pipeline have an interactive input() call
- NEVER let any module import from a module in a later iteration
  (preprocessing cannot import from models, etc.)

## Checker rules
- NEVER mark iteration complete without checker PASS verdict
- NEVER skip the checker because "tests already passed"
- NEVER have the maker review its own work
- NEVER request re-review without fixing ALL failures listed in verdict
- NEVER modify checker files — they are read-only for the maker

## Session continuity rules
- NEVER start a session without running python scripts/resume.py first
- NEVER assume you know where you left off — always read the resume output
- NEVER rewrite a function already marked [x] in a checkpoint file
- NEVER write two functions without a checkpoint between them
- NEVER let a session end without committing at minimum a WIP commit
- NEVER ignore low-context warning signs — checkpoint and stop cleanly
