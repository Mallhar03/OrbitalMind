# Checkpoint System

Claude Code writes a checkpoint after completing each function.
If a session ends mid-iteration, the next session reads these
checkpoints to know exactly where to resume.

## Format
One file per iteration: checkpoint_iter_N.md
Updated after every function written and verified.

## Rule
Never delete checkpoint files.
Never mark a checkpoint complete without running the function's unit test.
