# Checkpoint: Iteration 9
Last updated: 2026-08-19 22:55:41
Last safe point: audit_and_rebuild — complete

## Completed functions (tested and verified)
- [x] audit_and_rebuild

## Notes
Fixed 7 defects: circular NF, reconstruction anchor, RMSE measured in filtered space, hardcoded 480/576/672 windows, silent zeros, fixed IOD threshold, IOD coordinate-frame mismatch. Added splits.py + paths.py. 102 tests pass.

## Resume instruction
All registered functions complete — run make verify-9
