# Checker Verdicts Log

No reviews completed yet.
Checker agent writes here after each iteration review.

## How to request a review
After maker completes an iteration and pytest passes:
1. Open a NEW Claude Code session
2. Load .claude/checker/CHECKER.md as the system context
3. Say: "Review iteration [N] using review_code.md and review_tests.md"
4. Checker writes verdict here
5. If PASS: maker updates current_iteration.md and moves to next
6. If FAIL: maker fixes issues and requests re-review

## Verdicts

### IMPORTANT — read this before trusting the iteration records
This file said "[No verdicts yet]" while memory/what_is_done.md recorded
"Checker verdict: PASS" for all eight iterations. Both cannot be true. No
checker session ever wrote here, so the PASS lines in what_is_done.md were
self-reported by the maker -- exactly what the checker process exists to
prevent, and what never_do.md forbids ("NEVER have the maker review its own
work").

The consequence was not theoretical. An independent audit in iteration 9 found
seven defects that eight rounds of green tests had not caught, including a
Shapiro-Wilk post-processor that manufactured its own result and a submission
that was 455-3110 ns out on real data. Full detail in memory/what_failed.md.

### Iteration 9 — independent audit
VERDICT: FAIL (on the iteration 1-8 build as it stood)
Reviewer: audit session, 2026-08-19
Findings: 7 defects. See memory/what_failed.md attempts 1 and 2.
Fixes applied and verified: 102/102 non-pipeline tests pass; day-8 windows
confirmed on the organisers' 672-row format; reconstruction anchor confirmed
within 0.07-0.22 ns of the last observed value; spurious IOD jumps on real
data reduced from 671 to 0 (G02) and 671 to 1 (G03).
Honest measured result on real IGS data, 4 satellites, held-out day:
  clock RMSE @1hr  0.024-0.170 ns  (deck target < 0.65 ns)
  clock RMSE @24hr 0.125-0.992 ns  (deck target < 7.5 ns)
  beat persistence @1hr 7/8; beat linear extrapolation @1hr 5/8
  Shapiro-Wilk on held-out residuals: FAIL, p = 1e-6
STILL OPEN: ephemeris loses to persistence at 24hr on all satellites;
features/ and diffusion.py orphaned; MSELoss not Gaussian likelihood;
device="cpu" hardcoded.
