# Failed Attempts Log

Nothing failed yet. Update this file immediately when a verify step fails.

## Format for each entry:
---
Iteration: [N]
Attempt: [attempt number within this iteration]
What was tried: [description]
Error: [exact error message]
Root cause: [what actually caused it]
Fix applied: [what was changed]
Outcome: [did the fix work?]
---

## IMPORTANT
If the same error appears in two consecutive entries for the same iteration,
stop and read memory/decisions.md before trying again.

---
Iteration: 9 (audit of the "complete" v1 build against the proposal deck)
Attempt: 1
What was tried: Verifying the iteration-8 build against latest.pdf (SHIH26-TID-500)
                before submission, and checking whether it reproduces on other
                machines and other datasets.
Error: No exception. Every test passed and every output file was produced. The
       failure was silent, which is why eight iterations of green checkmarks
       did not catch it.
Root cause: Five independent defects, all invisible to the existing tests.

  1. Circular Normalizing Flow. normalizing_flow.py computed
       _correction = (z_gaussian * res_std + res_mean) - residuals
     from the validation ground truth, so (actual - corrected) was BY
     CONSTRUCTION a vector of exact normal quantiles. Shapiro-Wilk returned
     p ~= 0.9999 for every possible input. Verified by feeding residuals that
     were half constant, half exponential (raw Shapiro p = 0.0): the
     post-processor still reported p = 0.9999. The metric measured nothing.
     That same ground-truth-derived vector was then subtracted from the day-8
     forecast, injecting day 6's errors into the prediction.

  2. Wrong reconstruction anchor. run_pipeline passed cleaned[0] -- the first
     sample of the whole record -- as the anchor for a day-8 forecast, so
     every prediction carried the entire drift accumulated since day 1.
     Measured original-scale RMSE on real IGS data: G01 455 ns, G02 3110 ns,
     G05 738 ns, against a persistence baseline of 47 / 280 / 72 ns. The
     ensemble was six to eleven times worse than carrying the last value
     forward. Target in the deck is 0.65 ns at 1hr.

  3. RMSE measured in the wrong space. The evaluation report scored
     predictions against `combined` (trend + periodic of the DIFFERENCED
     signal, noise IMF discarded), not against ClockError_ns in nanoseconds.
     For GPS the differenced clock error is a near-constant ramp, so the
     models trivially predicted a constant and 22 of 31 satellites scored
     exactly 0.000000 at every horizon. Those zeros read as a triumph and
     were in fact the tell.

  4. Hardcoded windows 480/576/672. These only ever matched the 768-row
     (8-day) output of our own synthetic generator. The deck specifies the
     organisers' format as ~672 rows per satellite = 7 days, where [576:672]
     falls INSIDE the file: the pipeline would have re-predicted day 7 and
     labelled it day 8, with 95 evaluation points instead of 96. On the real
     14-day record it silently discarded 671 of 1343 rows.

  5. Silent zeros on failure. `except Exception` wrote np.zeros(96) into the
     submission and continued, so a total failure produced a complete-looking
     CSV. Combined with defect 3, nothing in the test suite could tell a
     working satellite from a broken one.

     Also found: orbit type was inferred as `startswith("GEO")` while an
     OrbitType column sat unused in the CSV, so every real constellation ID
     (G01..G32) was branched as MEO -- violating the GEO/MEO separation rule
     in this very directory's never_do.md. And requirements.txt pinned
     torch==2.1.0, which publishes no cp312 wheel, so a teammate on Python
     3.12 could not install the project at all; our own venv had 2.2.0.

Fix applied: New orbitalmind/splits.py derives every window from the actual
             series length and emits two plans -- a backtest that holds out
             the final 24 hours, and a submission that forecasts PAST the end
             of the record. Base models now train on exactly the array they
             are handed (the [:480] slices are gone) and predict_tft takes an
             explicit input window. The flow now fits out-of-sample residuals
             and returns a scalar bias plus a predictive distribution;
             Shapiro-Wilk is reported on held-out residuals, not manufactured.
             Reconstruction anchors on cleaned[t0]. RMSE is reported in ns and
             metres beside a persistence baseline. Failures fall back to
             persistence and are listed in the report. OrbitType is read from
             the column. Paths resolve from the repo root. Pins corrected to
             torch 2.2.0 / torchvision 0.17.0.

Outcome: 93/93 non-pipeline tests pass. Full-pipeline verification in progress.
Lesson: A test that asserts "it ran and produced a file" cannot distinguish a
        working model from a constant. Every accuracy claim must be measured
        in the units the metric is defined in, against a baseline that a
        trivial model could achieve.
---

---
Iteration: 9
Attempt: 2
What was tried: Running the rebuilt pipeline against the organisers' actual
                format -- real IGS data truncated to 672 rows per satellite --
                to confirm the day-8 fix.
Error: No exception. The day-8 window was correct, but the step-1 forecast
       jumped by -3339 ns (G02) and -18216 ns (G03) away from the last
       observed value. G01 was correct at +0.85 ns.
Root cause: Two further defects, neither reachable from synthetic data.

  6. correct_iod_jumps used a FIXED threshold (2.0 ns for clock, 1.0 m for
     ephemeris). GPS clock drift rates vary by orders of magnitude across the
     constellation: G02 drifts about 5 ns per 15-minute step and G03 about
     27 ns. Every single step therefore exceeded the threshold -- 671 jumps
     detected out of 671 possible steps for both -- and subtracting all those
     offsets removed the entire trend. The models were fitting the residue of
     a flattened series. An IOD upload is a discontinuity relative to a
     satellite's OWN step behaviour, so the threshold must come from that.

  7. Coordinate-frame mismatch in reconstruction. IOD correction shifts the
     series into its own frame. preprocess_satellite only returned the
     corrected series, so reconstruction anchored there and the submission
     came out in the corrected frame while the judges score against the
     measured frame. The gap is the total accumulated offset: 3339 ns for
     G02, 18216 ns for G03. Fixing defect 6 shrinks but does not remove this
     -- G03 has one genuine jump worth 2307 ns.

Fix applied: iod_correction.py now derives its threshold from the robust
             spread of the satellite's own first differences
             (median + 8 x MAD x 1.4826), with the explicit threshold kept as
             an override. Added count_jumps() so detection can be tested
             without mutating anything. preprocess_satellite now returns BOTH
             'observed' (outlier-cleaned, measurement frame) and
             'original_cleaned' (additionally IOD-corrected, modelling frame),
             and run_pipeline anchors and scores on 'observed'.

Outcome: Spurious jumps on real data went 671 -> 0 (G02) and 671 -> 1 (G03).
         The reconstruction accuracy ceiling on real IGS clock data is now
         0.03-0.17 ns, comfortably inside the 0.65 ns proposal target.
         102/102 non-pipeline tests pass.
Lesson: Any threshold expressed as an absolute physical constant is an
        assumption about scale. Synthetic data generated with that same
        assumption baked in can never expose it. Test preprocessing against
        real data early, and prefer thresholds derived from each series' own
        robust statistics.
---
