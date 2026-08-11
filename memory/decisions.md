# Architecture Decisions Log

Record every significant technical decision here with the reason.
Before changing an approach, check if it was already decided here.

## Format:
---
Decision: [what was decided]
Date: [when]
Reason: [why this and not something else]
Alternatives rejected: [what else was considered and why rejected]
---

## Decision 001
Decision: Use EMD-signal library (PyEMD) for signal decomposition, not PyWavelets
Date: setup phase
Reason: PyEMD implements true EMD which separates GNSS error signal into
        physically meaningful IMFs. PyWavelets uses fixed basis functions
        which do not adapt to the non-stationary nature of the error signal.
Alternatives rejected: PyWavelets (fixed basis), scipy.signal (not designed for IMF decomposition)

## Decision 002
Decision: Train separate models for GEO and MEO orbit types
Date: setup phase
Reason: GEO satellites have 24-hour periodicity. MEO have 12-hour periodicity.
        A single model averages these out and loses accuracy on both.
Alternatives rejected: single model with orbit_type as a feature (loses periodicity signal)

## Decision 003
Decision: Use sequence_length=96 (24 hours of history) as minimum
Date: setup phase
Reason: Satellite clock errors have 24-hour cycles. The existing GitHub repo
        (Amit-jha98/GNSS_Error_Predictor) uses only 6 steps (90 min) and
        cannot see these cycles. Our edge over that repo is this full window.
Alternatives rejected: sequence_length=6 (cannot see 24hr cycle),
                       sequence_length=48 (misses full cycle)

## Decision 004
Decision: Normalizing Flow as post-processor, not GP for Gaussian metric
Date: setup phase
Reason: Gaussian Process scales poorly with 672 training points per satellite
        across multiple satellites. Normalizing Flow is trainable end-to-end
        and directly maps output distribution to Gaussian by construction.
Alternatives rejected: GP (scaling issues), hoping residuals are Gaussian (not engineered)

## Decision 005
Decision: LightGBM as meta-learner, not a neural network
Date: setup phase
Reason: LightGBM is fast, interpretable, and works well on small tabular
        inputs (3-4 model outputs as features). A neural meta-learner would
        overfit on the small number of meta-training samples.
Alternatives rejected: MLP meta-learner (overfits), simple average (suboptimal weighting)
