# Engineering Decisions

Why the pipeline is built the way it is. Each entry records what was decided, the
reason, and what was rejected — so a choice is not silently reversed by someone who
only sees the code.

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

CORRECTION 2026-08-31 (measured during the S3 audit):
        The stated Reason does not hold on real multi-GNSS data. Measured across
        all 190 series: GEO 5/7 (71%) match on clock and 3/7 (43%) on ephemeris;
        MEO only 23/88 (26%) on clock and 40/88 (45%) on ephemeris. 65 of 88 MEO
        satellites are 24-hour dominant.
        The DECISION (separate GEO/MEO branches) is NOT overturned here — it may
        still be justified by other differences between the orbit types, and the
        deck sells it as a headline innovation (C-02, C-06). But the periodicity
        argument given above is no longer the reason, and must not be offered to
        judges as one. Whether the split still earns its place is an open
        question for the S4 gate.

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

## Decision 006
Decision: Pull the latest 8 days of real GNSS data; train on the first 7, hold
          out day 8 as truth. Day 8 is written to a SEPARATE file
          (data/raw/gnss_holdout.csv) and must never be trained on, fitted on,
          or inspected while making modelling choices.
Date: 30 August 2026
Reason: Mallhar's instruction. The hackathon task is exactly this shape — 7 days
        of error history in, day 8 predicted at 15-minute intervals — so the
        local setup should mirror it, and the team needs a real outcome to
        compare forecasts against. Keeping day 8 in a separate file makes
        leakage an explicit act rather than an easy accident.
Alternatives rejected: 14-day pull with an internal split (the S1 audit flagged
        the undocumented 14-day window as silent drift from the deck's stated
        ~672 rows/satellite); merging day 8 into one file (too easy to leak).

## Decision 007
Decision: Source SP3 from the GFZ multi-GNSS rapid combination (GFZ0MGXRAP),
          not the IGS rapid combination (IGS0OPSRAP).
Date: 30 August 2026
Reason: IGS0OPSRAP is GPS-only by construction, and GPS is entirely MEO, so it
        can never satisfy the deck's GEO/GSO requirement — the S1 gate failed on
        exactly this. GFZ0MGXRAP carries GPS, BeiDou, Galileo, GLONASS and QZSS,
        and supplies 7 geosynchronous satellites (C06-C08 BeiDou IGSO, J02-J04
        QZSS IGSO, J07 true GEO). Mallhar confirmed the organisers' files will
        include GEO satellites, so a GPS-only dataset was never acceptable.
Alternatives rejected: WUM0MGXFIN / COD0MGXFIN (not published for the target
        dates); staying GPS-only and demoing GEO on synthetic data.

## Decision 008
Decision: Classify OrbitType from orbital geometry (median orbital radius above
          40000 km is GEO), not from a PRN lookup table.
Date: 30 August 2026
Reason: The contest organisers' labelling convention is unknown, so the code must
        not assume one. Geometry is self-evident from the SP3 positions and works
        on any constellation. Inclined geosynchronous satellites (IGSO/GSO) are
        labelled GEO alongside true equatorial GEO because both carry the 24-hour
        periodicity the deck's GEO branch models, and the deck itself says
        "GEO/GSO" as one class.
Alternatives rejected: hardcoded GEO_PRNS set (silently produced zero GEO rows
        for every real pull, because those PRNs were absent from the product).

## Decision 009
Decision: Never clip EphemerisError_m during ingestion.
Date: 30 August 2026
Reason: The previous implementation clipped to [-5, +5] m. The S1 audit measured
        3,069 of 43,007 rows (7.1%) sitting exactly on the -5.0 m floor, meaning
        the true error magnitude was destroyed for one row in fourteen and any
        model would learn a false ceiling. Outlier handling is S2's job — MAD
        outlier removal exists for precisely this — and ingestion must report
        what the data says.
Alternatives rejected: widening the clip (still censors); documenting the clip
        in the deck (the model still learns the false ceiling).

## Decision 010
Decision: Soften the deck's Normalizing Flow wording. The Flow CALIBRATES the
          residual distribution; Gaussianity is MEASURED and REPORTED, never
          engineered or guaranteed.
Date: 31 August 2026, approved by Mallhar
Reason: The deck promised residuals "engineered Gaussian by design" and "by
        construction". Implemented as written, that guarantee was delivered the
        only way a guarantee can be: the code manufactured it, returning
        p = 0.9999 for every input including maximally non-Gaussian residuals
        (iteration 9). The wording made an honest
        implementation impossible. Exact replacement text is in
        the deck-corrections list kept with the team’s slides.
Alternatives rejected: keeping the wording and making the test pass (that is the
        iteration-9 fraud); dropping the Flow entirely (it does real work
        calibrating the predictive distribution).

## Decision 011
Decision: Remove GPyTorch from the deck's tech stack. State Python as 3.10-3.12
          and name normflows, which is the library actually used.
Date: 31 August 2026, approved by Mallhar
Reason: GPyTorch appears nowhere in the project — not in requirements.txt, not in
        any import under src/, not in the team's internal notes. Only on the
        judge-facing slide, where one grep disproves it. The Gaussian Process work
        it implies was deliberately rejected in Decision 004 because a GP scales
        poorly across many satellites at ~672 points each, so the rejection is a
        point in the project's favour rather than something to hide.
Alternatives rejected: adding GPyTorch to the project to make the slide true
        (Decision 004 already rejected the GP approach on its merits).

## Decision 012
Decision: Correct the deck to say EMD, not EWT, and defend the choice rather than
          hide it.
Date: 31 August 2026, approved by Mallhar
Reason: The code has used EMD since Decision 001 and the deck was never updated.
        EMD derives its basis from the signal instead of imposing a fixed one,
        which suits a drifting non-stationary error, and its completeness property
        gives exact reconstruction — measured at 1.39e-17 to 1.78e-15 across all
        190 real series. That is a checkable claim and a stronger pitch than EWT.
Alternatives rejected: switching the implementation to EWT to match the slide
        (would discard exact reconstruction to satisfy a typo).

## Decision 013
Decision: Wire the Diffusion model into the ensemble during S4, making the deck's
          four-model claim true, and fix slide 5 to list the same four models as
          slides 3 and 4.
Date: 31 August 2026, approved by Mallhar
Reason: The deck sells TFT + TCN-LSTM + Neural ODE + Diffusion on slides 3 and 4
        but lists only three on slide 5 — it contradicts itself. The code trains
        LSTM + TCN-LSTM + TFT + Neural ODE, with no decision record authorising
        the LSTM that took Diffusion's place. models/diffusion.py is fully
        implemented and tested but imported by nothing. Wiring it makes the deck
        honest without shrinking the project's claims.
Alternatives rejected: dropping Diffusion from all three slides (still available
        as a fallback if S4 measurement shows it does not earn its place — but
        that call must rest on evidence, not on an assumption made now).

## Decision 014
Decision: State the separation honestly — the pipeline trains a FRESH MODEL PER
          SATELLITE, not one shared model per orbit type. Orbit type selects no
          architecture and no hyperparameter; it only labels saved weights.
Date: 31 August 2026
Reason: The S4 audit found orbit_type reaches nothing but filenames in all four
        trainers, so "orbit-type branching" has no architectural code path. But
        the code does something STRONGER than the deck claims: _train_base_models
        is called per satellite with freshly initialised models, so no two
        satellites share weights at all, let alone two orbit types. Decision 002's
        stated worry — "a single model averages GEO and MEO out" — cannot occur.
        This matters more now that Decision 002's periodicity reason is known
        false (65 of 88 MEO satellites are 24-hour dominant): branching on that
        physics would encode an assumption the data contradicts, whereas
        per-satellite models make the question moot.
        Also fixed: saved weights were keyed on orbit type alone, so every GEO
        satellite overwrote the previous one and only the last satellite's weights
        survived on disk under a name implying they represented all of them.
        Weights are now keyed by satellite id.
Alternatives rejected: adding orbit-type-specific architecture (would encode the
        periodicity assumption the data contradicts); leaving the deck as-is
        (describes branching the code does not do).

## Decision 015
Decision: Feed the meta-learner CAUSAL engineered features computed in
          run_pipeline._meta_features(), rather than wiring
          features/build_feature_matrix() in literally as plan/day-04 asks.
Date: 31 August 2026
Reason: day-04 says "Wire build_feature_matrix() into the pipeline". That function
        produces lag and rolling features of each timestep. The meta-learner fuses
        forecasts for 96 steps the pipeline has NOT observed, so a lag feature of
        those steps would require the very values being predicted. Wiring it
        literally would either be impossible at forecast time, or would create
        train/serve skew — features present when fitting and absent when
        predicting — which is worse than not having them.
        What is fed instead is causal and available at forecast time: FFT
        amplitudes at 24h and 12h measured over the 96-step window ending at the
        last OBSERVED sample, sine/cosine encodings of position in the 24h and 12h
        cycles, the horizon step index, and the level and slope of the history
        window. Verified reaching the meta-learner: 13 features (4 base model
        outputs + 9 engineered), with the periodic encodings ranking top by gain.
        This is what makes deck claim C-04 — "FFT spectral features capturing
        12-hour and 24-hour satellite periodicity" — true for the first time.
        Controlled by --features, OFF by default, so the effect can be MEASURED
        rather than assumed. No accuracy claim is made yet; day-04 explicitly
        says today is plumbing.
Alternatives rejected: wiring build_feature_matrix() for the calibration window
        only (train/serve skew); forecasting the features themselves (compounds
        forecast error into the fusion inputs).

## Decision 016
Decision: Remove the constant FFT/history features from the meta-learner, and do
          NOT claim deck C-04 ("FFT spectral features capturing 12-hour and
          24-hour periodicity") is satisfied.
Date: 31 August 2026
Reason: Decision 015 fed fft_24h, fft_12h, hist_level and hist_slope to the
        meta-learner. The S4 audit measured them on real data (C06) and found
        exactly 0.0 gain — not by chance but by construction. Each describes the
        history the whole horizon is forecast from, so _meta_features() broadcast
        one value across every sample with np.full(). A feature with zero variance
        inside the training window cannot be split on, so it can never influence a
        prediction. My own verification checked that the columns REACHED the
        meta-learner and never checked whether they could DO anything.
        The underlying constraint is physical, not a coding slip: FFT-of-history
        cannot vary across a pure forecast horizon, because no new observations
        arrive during it. The meta-learner sees only the horizon, so this is the
        wrong place for those features. Their right home is the base models, which
        consume history directly — that means multivariate base models, which is
        day-05 scope, not day-04.
        The features that remain (horizon step, and sine/cosine encodings of the
        24h and 12h cycles) do vary per sample and do rank top by gain. They give
        the ensemble periodicity information, which is worth having, but they are
        NOT the FFT spectral features C-04 promises.
Alternatives rejected: leaving the inert columns in place (they imply C-04 is
        satisfied while being incapable of affecting anything — exactly the class
        of dishonesty this audit system exists to catch); recomputing the FFT using
        the models' own forecasts (compounds forecast error into the fusion input).

## Decision 017
Decision: REVISE Decision 013. Do NOT wire the Diffusion model into the ensemble.
          Drop it from the deck's four-model claim instead.
Date: 31 August 2026
Reason: Decision 013 assumed wiring was mechanical — the module is implemented and
        tested, so connecting it looked like an import. The S4 audit found the
        structural reason that is wrong: train_diffusion() fits an UNCONDITIONAL
        Gaussian to pooled residual statistics. It takes no input sequence, unlike
        every other predict_X() in the ensemble. Plugged into _base_forecasts() as
        it stands it would hand the meta-learner noise drawn from a fixed
        distribution, not a forecast conditioned on the satellite's history.
        Making it conditional is a redesign, and inventing a fourth model days
        before a hackathon to satisfy a slide is the wrong order of priorities.
        Slides 3 and 4 must therefore be corrected to a three-model ensemble
        (TFT + TCN-LSTM + Neural ODE), which also resolves the deck's existing
        self-contradiction — slide 5 already lists only three.
        Note the code trains a fourth model, LSTM, that the deck never mentions.
        The honest description is: LSTM + TCN-LSTM + TFT + Neural ODE, fused by
        LightGBM.
Alternatives rejected: wiring it as-is (feeds unconditioned noise into the
        fusion); redesigning it now (real work, no evidence it would earn its
        place, and day-06's ablation is the proper way to decide that).

## Decision 018
Decision: Correct the deck for C-05 ("Gaussian likelihood loss built into
          training") rather than implementing a variance head in all four models.
Date: 31 August 2026
Reason: All four trainers use nn.MSELoss() with no variance output, so the claim is
        false as written. It could be made true — a variance head trained on
        Gaussian NLL is standard — but it would be redundant here. The predictive
        uncertainty the claim implies is ALREADY delivered, by the Normalizing Flow
        post-processor, which produces the calibrated interval the submission
        reports (C-26, verified built). Adding a second, separate uncertainty
        mechanism days before the hackathon would duplicate what already works and
        put the calibrated intervals at risk for no gain the deck needs.
        The honest slide describes where the uncertainty actually comes from.
Alternatives rejected: implementing a variance head and NLL (real work, duplicates
        the Flow, risks the one uncertainty path that is verified working);
        leaving the claim (false as written).

## Decision 019
Decision: Resolve MANDATORY 1 — whether to keep the discarded IMF[0] — through
          day-06's ablation, not by a judgement call now.
Date: 31 August 2026
Reason: S3 established the discarded component is 33-77% of variance (mean 51%) and
        is NOT clean noise: spectral flatness 0.42 against a white-noise baseline of
        0.566, lag-1 autocorrelation -0.23, and 27.7-37.5% of the periodic
        component's amplitude at the 12.16 cycles/day upload cadence leaking into
        it. But most of its energy is broadband and near Nyquist, which no model
        can predict.
        So the physics does not settle it: there is real signal in there and real
        noise, and the question is whether the recoverable part outweighs the noise
        the models would have to learn around. That is an empirical question, and
        the project already has the right mechanism for it — day-06's ablation.
        Critically, that ablation must run on VALIDATION data, never on the
        2026-08-28 holdout. Comparing holdout RMSE with and without would be tuning
        against the answer, which the integrity charter forbids.
Alternatives rejected: keeping IMF[0] now (adds ~51% variance of mostly
        unpredictable noise on an assumption); dropping the question (it puts a
        measurable ceiling on accuracy and a judge may well ask).

## Decision 020
Decision: All four base models — LSTM, TCN-LSTM, TFT, Neural ODE — are RETAINED
          unconditionally. Ablation results are diagnostic, never grounds for
          removing a component.
Date: 31 August 2026, instruction from Mallhar
Reason: A day-6 ablation on two satellites showed that dropping the LSTM (1hr RMSE
        0.1251 vs baseline 0.1634) and the Neural ODE (0.1540) appeared to IMPROVE
        the fused forecast, while dropping TFT (0.1687) and TCN-LSTM (0.1729) made
        it worse. Mallhar's instruction on seeing this was direct: the models are
        the most important point of the solution, retain them and move ahead.
        This is consistent with the standing rule that the architecture is frozen —
        that rule now cuts both ways, forbidding removal as well as addition. The
        Neural ODE is deck claim C-08, one of the three named innovations.
Consequence for how ablation is used: results become PREPARATION, not pruning. If a
        model contributes little, the team needs to know before a judge asks what it
        adds. A near-zero contribution is also often a symptom — undertrained, badly
        scaled, or fed the wrong window — which is a reason to investigate that model,
        not to delete it.
Alternatives rejected: dropping the LSTM and Neural ODE on the ablation evidence
        (overruled); deciding on a larger sample first (the instruction is
        unconditional, so more evidence would not change the outcome).

## Decision 021
Decision: The shaping/calibration lane supplies a predictive interval and the
          Q-Q diagnostic ONLY. It does not modify the submitted point forecast,
          and there is no residual transform anywhere in the pipeline.
Date: 3 September 2026
Reason: Two facts settle this. (1) At evaluation the team submits PREDICTIONS;
        the organisers compute the residual as (prediction - hidden truth) and
        run Shapiro-Francia themselves (Note.pdf 1c-1e). A transform applied to
        a residual therefore never reaches the scorer — it would only flatter
        the team's own internal numbers, which is self-deception. (2) The
        Shapiro-Francia W statistic is invariant to location and scale, so any
        affine correction to the point forecast is provably W-neutral. Improving
        W is the forecaster's job (a better point forecast leaves whiter
        residuals), never the calibrator's.
        A per-parameter bias correction (gated on a held-out training tail) was
        built and MEASURED on the shipped day-8 data. It made the priority-2
        residual mean WORSE on 4 of 5 series (mean |residual mean| 0.36 -> 0.62):
        a bias fit on the calm training week does not transfer through day-8's
        regime change, least of all the GEO divergence. Per the anti-clutter
        rule it did not earn its place and was removed. What remains — a robust
        (MAD-based) dispersion for the predictive interval, and Q-Q plots — is
        fit on training only and cannot alter the forecast or W.
        tests/test_shaping.py pins the guarantee: Calibration exposes no
        point-altering method, and the entrypoint's submitted point equals the
        raw model output exactly.
Measured normality on the shipped day-8 data (leak-free model selection): 3 of 5
        series pass (H=0); GEO fails because the satellite genuinely diverges at
        week's end (clock std 2->16 ns across days 6-8), and MEO_Train2-b0 fails
        on small-sample (n=18) single-outlier fragility. Neither is fixable by
        any honest post-processing; both are reported as measured outcomes.
Alternatives rejected: a residual-space whitening transform (never reaches the
        real submission — self-deception); a Yeo-Johnson target transform,
        including a leak-free per-series selected variant (measured net-zero on
        normality across the five series, so clutter); keeping the bias
        correction (measured to hurt priority-2).
