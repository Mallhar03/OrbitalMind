# Deck corrections — exact replacement wording
# For latest.pdf, the Smart Horizon SH-DST-03 submission.
# Approved by Mallhar 31 August 2026. Evidence in memory/ppt_audit/.

NINE corrections. Each gives the slide, the current wording, and the exact text to
replace it with. (This intro said "four" while the body already held nine — do not
skim past corrections 5 to 9.) Nothing here changes the code — three of the four are cases where
the code is right and the slide is wrong.

---

## 1. Normalizing Flow — slide 3 "Innovation", slide 5, slide 7

**This is the most important one, and it is not cosmetic.**

### Current wording

> Slide 3: "Normalizing Flow post-processor that engineers Gaussian residuals
> **by design**."
> Slide 5: "A probability distribution from Normalizing Flow ensuring Gaussian
> residuals **by construction**."
> Slide 7: "Residuals **engineered** to follow Normal distribution via Normalizing
> Flow post-processor, **not left to chance**."

### Why it has to change

The claim promises a guaranteed outcome. When it was implemented as written, the
code delivered the guarantee the only way a guarantee can be delivered: it
manufactured the result. The Shapiro-Wilk test returned p = 0.9999 for every
possible input, including maximally non-Gaussian residuals, because it was reading
the answer rather than testing the data. That is recorded in
`memory/what_failed.md` iteration 9.

A judge who asks "what happens if the residuals aren't Gaussian?" needs an answer.
Under the current wording there isn't one, because the slide says that cannot
happen. Under the corrected wording the answer is straightforward and creditable:
we measure it, and right now it fails, which tells us something real about the
ensemble.

### Replacement text

> **Slide 3:** "Normalizing Flow post-processor that calibrates the residual
> distribution, with Gaussianity measured and reported rather than assumed."
>
> **Slide 5:** "A calibrated predictive distribution from the Normalizing Flow,
> with residual normality verified by Shapiro-Wilk and reported honestly."
>
> **Slide 7:** "Residual distribution calibrated by the Normalizing Flow
> post-processor, with normality measured by Shapiro-Wilk and reported —
> pass or fail — rather than assumed."

### The line to have ready in Q&A

"Our current Shapiro-Wilk result is a fail. That is a finding, not a bug: it means
the ensemble is still leaving structure in its errors, and the place to fix that is
the models, not the test. We deliberately do not tune the post-processor to make
the statistic pass, because a previous version of this project did exactly that and
produced a meaningless p-value."

---

## 2. GPyTorch — slide 4 "Tech Stack"

### Current wording

> "Programming & Frameworks: Python 3.11, PyTorch 2.x, LightGBM, GPyTorch"

### Why it has to change

GPyTorch appears nowhere in the project. Not in `requirements.txt`, not in any
import under `src/`, not even in the team's own internal tech-stack notes — only on
the judge-facing slide. It is the single most falsifiable line on the deck: one
`grep` settles it.

The Gaussian Process work it implies was considered and deliberately rejected —
`memory/decisions.md` Decision 004 chose the Normalizing Flow over a GP because a GP
scales poorly across many satellites with ~672 points each. So the honest slide does
not merely drop the name; the rejection is a point in the project's favour.

### Replacement text

> "Programming & Frameworks: Python 3.10-3.12, PyTorch 2.x, LightGBM, normflows"

Python is stated as a range because the project is verified on 3.10 through 3.12 and
the local environment is 3.12 — the deck currently says 3.11, which is narrower than
the truth. `normflows` is the library that actually provides the Normalizing Flow.

---

## 3. EWT → EMD — slide 3 and slide 4

### Current wording

> Slide 3: "...physics-aware preprocessing, **EWT decomposition**, multi-model
> training..."
> Slide 4: "...single-difference transformation, **EWT decomposition** into
> trend/periodic/noise layers."

### Why it has to change

The code uses EMD (Empirical Mode Decomposition, via PyEMD), not EWT, and that was a
deliberate choice recorded as Decision 001. It is a choice worth defending rather
than hiding: EWT and wavelet methods impose a fixed basis, while EMD derives its
basis from the signal itself, which suits a non-stationary error that drifts.

It also buys something concrete and measurable. EMD's completeness property means
trend + periodic + noise reconstructs the input **exactly** — measured at
1.39e-17 to 1.78e-15 across all 190 real series. Nothing is lost in the
decomposition, and that is checkable on the spot.

### Replacement text

> **Slide 3:** "...physics-aware preprocessing, **EMD decomposition**, multi-model
> training..."
>
> **Slide 4:** "...single-difference transformation, **EMD decomposition** into
> trend/periodic/noise layers."

### The line to have ready in Q&A

"We use EMD rather than EWT because the basis comes from the signal instead of being
imposed on it, which matters for a drifting non-stationary error. It also gives us
exact reconstruction — we measured the residual at 10^-15 across all 190 series, so
the decomposition provably loses nothing."

---

## 4. Diffusion model — slide 3, slide 4, slide 5

### The inconsistency, and it is the deck's own

Slides 3 and 4 sell a **four-model** ensemble: TFT + TCN-LSTM + Neural ODE +
Diffusion. Slide 5 then lists the fusion as TFT + TCN-LSTM + Neural ODE — three
models, no Diffusion. **The deck contradicts itself about its own ensemble**, which
is the kind of thing a careful judge notices.

Meanwhile the code trains LSTM + TCN-LSTM + TFT + Neural ODE. `models/diffusion.py`
is fully implemented and tested and imported by nothing, and there is no entry in
`memory/decisions.md` authorising the LSTM that took its place.

### Decision taken, then REVISED

The first decision was to wire Diffusion in, on the assumption that connecting an
already-implemented, already-tested module was mechanical. **That was wrong**, and
the S4 audit found why.

`train_diffusion()` fits an **unconditional** Gaussian to pooled residual
statistics. It takes no input sequence, unlike every other model in the ensemble.
Connected as it stands, it would hand the meta-learner noise drawn from a fixed
distribution rather than a forecast conditioned on that satellite's history. Making
it conditional is a redesign, not plumbing — and building a fourth model days before
a hackathon to satisfy a slide is the wrong order of priorities.

**Revised decision (017): drop Diffusion from the deck.** This also resolves the
deck's own self-contradiction, since slide 5 already lists three models.

Note the code trains a fourth model the deck never mentions: an LSTM. The honest
description of what runs is **LSTM + TCN-LSTM + TFT + Neural ODE**, fused by
LightGBM.

### Replacement text

> **Slide 3:** "...multi-model training (TFT + TCN-LSTM + Neural ODE + LSTM)..."
>
> **Slide 4:** "TFT for multi-horizon attention, TCN-LSTM for short jumps, Neural
> ODE for drift physics, LSTM as a sequence baseline."
>
> **Slide 5:** "A point estimate from LightGBM fusing TFT + TCN-LSTM + Neural ODE
> + LSTM"

`models/diffusion.py` stays in the repository, implemented and tested, as
groundwork. Whether it earns a place belongs to day-06's ablation, on evidence.

---

## Not changed, and why

**The 0.65 ns and 7.5 ns targets, and the 10-15% improvement over SSA-TCN.** These
stay as written for now because they are targets and comparisons, not claims of
current achievement — but none of them is yet measured. No full-pipeline run against
real data exists. Before any of these three numbers is spoken to a judge as an
achieved result, that run has to happen and land in `outputs/`.

---

## 5. The "full 7-day context window" — slide 3 and slide 5

### Current wording

> Slide 3: "Full **7-day** context window with 96 lag features."
> Slide 5: "Full **7-day** context window."
> Slide 3 comparison: "Existing LSTM baselines use only 6 time steps (90 min) of
> history. Our pipeline uses the full 7-day window, capturing 12-hr and 24-hr
> orbital cycles."

### Why it has to change

`splits.py:32` sets `SEQ_LEN = 96`, and 96 fifteen-minute steps is **24 hours**,
not seven days. Every model sees a 24-hour lookback at both training and inference.
Seven days of data are ingested and preprocessed, but no model receives a seven-day
window.

The comparison against the 6-step baseline is still true and still strong — a
24-hour window is **16 times** more history than 90 minutes, and it is enough to
see the 24-hour cycle the physics argument rests on. Only the "7-day" figure is wrong.

### Replacement text

> **Slide 3:** "Full 24-hour context window (96 steps), from a 7-day ingested record."
>
> Precision note for Q&A: the file spans 7 days, but the training window is not
> the whole of it — the final day is held back for calibration and the forecast
> target. Say "7-day record" rather than "trained on 7 days" if a judge presses.
>
> **Slide 5:** "24-hour context window, 96 steps."
>
> **Comparison:** "Existing LSTM baselines use only 6 time steps (90 min) of history.
> Our models use a 96-step, 24-hour window — 16x more context, enough to capture the
> daily orbital cycle."

---

## 6. pytorch-forecasting — slide 4 "Tech Stack"

This is the same shape as the GPyTorch problem, found by the same kind of check.

`pytorch-forecasting==1.1.1` is pinned in `requirements.txt` and named in the
project's internal tech-stack notes, but **nothing imports `pytorch_forecasting`**
anywhere in the codebase.

What `tft.py` actually implements (`DirectTFT`, line 33) is a hand-written single-head
`TransformerEncoderLayer` with a skip connection. That is real attention, genuinely
trained and tested — the model is not fictional. But it is not the published Temporal
Fusion Transformer architecture, and the deck's tech-stack line implies the library.

### What to do

Either drop the dependency and describe the model accurately, or adopt the library.
The honest short-term fix is the former, because the code already works:

> Describe it as: "attention-based multi-horizon model (single-head transformer
> encoder with skip connection)" rather than implying the reference TFT
> implementation.

Remove `pytorch-forecasting` from `requirements.txt` only after confirming nothing
imports it — an unused pinned dependency is also a supply-chain surface.


---

## 7. FFT spectral features — slide 3 "Features"

### Current wording

> "FFT spectral features capturing 12-hour and 24-hour satellite periodicity."

### Why it has to change

This is now closer to true than it was, but it is still not true, and the reason is
worth understanding rather than papering over.

The feature code exists and is correct. The pipeline now does feed the meta-learner
periodicity information — sine and cosine encodings of position in the 24-hour and
12-hour cycles — and those rank top by gain, so periodicity genuinely reaches the
ensemble.

But they are not FFT spectral features. Actual FFT amplitudes were tried and
measured **exactly 0.0 gain**, by construction: they describe the history the whole
forecast is made from, so they take a single value across the entire horizon, and a
feature with no variance cannot influence a tree model. That is a physical
constraint, not a bug — no new observations arrive during a forecast, so the
spectrum of the history cannot change.

Their correct home is the base models, which consume history directly. That is a
multivariate-input change, and it is day-05 work.

### Replacement text

> "Explicit 12-hour and 24-hour periodicity encodings supplied to the ensemble,
> with FFT spectral analysis of the input window available for model features."

Only claim the FFT half once base models consume it and an ablation shows it
changes the forecast.

### GATING CAVEAT — check this before the claim is spoken

The periodicity encodings reach the meta-learner **only when the pipeline is run
with `--features`**, which defaults to OFF, and the Makefile's default target does
not pass it. The present tense in the replacement wording is therefore true only of
a features-enabled run.

The day-6 ablation shows features help (-2.2% at 1 hour, the best configuration
measured), and a full `--features` run is being produced into `outputs_features/`.
**Before this claim goes on a slide, confirm the numbers being presented come from
that run and not from the default one.** Presenting default-run numbers while
claiming periodicity features reach the ensemble would be an overclaim of exactly
the kind this document exists to remove.

---

## 8. Gaussian likelihood loss — slide 3 "Features"

### Current wording

> "Gaussian likelihood loss built into training."

### Why it has to change

No model does this. All four trainers use `nn.MSELoss()` and none has a variance
output (`lstm.py:88`, `tcn_lstm.py:108`, `tft.py:144`, `neural_ode.py:114`). A judge
who asks "which loss function, show me" gets nothing.

It could be built — a variance head trained on Gaussian negative log-likelihood is
standard. But it would duplicate work the project already does correctly. The
predictive uncertainty this claim implies is delivered by the Normalizing Flow
post-processor, which produces the calibrated intervals the submission actually
reports. Adding a second uncertainty mechanism days before the event would risk the
one that is verified working.

### Replacement text

> "Calibrated predictive uncertainty from the Normalizing Flow post-processor,
> producing 95% intervals per forecast step."

This is both true and a better answer, because it points at something demonstrable.

---

## 9. Physics-informed constraints — slide 3 "Innovation"

### Current wording

> "Innovation 1: orbit-type branching with **physics-informed constraints**."

### Why it has to change

There is no constraint code. A search for "constraint" across `src/` returns one
docstring word and no logic. The phrase currently rests on the Neural ODE existing —
which is already sold separately as Innovation 3. **The deck counts one idea twice**,
and a judge who reads both bullets will notice.

The orbit-type branching half also needs care: `orbit_type` selects no architecture
and no hyperparameter anywhere. What the code does instead is stronger — it trains a
fresh model per satellite, so no two satellites share weights at all, let alone two
orbit types (Decision 014).

### Replacement text

> **Innovation 1:** "Per-satellite models — every satellite gets its own fitted
> ensemble, so no satellite's dynamics are averaged into another's."

That is true, checkable, and a stronger claim than branching by orbit type,
especially now that the periodicity argument behind orbit-type branching is known
not to hold for most MEO satellites.
