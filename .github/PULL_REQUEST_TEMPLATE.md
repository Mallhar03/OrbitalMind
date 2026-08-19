## What and why

<!-- One paragraph. What changes, and what problem it solves. -->

## Type

- [ ] Bug fix
- [ ] Accuracy improvement
- [ ] Refactor / tooling
- [ ] Documentation

---

## Merge bar

Every box must be ticked or explicitly marked N/A with a reason.
See CONTRIBUTING.md.

### Tests
- [ ] `pytest tests/ --ignore=tests/test_pipeline.py -q` passes
- [ ] New behaviour has a test

**Which test would have caught this bug?**
<!-- Name it. "Added tests" is not an answer. -->

### If this claims an accuracy improvement
- [ ] Measured in ns / metres, not differenced or filtered space
- [ ] Measured on the held-out backtest
- [ ] Compared against linear extrapolation, not only persistence

| Horizon | Before | After | Linear baseline |
|---------|--------|-------|-----------------|
| 1 hr    |        |       |                 |
| 24 hr   |        |       |                 |

<!-- No table, no accuracy claim. Delete this block if N/A. -->

### Silent failure
- [ ] No bare `except:` or swallowing `except Exception:`
- [ ] Any fallback warns loudly AND is recorded in the output
- [ ] No default that hides a missing input

### Hardcoded assumptions
- [ ] No literal row counts or window indices — windows come from `orbitalmind.splits`
- [ ] No absolute physical thresholds — derive from the data's robust statistics
- [ ] Satellite properties read from columns, not parsed from ID strings
- [ ] Paths via `orbitalmind.paths`, not relative to the working directory

### Evaluation integrity
- [ ] No post-processor uses ground truth from the window it is scored against

**If this touches evaluation or post-processing — what data was it fitted on,
and why is that out-of-sample?**
<!-- Required. Leave blank only if genuinely untouched. -->

### Docs
- [ ] Docstrings with Args and Returns
- [ ] `ARCHITECTURE.md` / `README.md` / `skills/*.md` updated if behaviour changed

### Scope
- [ ] One concern only

---

## Reviewer

Do not approve until you can answer every question above in your own words.
If you cannot, ask. Approving what you do not understand is how seven defects
survived eight iterations.
