# R75 independent calibration--freeze--test protocol

Amended and refrozen on 2026-09-23 before any formal R75 calibration outcome was generated.  The excluded pilot showed that the simulator's top-level `status=failed` denotes an observed QoS-safety failure even when `execution_status=completed`.  This amendment keeps such observations as risk losses and treats only an incomplete execution as a computational failure.  The candidate set, seeds, sample sizes, risk budget, selection rule, and test gate are unchanged.

## Objective and claim boundary

R75 tests whether a past-only response guardrail can be selected without looking at held-out outcomes while controlling whole-episode QoS risk.  The certified population is the fixed empirical mixture of six preregistered GeoLife weeks and the paper's modeled task, cost, capacity, response, and payment generator.  It is not a certificate for human economic behavior, unseen mobility populations, or arbitrary response paths.

## Fixed inputs

- Corrected R56/PASI core, with coverage-first matching, design-side decision fees, and execution settlement reported separately.
- Six non-overlapping GeoLife weeks frozen in R73.  Raw traces, position caches, and generated episodes remain restricted and are never copied to the public package.
- Workload `medium`; global response envelope `[0.70, 1.30]`; adaptive history 24 slots; rolling window 96; adaptive tail 0.05; margin 0.02.
- Response scenarios: `stable`, `down_step`, `up_step`, `down_ramp`, `alternating`, and `hidden_blocks`.
- Candidate guardrail weights: `g in {0, 0.25, 0.50, 0.75, 1}`.  Larger `g` moves the past-only interval toward the fixed global envelope.

## Sampling and separation

- Pilot seed 749999 is excluded from all inference.
- Calibration seeds are 750001--751277 (1,277 complete episodes).
- Held-out test seeds are 760001--761000 (1,000 complete episodes).
- For each seed, a SHA-256-derived index selects one of the six R73 weeks from a uniform empirical mixture.  The selected week's position cache is then converted to a fresh complete episode with that seed.
- Test episodes may not be constructed until the calibration summary and frozen selection file have been written.

## Risk and simultaneous bound

For episode `i`, scenario `s`, and candidate `g`, define

`L_i(s,g) = V_i(s,g) / max(A_i(s,g), 1)`,

where `A` is the number of assigned services and `V` is the number that fail the modeled QoS threshold.  Thus `L` lies in `[0,1]`.  The risk target is `alpha = 0.005`.  A one-sided KL-Chernoff upper bound is computed for every one of the 30 candidate-scenario cells with Bonferroni familywise error `delta = 0.05`.

With zero observed episode loss, 1,277 calibration episodes are the minimum integer satisfying `1-(delta/30)^(1/n) <= alpha`.  This is a zero-loss certification threshold, not a generic 90% power calculation.

## Frozen selection rule

Choose the smallest candidate `g` whose simultaneous upper bound is no greater than `alpha` in all six scenarios.  If no candidate qualifies, freeze `g=1` as the structural global-envelope fallback and label the statistical certificate unsuccessful.  No held-out outcome may influence this choice.

## Held-out reporting

After selection is frozen, evaluate the selected policy, `g=0.25`, and `g=1` on all 1,000 held-out episodes (deduplicating identical policies).  Report seed-level results, scenario-level episode-risk bounds, pooled violation counts as descriptive quantities, assignment and qualified coverage, execution payment per assigned service, the selected/fallback status, and all failures.  Held-out results check reproducibility but do not retroactively alter the selected parameter or the calibration certificate.

## Stop and failure rules

- Keep every completed cell, including cells with QoS violations and top-level safety status `failed`.  Stop before making a claim if any execution is incomplete, any cell is missing, a source hash changes, a response leaves the global envelope, or an input has a non-GeoLife/re-distributable flag.
- Do not replace a mobility week, seed, scenario, or candidate after seeing an outcome.
- Any implementation amendment requires a new output directory and an explicit protocol amendment; existing R75 outputs are never overwritten.

## References

The experiment protocol itself adds no external reference.  The R75 manuscript revision separately adds related-work citations.
