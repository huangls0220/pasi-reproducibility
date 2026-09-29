# R76 prespecified nontrivial-policy certification protocol

## Objective

R76 asks whether one nontrivial past-only guardrail, `g=0.75`, can satisfy the same episode-risk budget as the full envelope while recovering coverage. R75 established the selection rule and structural fallback but selected `g=1`; it did not demonstrate certified adaptive coverage recovery.

The R76 candidate is chosen only from R75 calibration evidence. R75 held-out outcomes are excluded from candidate design and power analysis. R76 uses new model seeds and constructs its held-out episodes only after the calibration decision is frozen.

## Fixed model and inputs

- Corrected PASI core with coverage-first matching, design-side decision fees, and execution settlement reported separately.
- The fixed uniform empirical mixture of the six preregistered R73 GeoLife weeks.
- Medium workload, response support `[0.70, 1.30]`, history 24 slots, rolling window 96, adaptive tail 0.05, and safety margin 0.02.
- Scenarios: `stable`, `down_step`, `up_step`, `down_ramp`, `alternating`, and `hidden_blocks`.
- Single statistical candidate: `g=0.75`. The full-envelope policy `g=1` is the structural fallback and paired held-out reference, not an additional calibration candidate.

## Risk target and calibration

For episode `i` and scenario `s`, the bounded loss is `L_i(s)=V_i(s)/max(A_i(s),1)`, where `A` is assigned service and `V` is modeled QoS violation. The target is `alpha=0.005` with familywise `delta=0.05` across the six scenario cells. Each cell uses the one-sided KL-Chernoff upper bound with tail `delta/6`.

Calibration uses seeds 770001--771600, or 1,600 complete episodes per scenario. Select `g=0.75` only if every scenario upper bound is at most 0.5%. Otherwise freeze `g=1` as a structural fallback and report that the nontrivial certificate failed. The threshold, candidate, scenarios, and sample size are not changed after outcomes are observed.

## Held-out test

Test seeds 780001--781000 remain unconstructed until the selection file is written. If `g=0.75` passes, the test evaluates frozen `g=0.75` and `g=1` on the same 1,000 episodes per scenario. Assignment- and qualified-coverage gains are paired by seed and reported with prespecified 10,000-replicate bootstrap intervals. If calibration falls back to `g=1`, the test evaluates only the fallback and no nontrivial coverage claim is made.

Held-out risk is descriptive and cannot change the selected policy. It is not presented as a second 0.5% certificate.

## Information-value basis

R75 calibration gives mean episode losses 0.00044965 in `down_step` and 0.00038379 in `hidden_blocks` for `g=0.75`, with zero loss in the other four scenarios. With one candidate and six scenario cells, the corresponding deterministic planning thresholds are about 1,377, 1,315, and 956 episodes, respectively. An empirical bootstrap using only R75 calibration episodes estimates about 90% simultaneous pass probability at 1,400 episodes and more than 99% at 1,600 under the same planning distribution. The six-scenario mean coverage gain over `g=1` is 4.821 percentage points. These values justify the experiment but do not form the R76 certificate.

## Scope

The certificate is conditional on the fixed six-week empirical mobility mixture and the model-generated task, request, cost, capacity, response, and payment process. It does not establish real participant payment behavior, population mobility generalization, or universal payment superiority. QIM-E-C and CSOPT-C remain author-implemented common-protocol adaptations and are not part of R76.

## Failure and stop rules

- Completed QoS failures remain risk observations; only incomplete executions are computational failures.
- Stop without a claim if any calibration cell is missing, any source identity changes, an input leaves the global response support, or a restricted source is marked redistributable.
- Do not inspect or construct held-out episodes before the selection is frozen.
- Do not replace the candidate, scenarios, seeds, risk budget, or bound after seeing R76 outcomes.
- Restricted GeoLife inputs and generated episodes remain outside the public result package.

## References

This protocol changes no external reference.
