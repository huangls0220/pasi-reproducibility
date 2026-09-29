# R76 prespecified nontrivial-policy certificate

R76 follows the negative boundary found in R75. R75 validated the
calibration--freeze--test rule but selected only the fully conservative
`g=1` policy. Using R75 calibration evidence only, R76 prespecified `g=0.75`
as its sole statistical candidate, excluded every R75 held-out outcome from
design, and used new calibration and held-out seeds.

## Frozen protocol

- Risk budget: mean complete-episode QoS loss `V/max(A,1) <= 0.005`.
- Familywise error: `0.05` over six response scenarios.
- Statistical candidate: `g=0.75`; structural fallback/reference: `g=1`.
- Calibration: 1,600 new episodes per scenario, seeds 770001--771600.
- Held-out evaluation: 1,000 new episodes per scenario and policy, seeds
  780001--781000, constructed after the selection record was frozen.
- Mobility: fixed empirical mixture of six preregistered GeoLife weeks.
- Economics: requests, costs, bids, capacities, responses, and payments remain
  modeled or mechanism-computed.

`protocol.md` and `preregistration.json` define the rule. `frozen_runner/`
preserves the exact runner and protocol hashes used in the completed study.
`run_r76_public.py` changes only repository path resolution for a clean clone.

## Result

The worst calibration upper bound for `g=0.75` was 0.004332883, so the
candidate passed and was frozen. On the independent held-out episodes, frozen
`g=0.75` improved mean assignment coverage by 4.809 percentage points and
mean qualified coverage by 4.802 points relative to `g=1`. All five nonzero
scenario-specific paired-bootstrap intervals were strictly positive; the
alternating scenario was tied.

The held-out policy recorded 742 modeled QoS violations among 5,678,644
assignments. Its largest held-out cellwise bound was 0.00707293. Therefore the
held-out phase validates the coverage consequence but is not presented as a
second 0.5% certificate. The certificate comes from the fresh calibration
phase.

## Reconstruction

First reconstruct the six R73 windows as described in `cross_window/README.md`.
Keep the R73 caches, all generated R76 episodes, and result work directories
outside the repository. Then run the actions in order:

```powershell
python -B risk_certification/r76/run_r76_public.py freeze --r73 C:\restricted\r73-windows --restricted C:\restricted\r76 --results C:\results\r76
python -B risk_certification/r76/run_r76_public.py smoke --r73 C:\restricted\r73-windows --restricted C:\restricted\r76 --results C:\results\r76
python -B risk_certification/r76/run_r76_public.py prepare-cal --r73 C:\restricted\r73-windows --restricted C:\restricted\r76 --results C:\results\r76
python -B risk_certification/r76/run_r76_public.py calibrate --r73 C:\restricted\r73-windows --restricted C:\restricted\r76 --results C:\results\r76 --workers 16
python -B risk_certification/r76/run_r76_public.py select --r73 C:\restricted\r73-windows --restricted C:\restricted\r76 --results C:\results\r76
python -B risk_certification/r76/run_r76_public.py prepare-test --r73 C:\restricted\r73-windows --restricted C:\restricted\r76 --results C:\results\r76
python -B risk_certification/r76/run_r76_public.py test --r73 C:\restricted\r73-windows --restricted C:\restricted\r76 --results C:\results\r76 --workers 16
python -B risk_certification/r76/run_r76_public.py summarize --r73 C:\restricted\r73-windows --restricted C:\restricted\r76 --results C:\results\r76
```

The public `results/` directory includes non-sensitive seed-level CSV files,
cell summaries, selection and construction gates, source hashes, and all
negative outcomes. It excludes the GeoLife archive, caches, positions,
coordinates, and generated episodes.

## Claim boundary

This is a conditional model-population certificate for the fixed six-week
mobility mixture and modeled task/economic/response generator. It is not
evidence of real participant payment behavior, unseen-population mobility
generalization, or universal payment superiority. QIM-E-C and CSOPT-C remain
author-implemented common-protocol adaptations and are not part of R76.
