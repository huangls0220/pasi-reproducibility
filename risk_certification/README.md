# R75 independent risk certification

This directory implements the manuscript's calibration--freeze--held-out-test rule on complete modeled episodes. It does not treat assignments as independent samples and does not use held-out outcomes to select the guardrail weight.

## Scope

- Risk target: mean complete-episode QoS loss `V/max(A,1)` no greater than 0.005.
- Familywise error: 0.05 across six response scenarios and five predeclared guardrail weights.
- Calibration: 1,277 disjoint seeds, the zero-loss threshold required by the frozen bound.
- Held-out test: 1,000 disjoint seeds, constructed only after selection is written.
- Mobility: a fixed empirical mixture of six preregistered GeoLife weeks.
- Economics: requests, costs, bids, capacities, response, and payments remain modeled or mechanism-computed. The certificate does not cover real participant behavior or unseen mobility populations.

`protocol.md` and `preregistration.json` state the complete rule. `frozen_runner/` preserves the exact local-layout runner and protocol whose hashes appear in the formal results. `run_r75_public.py` changes only repository path resolution so the same logic can run from a clean public clone.

## Frozen result

All 38,310 calibration and 12,000 held-out executions completed. Calibration
selected `g=1.0` as the only qualifying candidate: its worst simultaneous
episode-risk upper bound was 0.004996816, whereas `g=0.75` reached
0.006651055. The held-out frozen policy recorded zero QoS violations in
5,125,500 assignments and mean service coverage 0.536459. Its simultaneous
upper bound was 0.005465648, so the 1,000 held-out episodes per cell do not
constitute a second 0.005 certificate; the corresponding all-zero threshold is
1,094. The predeclared `g=0.25` diagnostic recorded 3,051 violations and a
worst upper bound of 0.011326834 while raising mean coverage to 0.645934.

The statistical pass comes from calibration. Held-out outcomes were not used
to select or revise `g`.

## Restricted input

First reconstruct the six R73 windows using `cross_window/run_cross_window.py prepare` and data obtained through GeoLife's authorized access process. Keep the resulting position caches and all generated episodes outside this repository. The R75 runner rejects a restricted path located inside the public package.

## Ordered commands

Use fresh `RESTRICTED` and `RESULTS` directories. The action order enforces the test gate.

```powershell
python -B risk_certification/run_r75_public.py freeze --r73 C:\restricted\r73-windows --restricted C:\restricted\r75 --results C:\results\r75
python -B risk_certification/run_r75_public.py smoke --r73 C:\restricted\r73-windows --restricted C:\restricted\r75 --results C:\results\r75
python -B risk_certification/run_r75_public.py prepare-cal --r73 C:\restricted\r73-windows --restricted C:\restricted\r75 --results C:\results\r75
python -B risk_certification/run_r75_public.py calibrate --r73 C:\restricted\r73-windows --restricted C:\restricted\r75 --results C:\results\r75 --workers 16
python -B risk_certification/run_r75_public.py select --r73 C:\restricted\r73-windows --restricted C:\restricted\r75 --results C:\results\r75
python -B risk_certification/run_r75_public.py prepare-test --r73 C:\restricted\r73-windows --restricted C:\restricted\r75 --results C:\results\r75
python -B risk_certification/run_r75_public.py test --r73 C:\restricted\r73-windows --restricted C:\restricted\r75 --results C:\results\r75 --workers 16
python -B risk_certification/run_r75_public.py summarize --r73 C:\restricted\r73-windows --restricted C:\restricted\r75 --results C:\results\r75
```

The top-level simulator status may be `failed` when an observed QoS violation occurs even though computation completed. Such rows remain in the risk sample. Only `execution_status != completed`, a missing cell, a changed frozen hash, an out-of-envelope response, or invalid source metadata constitutes an execution failure.

## Public outputs

The public repository includes the frozen selection, cell summaries, source
hashes, completion record, and the exact runner that regenerates the
per-execution tables.  The complete non-sensitive per-execution CSV files are
retained in the round archive rather than duplicated in the repository.  The
release excludes the GeoLife archive, caches, episodes, coordinates, and
provider/task/slot logs.
