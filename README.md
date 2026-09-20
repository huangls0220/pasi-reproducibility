# PASI reproducibility materials

This repository accompanies the manuscript **PASI: QoS-Safe Incentive
Contracting under Runtime-Design Path-State Mismatch in Mobile
Crowdsensing**. The current synthetic experiments use
one formal protocol: the repaired implementation, lexicographic
coverage-first matching, minimization of the design-side decision fee at fixed
coverage, and separate reporting of execution settlement.

The materials support inspection and reconstruction; they do not establish
observations of real bidding, cost, capacity, effort, or payment response.
GeoLife contributes mobility and availability only. The economic variables and
responses remain model-generated.

## Evidence map

| Evidence | Location | Interpretation |
| --- | --- | --- |
| Main-180 | `results/main180` | Current canonical protocol, 180 paired runs |
| E1 path-state formation | `results/e1` | Current canonical protocol, 180 runs |
| Scale | `results/scale` | Current canonical protocol, 80 runs |
| Observable State-Reset | `results/state_reset` | Current canonical protocol, 40 runs |
| E2 component ablations | `results/e2` | Current canonical protocol, 450 runs |
| E4 migration diagnostic | `results/e4_diagnostic` | One repeat in 24 seed-001 cells; not a performance benchmark |
| E5 GeoLife | `results/e5` | Trace-driven public dynamics; restricted inputs omitted |
| E7 directional mismatch | `results/e7` | Trace-driven public dynamics; restricted inputs omitted |
| E16 contract comparison | `results/e16` | Coverage-first comparison; restricted inputs omitted |
| Response-safety study | `legacy/response_safety` | Separate frozen repaired implementation and replay wrapper |
| E27 public mechanisms | `legacy/e27` | Historical implementation retained separately; conditional equivalence, not a current-core rerun |
| R73 temporal-window response safety | `cross_window` | Six distinct GeoLife weeks, three modeled seeds/week; descriptive safety--coverage robustness, not risk certification |

E27 is not evidence of universal payment superiority. QIM-E and CSOPT have
lower payment in all 90 tested restricted-subset comparisons, under different
information, QoS, contract-domain, and objective assumptions.

## Environment and current synthetic runs

Reference environment: Windows x64 and Python 3.14.4. The resolved Windows
dependency lock is `core_source/requirements-win-py314-lock.txt`.
`core_source/scripts` includes historical utilities needed by the frozen test
suite; their presence does not certify retired experiments as current evidence.
All 241 source tests passed in a clean copy of this public artifact on the
reference machine; see `test_validation.json` for the exact command and scope.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r core_source/requirements-win-py314-lock.txt
.venv/Scripts/python -m pip check
.venv/Scripts/python -B -m pytest core_source/tests -q -p no:cacheprovider --basetemp .test-tmp

# Run in a disposable clone because these commands regenerate result folders.
.venv/Scripts/python run_e1_current180.py --workers 4
.venv/Scripts/python run_scale_current80.py --workers 4
.venv/Scripts/python run_state_reset_current40.py --workers 4
.venv/Scripts/python run_e2_current450.py --workers 4
.venv/Scripts/python analyse_e2_current450.py
```

`run_main180_coverage_first.py` is preserved as the exact audit runner. Its
preparation stage records the original local archive path and is therefore not
a portable entry point. `run_e1_current180.py` regenerates the same 180
scenario--method--seed cells under the same canonical core and additionally
records phase diagnostics. The frozen outputs and hashes remain in
`results/main180`; `main_e1_equivalence.json` reports a maximum monetary
difference of 1.46e-11 and zero assignment difference across
the 90 paired cells.

`run_e4_seed001_diagnostic.py` is an information-value diagnostic for an old
timing claim. It is not a replacement for the historical 360 timed executions
and must not be cited as a performance benchmark.

## GeoLife reconstruction boundary

Obtain GeoLife GPS Trajectories 1.3 through the dataset owner's authorized
access procedure and accept its terms. This repository deliberately excludes
the original archive, position cache, frozen episodes, coordinates, and
provider/task/slot logs. The expected original archive SHA-256 is
`1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6`;
the frozen position-cache SHA-256 is
`ab4ce644cbd4ce689f7f829710933a619e38c121dc9054af72e0b175ecbedb80`.

The response-safety reconstruction wrapper is namespaced under
`legacy/response_safety`. It rejects generated data inside the repository and
checks the legally reconstructed episode hashes before replay. Its published
reference covers formal seeds 2--30; seed 1 has no row in that reference and
cannot count as a comparison. E5, E7, and E16 retain their historical audit
runners and preregistrations under `evidence_runners`. Those archival runners
refer to local snapshots. New portable entry points below replay the published
scientific outcome columns using the archived repaired core and frozen input
hashes; they do not reproduce private candidate logs or historical runtimes.

From a clean clone, after installing the Windows Python 3.14 environment,
use GeoLife data obtained under the owner's terms. The first command rebuilds
all 90 low/medium/high episodes and verifies all 360 file hashes against the
published E5 preregistration. Keep `RESTRICTED` outside the clone. For a short
comparison, use seed 1; pass `--seeds 1 2 ... 30` for the complete E5/E16
matrix one workload at a time. E7 accepts one published cell ID per call.

```powershell
python -B rebuild_geolife_all.py --raw "C:\path\to\Geolife Trajectories 1.3\Data" --out "C:\restricted\pasi-episodes"
python -B replay_e5_e16.py --study e5 --episodes "C:\restricted\pasi-episodes\episodes" --out "C:\restricted\e5-medium-seed1.json" --workload medium --seeds 1
python -B replay_e5_e16.py --study e16 --episodes "C:\restricted\pasi-episodes\episodes" --out "C:\restricted\e16-medium-seed1.json" --workload medium --seeds 1
python -B replay_e7.py --episodes "C:\restricted\pasi-episodes\episodes" --out "C:\restricted\e7-C002-seed1.json" --cell E7C002 --seeds 1
```

R73's separate cross-week design, commands, aggregate seed outcomes, and
negative findings are in `cross_window/README.md`. It does not reuse the
same physical week 30 times as if those were independent mobility episodes.
`replay_validation.json` records a clean local reconstruction of all 90
GeoLife episodes (360 file hashes matched) and 272 selected scientific
outcome comparisons across E5, E7, E16, and R56. This is a verified
reproduction entry point, **not** a claim that all historical simulation
cells were rerun in the clean validation.

## Integrity and privacy

`manifest.json` gives the SHA-256 and byte size of every included payload file
(the manifest cannot hash itself).
`privacy_audit.json` records the allow-list result. Negative outcomes are kept;
the public package excludes restricted data rather than substituting synthetic
evidence for it.

No software license has been granted in this version. Repository visibility
permits inspection but does not itself grant permission to reuse the code.
No DOI is claimed.
