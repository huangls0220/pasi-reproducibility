# PASI reproducibility materials

This repository accompanies the manuscript **Path-Aware Stateful Incentives for
Long-Term Mobile Crowdsensing Services**. The current synthetic experiments use
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

E27 is not evidence of universal payment superiority. QIM-E and CSOPT have
lower payment in all 90 tested restricted-subset comparisons, under different
information, QoS, contract-domain, and objective assumptions.

## Environment and current synthetic runs

Reference environment: Windows x64 and Python 3.14.4. The resolved Windows
dependency lock is `core_source/requirements-win-py314-lock.txt`.

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
difference of (1.46\times10^{-11}) and zero assignment difference across
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
checks the legally reconstructed episode hashes before replay. E5, E7, and E16
retain their exact runners and preregistrations under `evidence_runners`, but
local absolute paths must be replaced with the reader's legally reconstructed
input paths. Their seed-level aggregate outcomes are supplied under `results`.

## Integrity and privacy

`manifest.json` gives the SHA-256 and byte size of every included payload file
(the manifest cannot hash itself).
`privacy_audit.json` records the allow-list result. Negative outcomes are kept;
the public package excludes restricted data rather than substituting synthetic
evidence for it.

No software license has been granted in this version. Repository visibility
permits inspection but does not itself grant permission to reuse the code.
No DOI is claimed.
