# PASI — local reproducibility release candidate

Not yet published. No public repository, DOI, or code license is claimed.
This candidate does not yet cover every retained experiment in the manuscript.

## Evidence-to-code mapping

| Material | Code | Status |
| --- | --- | --- |
| Current six-row safety table: R56, 174 runs, seeds 002–030 | `current/code` | Frozen repaired implementation, copied without source changes |
| E27: 90 paired public-mechanism comparisons | `legacy/e27/code` | Original frozen implementation; conditional mathematical equivalence, not a new-core floating-point rerun |
| Main/E5/E7/E16 and E4 timing | Not yet included as current-core evidence | Compatibility remains unresolved; do not attribute old results to this repaired version |

Scripts retained inside each code snapshot include historical runners. Their
presence does not restore removed experiments or certify their old conclusions.
Use the entry points below. The current data are trace-driven model experiments,
not measurements of real bidding, cost, effort, or payment response.

## Environment and checks

Reference environment: Windows x64, Python 3.14.4. Resolved dependency pins are
in `current/code/requirements-win-py314-lock.txt`. Other platforms are untested.
From the candidate root:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r current/code/requirements-win-py314-lock.txt
.venv/Scripts/python -m pip check
.venv/Scripts/python reproduce.py verify-package
Push-Location current/code
../../.venv/Scripts/python -B -m pytest tests -q -p no:cacheprovider --basetemp ../../../pasi-test-temp
Pop-Location
```

## GeoLife reconstruction

Obtain GeoLife 1.3 through the dataset owner's authorized access procedure and
accept its terms yourself. No downloader or mirror is supplied. Raw traces,
position caches, derived episodes, and provider/task-level logs must remain
outside this repository and must not enter an Overleaf or public ZIP.
Original archive SHA-256:
`1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6`.

The `--raw` directory contains numbered user folders with `Trajectory/*.plt`.
The frozen window is February 14–21, 2009 UTC, in ten-minute slots.
Choose a new output directory outside this repository:

```powershell
python reproduce.py rebuild --raw D:/private-geolife/Data --out D:/private-geolife/rebuilt-r56
python reproduce.py check-inputs --episodes D:/private-geolife/rebuilt-r56/episodes/medium
```

The rebuild checks original cache and episode hashes; it needs no private
reference copy. If hashes differ, stop and retain the mismatch. `--cache` may
replace `--raw` when the legally held frozen position cache is already available.
This reconstruction is not a new scientific experiment.

## Current outcome replay

```powershell
python reproduce.py replay --episodes D:/private-geolife/rebuilt-r56/episodes/medium --seeds 2 --out D:/pasi-check/seed002
python reproduce.py replay --episodes D:/private-geolife/rebuilt-r56/episodes/medium --out D:/pasi-check/formal-174
```

The second command uses only the original 29 formal seeds, three scenarios and
two profiles. It saves all outcomes, including negative QoS/IR results, and
compares numerical outcome columns against the frozen seed CSV. It is a new
portable outcome wrapper, not the original audit-instrumented runner. It does
not reproduce old timing, shadow-information checks, or MILP certificates.
It has not yet been end-to-end validated by rerunning the simulation.
Input checks and package tests do not establish successful outcome replay.

## Historical E27

Use a separate process so that the legacy `src` cannot mix with the repaired
one. With legally reconstructed low/medium/high episodes:

```powershell
python legacy/e27/code/scripts/run_e27_public_payment_baselines.py --episodes D:/private-geolife/episodes --seeds 30 --output D:/pasi-check/e27
```

The supplied E27 results remain historical. Reproducing them does not establish
universal payment superiority; the public mechanisms have lower payments in
all 90 tested restricted-subset comparisons.

## Before public release

Close the unresolved evidence mappings; validate a clean outcome replay;
confirm the rights holders and author-owned code license; retain any applicable
third-party notices; then publish a versioned GitHub release and archive that
release with Zenodo. Only add a repository/DOI to the manuscript after anonymous
access has been verified. No raw or derived GeoLife data may be included.

Official workflow references:
[GitHub repository creation](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository),
[GitHub license instructions](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository),
[Zenodo GitHub release archiving](https://help.zenodo.org/docs/github/archive-software/github-upload/).
