"""E28: fail-closed audit of the empirical origin of PASI variables.

This is an evidence audit, not a new performance experiment. It verifies the
schemas actually available in the public participant data and records whether
each manuscript variable is observed, derived, calibrated, modeled, or a
mechanism-computed outcome. It deliberately does not infer bids, private costs,
request arrivals, capacity, or payment response from completed-task records.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd


NETEASE_COLUMNS = [
    "tasksetId", "taskId", "workerId", "answer", "completeTime", "truth", "capability"
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extrasensory_columns(path: Path) -> list[str]:
    with gzip.open(path, "rt", newline="") as handle:
        return next(csv.reader(handle))


def build_matrix() -> pd.DataFrame:
    rows = [
        ("provider mobility", "GeoLife GPS trajectories", "observed", "positions and candidate graph", "does not observe economic behavior"),
        ("sensor/network availability", "ExtraSensory", "observed", "current device-state multipliers", "not participation acceptance or capacity"),
        ("completed-work time and correctness", "NetEaseCrowd", "observed among completions", "held-out response-audit sequence", "non-completions and allocation exposure are absent"),
        ("participant compensation scale", "TILES-2018 schedule", "published aggregate schedule", "post-hoc monetary sensitivity", "not bids, private costs, or per-service PASI payments"),
        ("market wage/charge scale", "Hunan wage, TLC, AMT, Prolific", "published external scale", "post-hoc unit conversions", "not MCS participant cost observations"),
        ("request arrivals and task locations", "PASI generator", "modeled", "frozen common event tape", "no public request trace is claimed"),
        ("private effort cost", "PASI cost model", "modeled", "contract evaluation", "not participant reported"),
        ("direct reserve bids in E27", "modeled economic state", "derived model input", "QIM-E and CSOPT direct-bid input", "not participant reported"),
        ("concurrent provider capacity", "PASI generator", "modeled", "matching constraint", "completion counts are not relabeled as capacity"),
        ("effort-response mapping", "PASI response model", "modeled", "contract construction and QoS audit", "participant traces perturb selected dynamics only"),
        ("QoS targets", "PASI request model", "modeled", "shared eligibility threshold", "not requester declarations from a field platform"),
        ("contract payment", "PASI/QIM-E/CSOPT algorithms", "mechanism-computed", "reported payment and PPS", "not an observed payment or causal payment response"),
    ]
    return pd.DataFrame(rows, columns=[
        "variable", "source", "evidence_status", "use_in_evaluation", "claim_boundary"
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extrasensory-dir", type=Path, required=True)
    parser.add_argument("--netease-csv", type=Path, required=True)
    parser.add_argument("--e25-metadata", type=Path, required=True)
    parser.add_argument("--e26-metadata", type=Path, required=True)
    parser.add_argument("--e27-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    extra_files = sorted(args.extrasensory_dir.glob("*.csv.gz"))
    if len(extra_files) != 60:
        raise AssertionError(f"expected 60 ExtraSensory files, found {len(extra_files)}")
    extra_cols = extrasensory_columns(extra_files[0])
    if "timestamp" not in extra_cols:
        raise AssertionError("ExtraSensory schema lacks timestamp")
    if not any(col.startswith("label:") for col in extra_cols):
        raise AssertionError("ExtraSensory schema lacks label columns")

    netease_cols = pd.read_csv(args.netease_csv, nrows=0).columns.tolist()
    if netease_cols != NETEASE_COLUMNS:
        raise AssertionError(f"unexpected NetEaseCrowd schema: {netease_cols}")

    e25 = json.loads(args.e25_metadata.read_text(encoding="utf-8"))
    e26 = json.loads(args.e26_metadata.read_text(encoding="utf-8"))
    e27 = json.loads(args.e27_metadata.read_text(encoding="utf-8"))
    expected_modeled = {"requests", "costs", "capacities", "contract response mapping", "payments"}
    if set(e25["model_generated_fields"]) != expected_modeled:
        raise AssertionError("E25 modeled-field disclosure changed")
    if set(e26["model_generated_fields"]) != expected_modeled:
        raise AssertionError("E26 modeled-field disclosure changed")
    if e27["payment_status"] != "mechanism-computed outcome, not an observed field":
        raise AssertionError("E27 payment-origin disclosure changed")

    matrix = build_matrix()
    prohibited = {
        "request arrivals and task locations",
        "private effort cost",
        "direct reserve bids in E27",
        "concurrent provider capacity",
        "effort-response mapping",
        "QoS targets",
        "contract payment",
    }
    if set(matrix.loc[matrix["evidence_status"].str.contains(
            "observed", case=False), "variable"]) & prohibited:
        raise AssertionError("an unobserved economic or request variable was mislabeled as observed")

    args.out.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.out / "e28_data_origin_matrix.csv", index=False)
    status_summary = (matrix.groupby("evidence_status", as_index=False)
                      .agg(variable_count=("variable", "size")))
    status_summary.to_csv(args.out / "e28_evidence_status_summary.csv", index=False)
    metadata = {
        "artifact": "E28 empirical-evidence origin audit",
        "artifact_type": "schema and claim audit; not a performance experiment",
        "result": "PASS",
        "extrasensory_file_count": len(extra_files),
        "extrasensory_schema_has_timestamp": "timestamp" in extra_cols,
        "extrasensory_schema_has_labels": any(col.startswith("label:") for col in extra_cols),
        "netease_columns": netease_cols,
        "netease_missing_for_economic_validation": [
            "request release", "allocation exposure", "non-completion", "bid",
            "private cost", "concurrent capacity", "payment", "payment response"
        ],
        "participant_evidence_supported": [
            "mobility", "device availability", "completed-work time and correctness"
        ],
        "economic_evidence_supported": [
            "published aggregate compensation and wage scales only"
        ],
        "unresolved_external-validity_gap": [
            "private costs", "bids", "capacity", "requests", "causal payment response"
        ],
        "raw_participant_data_redistributed": False,
        "input_hashes": {
            "e25_metadata": sha256(args.e25_metadata),
            "e26_metadata": sha256(args.e26_metadata),
            "e27_metadata": sha256(args.e27_metadata),
        },
        "script_hash": sha256(Path(__file__)),
    }
    (args.out / "e28_evidence_origin_audit.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(matrix.to_string(index=False))
    print("E28 evidence-origin audit: PASS")


if __name__ == "__main__":
    main()
