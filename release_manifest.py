"""Build/check the content manifest and restricted-data boundary for this text-only artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
FORBIDDEN_COMPONENTS = {"raw", "restricted", "cache", "caches", "episodes", "cells", "participant_logs", "provider_logs", "task_logs", "slot_logs", ".git", ".venv", "__pycache__"}
FORBIDDEN_EXTENSIONS = {".parquet", ".plt", ".zip", ".pyc", ".pkl", ".pickle"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory() -> tuple[list[dict], list[str]]:
    rows, violations = [], []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if path.is_symlink() or any(part.lower() in FORBIDDEN_COMPONENTS for part in relative.parts) or path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            violations.append(relative.as_posix())
            continue
        rows.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    return rows, violations


def main(write: bool) -> None:
    rows, violations = inventory()
    if violations:
        raise ValueError(f"Restricted or unsafe package paths: {violations}")
    manifest = {"schema": 1, "files": rows}
    if write:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if old != manifest:
            before = {item["path"] for item in old["files"]}
            after = {item["path"] for item in rows}
            raise ValueError({"added": sorted(after - before), "removed": sorted(before - after), "changed": sorted(name for name in before & after if next(x for x in old["files"] if x["path"] == name) != next(x for x in rows if x["path"] == name))})
    print(json.dumps({"files": len(rows), "restricted_paths": 0, "manifest_sha256": sha(MANIFEST), "passed": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate after deliberate release edits")
    main(parser.parse_args().write)
