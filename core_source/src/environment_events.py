"""Environment events: shared state events applicable across mechanisms.

State-Reset: at t = T//2, exactly 50% of providers have their H_true
reset to 0, simulating platform migration / identity loss / history erasure.
The Simulator separately synchronizes the offer-side state when the event is
declared observable to the platform.

This is an ENVIRONMENT event — same reset_provider_ids for all mechanisms
sharing the same scenario+seed. Implemented as a standalone module to keep
the Simulator and pair_eval free of mechanism-specific reset logic.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def build_state_reset_ids(
    n_providers: int,
    fraction: float = 0.50,
    seed: int = 42,
    at_slot: int | None = None,
) -> np.ndarray:
    """Pre-compute the exact set of provider indices to reset.

    Uses exact rounding (round, not random threshold) for deterministic
    reproducibility.

    Args:
        n_providers: Total number of providers.
        fraction: Fraction to reset (default 0.50).
        seed: Seed for the selection RNG.
        at_slot: Slot at which reset occurs (for hashing only).

    Returns:
        Sorted array of provider indices selected for reset.  Empty if
        n_reset <= 0 or fraction <= 0.
    """
    rng = np.random.default_rng(seed)
    n_reset = int(round(fraction * n_providers))
    n_reset = max(0, min(n_reset, n_providers))
    if n_reset == 0:
        return np.array([], dtype=int)
    ids = rng.choice(np.arange(n_providers), size=n_reset, replace=False)
    ids.sort()
    return ids


def state_reset_event_hash(
    reset_provider_ids: np.ndarray,
    at_slot: int,
    fraction: float,
) -> str:
    """Deterministic hash of a reset event for cross-mechanism verification."""
    payload = {
        "at_slot": at_slot,
        "fraction": fraction,
        "n_reset": int(len(reset_provider_ids)),
        "ids_hash": hashlib.sha256(
            reset_provider_ids.tobytes()).hexdigest()[:16],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def apply_state_reset_if_due(
    slot: int,
    H_state: np.ndarray,
    reset_cfg: dict[str, Any] | None,
    reset_provider_ids: np.ndarray | None,
) -> dict:
    """Check and apply state reset at the designated slot.

    Args:
        slot: Current simulation slot.
        H_state: Mutable path-state array (H_true).
        reset_cfg: Dict with keys 'enabled', 'at_slot', 'fraction'.
        reset_provider_ids: Pre-computed reset ids.

    Returns:
        dict with keys: applied, n_reset, provider_ids, event_hash.
        applied=False if no reset was due this slot.
    """
    if reset_cfg is None:
        return {"applied": False, "n_reset": 0, "provider_ids": [],
                "event_hash": "", "reason": "no config"}

    enabled = bool(reset_cfg.get("enabled", False))
    at_slot = int(reset_cfg.get("at_slot", -1))
    fraction = float(reset_cfg.get("fraction", 0.0))

    if not enabled:
        return {"applied": False, "n_reset": 0, "provider_ids": [],
                "event_hash": "", "reason": "not enabled"}

    if slot != at_slot:
        return {"applied": False, "n_reset": 0, "provider_ids": [],
                "event_hash": "", "reason": f"slot {slot} != at_slot {at_slot}"}

    if reset_provider_ids is None or len(reset_provider_ids) == 0:
        return {"applied": False, "n_reset": 0, "provider_ids": [],
                "event_hash": "", "reason": "no reset provider ids"}

    # Apply reset
    H_state[reset_provider_ids] = 0.0
    event_hash = state_reset_event_hash(reset_provider_ids, at_slot, fraction)

    return {
        "applied": True,
        "n_reset": int(len(reset_provider_ids)),
        "provider_ids": reset_provider_ids.copy(),
        "event_hash": event_hash,
        "at_slot": at_slot,
        "fraction": fraction,
    }
