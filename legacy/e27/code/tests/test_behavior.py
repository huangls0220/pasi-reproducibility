"""Tests for behavior.py — Prelec probability weighting (Section 5.7)."""

import math

import pytest

from src.behavior import W, W_inv, W_prime


class TestPrelecW:
    def test_boundary_zero(self):
        assert W(0.0, 0.8) == 0.0

    def test_boundary_one(self):
        assert W(1.0, 0.8) == 1.0

    def test_monotonic(self):
        vals = [W(p, 0.8) for p in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1], f"W not monotonic at {i}: {vals}"

    def test_zeta_one_is_identity(self):
        """At zeta=1, W(p) should equal p."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert math.isclose(W(p, 1.0), p, rel_tol=1e-10)

    def test_zeta_lt_one_inverts(self):
        """zeta<1 gives overweighting of small probabilities."""
        w = W(0.1, 0.65)
        assert w > 0.1, f"Expected overweighting, got W(0.1)={w}"

    def test_zeta_gt_one(self):
        """zeta>1 can give underweighting."""
        w = W(0.5, 1.5)
        # Should still be valid
        assert 0 <= w <= 1

    def test_range(self):
        for zeta in [0.5, 0.8, 1.0, 1.5]:
            for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
                w = W(p, zeta)
                assert 0 <= w <= 1, f"W({p}, {zeta})={w} out of range"


class TestPrelecWInv:
    def test_roundtrip(self):
        """W_inv(W(p)) == p."""
        for zeta in [0.6, 0.8, 1.0, 1.2]:
            for p in [0.05, 0.2, 0.5, 0.8, 0.95]:
                w = W(p, zeta)
                p2 = W_inv(w, zeta)
                assert math.isclose(p, p2, rel_tol=1e-9), (
                    f"Roundtrip failed at p={p}, zeta={zeta}: W={w}, W_inv={p2}"
                )

    def test_boundary_zero(self):
        assert W_inv(0.0, 0.8) == 0.0

    def test_boundary_one(self):
        assert W_inv(1.0, 0.8) == 1.0


class TestPrelecWPrime:
    def test_positive(self):
        for p in [0.1, 0.5, 0.9]:
            assert W_prime(p, 0.8) > 0
