"""Tests for inter-cell masking in run_multitriangle_fl.

Overlapping cells make the per-cell sums a linear system in the
individual client updates: with enough cells the system has full rank
and every update is solvable from the sums alone, with no collusion and
no broken mask. The aggregator must therefore never observe a cell sum,
only the total in which the inter-cell masks cancel.

These tests check the construction that prevents it: masks on the edges
of a spanning tree over the cell dual, applied with opposite signs at
the two endpoints, so every submitted cell vector is uniform and the
masks vanish from the sum.

Plain-python asserts, no test framework (the repo has none). Run:

    python tests/test_cell_sum_privacy.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OPERATING_POINT  # noqa: E402
from modules.m3_mesh_topology import analyze_mesh  # noqa: E402
from modules.m4_federated_learning import (  # noqa: E402
    apply_modular_mask, derive_modular_mask, quantize_gradient,
    topology_from_mesh)

RATE = OPERATING_POINT["secret_rate_bps"]


def _mesh(n):
    m = analyze_mesh(n, 1e5, RATE, rng=np.random.default_rng(42))
    tri, tree = topology_from_mesh(m)
    return ([tuple(int(x) for x in t) for t in tri],
            [(int(a), int(b)) for a, b in tree])


def test_cell_sums_are_a_solvable_system_without_masking():
    """The motivation: unmasked cell sums leak individual updates."""
    for n, expect_full in ((12, True), (24, True)):
        tri, _ = _mesh(n)
        memb = {}
        for t in tri:
            for c in t:
                memb[c] = memb.get(c, 0) + 1
        M = np.zeros((len(tri), n))
        for i, t in enumerate(tri):
            for c in t:
                M[i, c] = 1.0 / memb[c]
        rank = np.linalg.matrix_rank(M)
        assert (rank == n) == expect_full, (
            f"n={n}: rank {rank} of {n}; the leak this module guards "
            "against has changed shape, so revisit the masking argument")
    print("  unmasked cell sums are full rank at n=12 and n=24: OK")


def test_masks_cancel_in_the_total():
    """Masked submissions sum to the same total as the plain cell sums."""
    for n in (6, 9, 12):
        tri, tree = _mesh(n)
        rng = np.random.default_rng(3)
        d = 64
        sums = [torch.tensor(rng.normal(size=d), dtype=torch.float32)
                for _ in tri]
        q = {i: quantize_gradient(sums[i]) for i in range(len(tri))}
        plain = np.zeros(d, dtype=np.uint32)
        for i in range(len(tri)):
            plain = (plain + q[i]).astype(np.uint32)
        for parent, child in reversed(tree):
            mask = derive_modular_mask(
                d, rng.integers(0, 2, size=d * 32, dtype=np.int8))
            q[child] = apply_modular_mask(q[child], mask, sign=+1)
            q[parent] = apply_modular_mask(q[parent], mask, sign=-1)
        masked = np.zeros(d, dtype=np.uint32)
        for i in range(len(tri)):
            masked = (masked + q[i]).astype(np.uint32)
        assert np.array_equal(masked, plain), \
            f"n={n}: inter-cell masks did not cancel in the total"
    print("  masks cancel exactly in the total at n=6, 9, 12: OK")


def test_every_submitted_cell_vector_is_masked():
    """No cell may be submitted in the clear: a spanning tree gives every
    cell degree >= 1, so every submission carries a fresh mask."""
    for n in (6, 9, 12, 24):
        tri, tree = _mesh(n)
        assert len(tree) == len(tri) - 1, (
            f"n={n}: {len(tree)} mask edges over {len(tri)} cells is not a "
            "spanning tree; cancellation needs a connected mask graph")
        touched = set()
        for parent, child in tree:
            touched.add(parent)
            touched.add(child)
        assert touched == set(range(len(tri))), (
            f"n={n}: cells {set(range(len(tri))) - touched} carry no mask "
            "and would be submitted in the clear")
    print("  every cell carries at least one mask at n=6, 9, 12, 24: OK")


if __name__ == "__main__":
    print("Inter-cell masking")
    test_cell_sums_are_a_solvable_system_without_masking()
    test_masks_cancel_in_the_total()
    test_every_submitted_cell_vector_is_masked()
    print("All inter-cell masking tests passed.")
