"""Tests for inter-triangle tree-mask key assignment in m4.

A Delaunay dual only ever joins triangles that share a whole side, so the
cross-link fallback in run_multitriangle_fl is unreachable from any
experiment's topology. These tests drive it directly on a vertex-joined
mesh and confirm the shared-side case still takes precedence.

Plain-python asserts, no test framework (the repo has none). Run:

    python tests/test_tree_mask_keying.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.m4_federated_learning import (  # noqa: E402
    ClinicalMLP, load_clinical_data, run_multitriangle_fl)

# T0 and T1 share the side (1,2); T1 and T2 meet only at vertex 3. The
# second join has no shared side, so its tree mask must fall back to a
# physical cross link.
VERTEX_JOINED = [(0, 1, 2), (1, 2, 3), (3, 4, 5)]
VERTEX_TREE = [(0, 1), (1, 2)]

# Every join shares a side - the Delaunay case.
SIDE_JOINED = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
SIDE_TREE = [(0, 1), (1, 2)]


def _run(triangles, tree_edges, n_clients=6, rounds=3):
    data, test, idim = load_clinical_data(n_clients, rng_seed=42,
                                          dirichlet_alpha=0.5)
    res, _ = run_multitriangle_fl(
        lambda: ClinicalMLP(input_dim=idim, hidden=128),
        data, test, num_clients=n_clients, num_rounds=rounds,
        local_epochs=1, batch_size=32, lr=0.01, target_accuracy=0.99,
        triangles=triangles, tree_edges=tree_edges,
        dropout_rate=0.0, verbose=False, seed=42)
    return res


def test_vertex_join_keys_on_a_physical_cross_link():
    """A vertex-only join keys its tree mask on a real triangle side."""
    res = _run(VERTEX_JOINED, VERTEX_TREE)
    physical = {tuple(sorted((t[i], t[j])))
                for t in VERTEX_JOINED for i, j in ((0, 1), (1, 2), (0, 2))}
    keyed = set(res.comm_rounds[-1].key_bits_per_edge)
    assert keyed <= physical, f"tree mask keyed off-mesh: {keyed - physical}"
    # The (1,2) side carries two intra masks plus the T0-T1 tree mask; the
    # fallback join must land somewhere else, on a side of T1 or T2.
    assert (3, 4) in keyed or (3, 5) in keyed or (1, 3) in keyed \
        or (2, 3) in keyed, "fallback did not key a cross link at vertex 3"
    print("  vertex-only join keys on a physical cross link: OK")


def test_side_join_prefers_the_shared_side():
    """With a shared side available the fallback must not be taken."""
    res = _run(SIDE_JOINED, SIDE_TREE)
    keyed = res.comm_rounds[-1].key_bits_per_edge
    for parent, child in SIDE_TREE:
        shared = sorted(set(SIDE_JOINED[parent]) & set(SIDE_JOINED[child]))
        assert tuple(shared[:2]) in keyed, \
            f"shared side {tuple(shared[:2])} carries no tree-mask demand"
    print("  shared-side join keys on the shared side: OK")


def test_no_physical_cross_link_is_rejected():
    """Disjoint triangles cannot carry a single-hop tree mask."""
    disjoint = [(0, 1, 2), (3, 4, 5)]
    try:
        _run(disjoint, [(0, 1)])
    except ValueError as exc:
        assert "no physical cross link" in str(exc), str(exc)
        print("  disjoint join rejected: OK")
        return
    raise AssertionError("disjoint triangles should not key a tree mask")


if __name__ == "__main__":
    print("Tree-mask key assignment")
    test_vertex_join_keys_on_a_physical_cross_link()
    test_side_join_prefers_the_shared_side()
    test_no_physical_cross_link_is_rejected()
    print("All tree-mask keying tests passed.")
