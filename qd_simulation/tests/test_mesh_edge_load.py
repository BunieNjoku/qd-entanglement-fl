"""Tests for the two structural facts the mesh scaling model rests on.

Masked cell submission puts at most three masks per round on any one
physical edge, whatever the mesh size, so entangled-key supply sets a
floor on the round time but does not bound the mesh. And the cell count
follows Euler with a logarithmic hull term, which is what lets a cell
budget be turned back into a node count.

Plain-python asserts, no test framework (the repo has none). Run:

    python tests/test_mesh_edge_load.py
"""
import math
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OPERATING_POINT  # noqa: E402
from modules.m3_mesh_topology import (  # noqa: E402
    MAX_MASKS_PER_EDGE_PER_ROUND, analyze_mesh, compute_max_mesh_size,
    nodes_for_cell_count)
from modules.m4_federated_learning import topology_from_mesh  # noqa: E402

RATE = OPERATING_POINT["secret_rate_bps"]


def _edge_load(n, seed):
    """Masks a round puts on each physical edge: two per adjacent cell
    plus one per inter-cell mask keyed on that shared side."""
    m = analyze_mesh(n, 1e5, RATE, rng=np.random.default_rng(seed))
    tri, tree = topology_from_mesh(m)
    tri = [tuple(int(x) for x in t) for t in tri]
    load = {}
    for t in tri:
        for i, j in ((0, 1), (1, 2), (0, 2)):
            e = tuple(sorted((t[i], t[j])))
            load[e] = load.get(e, 0) + 1
    for parent, child in tree:
        shared = sorted(set(tri[int(parent)]) & set(tri[int(child)]))
        assert len(shared) >= 2, "inter-cell mask has no shared side to key on"
        e = (int(shared[0]), int(shared[1]))
        load[e] = load.get(e, 0) + 1
    return max(load.values())


def test_edge_load_is_bounded_and_size_independent():
    """At most three masks per edge per round, at every scale."""
    worst = 0
    for n in (6, 9, 12, 24, 40, 50):
        for seed in range(10):
            worst = max(worst, _edge_load(n, seed))
    assert worst == MAX_MASKS_PER_EDGE_PER_ROUND, (
        f"busiest edge carries {worst} masks, not "
        f"{MAX_MASKS_PER_EDGE_PER_ROUND}; the key-floor argument in "
        "compute_max_mesh_size assumes the bound")
    print(f"  busiest edge carries {worst} masks at every scale: OK")


def test_hull_follows_renyi_sulanke():
    """E[h] = (2k/3) ln n for a convex k-gon; the campus is a square."""
    ns = [6, 12, 24, 50, 120, 200]
    meas = np.array([
        np.mean([len(ConvexHull(
            analyze_mesh(n, 1e5, RATE, rng=np.random.default_rng(s)
                         ).node_positions).vertices) for s in range(20)])
        for n in ns])
    law = (8.0 / 3.0) * np.log(ns)
    rms = float(np.sqrt(np.mean((meas - law) ** 2)))
    assert rms < 0.5, (
        f"hull deviates from (8/3)ln n by RMS {rms:.2f} vertices; "
        "nodes_for_cell_count assumes that law")
    print(f"  hull matches (8/3)ln n to RMS {rms:.2f} vertices, "
          "no free parameter: OK")


def test_cell_count_inverts_to_node_count():
    """Generated meshes round-trip through the Euler inversion."""
    for n in (12, 24, 50, 120):
        cells = analyze_mesh(n, 1e5, RATE,
                             rng=np.random.default_rng(0)).n_triangles
        back = nodes_for_cell_count(cells)
        assert abs(back - n) <= 2, \
            f"n={n} -> {cells} cells -> inverted to {back}"
    print("  cell count inverts to node count within 2: OK")


def test_key_floor_reports_infeasible_rather_than_clamping():
    """A round shorter than the key floor supports no mesh at all."""
    bits = 28866 * 32
    floor = MAX_MASKS_PER_EDGE_PER_ROUND * bits / RATE
    cells, nodes = compute_max_mesh_size(28866, floor * 0.5, RATE)
    assert (cells, nodes) == (0, 0), \
        f"round below the {floor:.0f}s key floor returned {cells} cells"
    cells, nodes = compute_max_mesh_size(28866, floor * 2.0, RATE)
    assert cells > 0 and nodes >= 3, \
        "round above the key floor should support some mesh"
    print(f"  key floor {floor:.0f}s reported as infeasible, not clamped: OK")


if __name__ == "__main__":
    print("Mesh edge load and scaling")
    test_edge_load_is_bounded_and_size_independent()
    test_hull_follows_renyi_sulanke()
    test_cell_count_inverts_to_node_count()
    test_key_floor_reports_infeasible_rather_than_clamping()
    print("All mesh scaling tests passed.")
