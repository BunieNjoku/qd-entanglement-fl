"""
Module 3: Mesh topology and aggregation scheduling.

Generates overlapping 3-clique (triangle) meshes from a Delaunay
triangulation of the campus, constructs an aggregation tree over the
triangle dual graph, and computes per-level latencies under the
shared-bit-rate budget. Triangles are the minimum geometric unit
providing complete pairwise coverage for 3-client Bonawitz masking.
"""
import math

import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MeshResult:
    """Results from mesh topology analysis."""
    n_nodes: int
    n_triangles: int
    n_links: int
    aggregation_depth: int
    aggregation_tree: nx.DiGraph
    node_positions: np.ndarray
    triangles: List[Tuple[int, int, int]]
    per_level_latency_s: List[float]
    total_aggregation_time_s: float
    randomness_utilization: float
    # Secret rate of each physical edge at its own length and its
    # endpoints' degrees; keys are (i, j) with i < j.
    edge_secret_rates: dict = field(default_factory=dict)
    node_degrees: dict = field(default_factory=dict)


def generate_campus_mesh(n_nodes, campus_size_m=2000, rng=None):
    """Generate a campus layout with Delaunay triangulation.

    Args:
        n_nodes: Number of nodes to place.
        campus_size_m: Campus area side length.
        rng: Random generator.

    Returns:
        positions: (n_nodes, 2) array of node positions.
        triangles: List of (i, j, k) triangle vertex indices.
        graph: NetworkX graph with edges.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Place nodes with minimum distance constraint
    positions = np.zeros((n_nodes, 2))
    min_dist = campus_size_m / (2 * np.sqrt(n_nodes))

    positions[0] = rng.uniform(0, campus_size_m, 2)
    for i in range(1, n_nodes):
        for attempt in range(1000):
            candidate = rng.uniform(0, campus_size_m, 2)
            dists = np.linalg.norm(positions[:i] - candidate, axis=1)
            if np.all(dists > min_dist):
                positions[i] = candidate
                break
        else:
            positions[i] = rng.uniform(0, campus_size_m, 2)

    if n_nodes < 3:
        return positions, [], nx.Graph()

    # Delaunay triangulation
    tri = Delaunay(positions)
    triangles = [tuple(sorted(simplex)) for simplex in tri.simplices]

    # Build graph
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i, pos=positions[i])
    for t in triangles:
        for a, b in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]:
            dist = np.linalg.norm(positions[a] - positions[b])
            G.add_edge(a, b, distance=dist)

    return positions, triangles, G


def build_aggregation_tree(triangles, root_idx=0):
    """Build a tree of triangles for hierarchical aggregation.

    Constructs a dual graph (triangles as nodes, shared edges as connections),
    then builds a BFS spanning tree.

    Returns:
        tree: nx.DiGraph with triangle indices as nodes.
        depth: Tree depth.
    """
    if len(triangles) <= 1:
        tree = nx.DiGraph()
        if len(triangles) == 1:
            tree.add_node(0)
        return tree, 0

    # Build dual graph
    dual = nx.Graph()
    for i in range(len(triangles)):
        dual.add_node(i)

    # Two triangles are adjacent if they share an edge (2 vertices)
    for i in range(len(triangles)):
        for j in range(i + 1, len(triangles)):
            shared = set(triangles[i]) & set(triangles[j])
            if len(shared) >= 2:
                dual.add_edge(i, j)

    # BFS spanning tree from root
    if not dual.has_node(root_idx):
        root_idx = 0

    tree = nx.DiGraph()
    visited = {root_idx}
    queue = [root_idx]
    tree.add_node(root_idx)

    while queue:
        current = queue.pop(0)
        for neighbor in dual.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                tree.add_edge(current, neighbor)
                queue.append(neighbor)

    # Handle disconnected components
    for node in dual.nodes():
        if node not in visited:
            tree.add_node(node)

    depth = nx.dag_longest_path_length(tree) if len(tree) > 0 else 0
    return tree, depth


# Masks a single physical edge must supply per round. An edge of a
# planar triangulation borders at most two triangles, contributing at
# most two intra-cell masks. The inter-cell mask graph is a spanning
# tree of the cell dual and two adjacent cells share exactly one side,
# so a given edge carries at most one inter-cell mask. The bound is
# therefore three, is attained on interior edges whose dual edge is in
# the spanning tree, and does not grow with the mesh. Asserted over
# generated meshes in tests/test_mesh_edge_load.py.
MAX_MASKS_PER_EDGE_PER_ROUND = 3


def compute_aggregation_latency(
    n_triangles, gradient_bits, secure_key_rate_bps,
    fpga_latency_s=0.005
):
    """Round latency of one masked aggregation.

    Cells submit their masked sums together and the aggregator forms one
    total, so aggregation is a single communication round whatever the
    dual's shape; tree depth sets which cells share a mask, not how many
    times a value is relayed.

    Key wait is set by the busiest physical edge, which supplies at most
    MAX_MASKS_PER_EDGE_PER_ROUND masks per round at any mesh size.

    Args:
        n_triangles: Number of cells in the mesh.
        gradient_bits: Gradient size in bits.
        secure_key_rate_bps: Final secure key rate per link.
        fpga_latency_s: FPGA processing latency (Assumption C).

    Returns:
        per_level_latencies: Single-element list, kept as a list so
            callers that reported per-level timings still work.
        total_time: Round latency.
    """
    key_wait = (MAX_MASKS_PER_EDGE_PER_ROUND * gradient_bits
                / max(secure_key_rate_bps, 1))
    try:
        from config import CLASSICAL_LINK_BPS
    except ImportError:
        from ..config import CLASSICAL_LINK_BPS
    # Every cell submits one masked vector to the aggregator's single
    # access port, so the classical leg scales with the cell count.
    communication_time = n_triangles * gradient_bits / CLASSICAL_LINK_BPS
    total_time = key_wait + fpga_latency_s + communication_time

    return [total_time], total_time


def per_edge_secret_rates(positions, links, degrees):
    """Secret-key rate of every physical edge, at its own length and load.

    Both endpoints host a source and each splits its pairs across the
    edges incident to it, so an edge (i, j) of length L receives
    R(L)/deg(i) from one end and R(L)/deg(j) from the other, where R is
    the unmultiplexed source-link coincidence rate. In a lone triangle
    both degrees are 2 and this returns R(L), which is the
    single-rate value config.OPERATING_POINT carries; in a mesh the
    well-connected nodes split further and their edges run slower.

    QBER is length-independent in the analytic model
    (m1.compute_qber_analytical depends on Bell weight, PBS extinction
    and the birefringence allowance, not on loss), so only the raw rate
    varies across edges.

    Args:
        positions: (n, 2) node coordinates in metres.
        links: iterable of (i, j) node-index pairs.
        degrees: mapping node -> number of incident edges.

    Returns:
        dict mapping (i, j) with i < j to its final secure rate in bit/s.
    """
    try:
        from config import QUANTUM, _rate_waterfall_at
        from modules.m1_quantum_source import (
            compute_coincidence_rate, compute_qber_analytical)
    except ImportError:
        from ..config import QUANTUM, _rate_waterfall_at
        from .m1_quantum_source import (
            compute_coincidence_rate, compute_qber_analytical)

    qber = compute_qber_analytical(
        QUANTUM['fidelity'], pbs_extinction=QUANTUM['pbs_extinction_ratio'],
        bire_contribution=QUANTUM['analytic_qber_birefringence_allowance'],
    )
    rates = {}
    for a, b in links:
        a, b = int(a), int(b)
        key = (a, b) if a < b else (b, a)
        length_m = float(np.linalg.norm(positions[a] - positions[b]))
        per_source = compute_coincidence_rate(
            QUANTUM['source_rep_rate_hz'], QUANTUM['source_brightness'],
            QUANTUM['coupling_efficiency'], length_m,
            QUANTUM['fiber_loss_db_per_km_780nm'],
            QUANTUM['detector_efficiency'],
            colocated=QUANTUM['colocated'],
            local_optics_efficiency=QUANTUM['local_optics_efficiency'],
        )
        share = 1.0 / max(degrees.get(a, 2), 1) + 1.0 / max(degrees.get(b, 2), 1)
        raw = per_source * share
        rates[key] = _rate_waterfall_at(raw, qber)['final_secure_rate']
    return rates


def nodes_for_cell_count(n_cells):
    """Invert the planar-triangulation cell count to a node count.

    Euler gives K = 2n - 2 - h exactly for a triangulation of n points
    with h on the convex hull. For uniform points in a convex k-gon the
    hull size is E[h] = (2k/3) ln n + O(1) (Renyi and Sulanke, Z.
    Wahrscheinlichkeitstheorie 2, 75 (1963)); the campus is a square, so
    k = 4 and E[h] = (8/3) ln n. That law, with no fitted parameter,
    tracks this generator's meshes to RMS 0.25 vertices from n = 6 to
    n = 200 and beats a two-parameter power law - checked in
    tests/test_mesh_edge_load.py.

    Since h grows logarithmically it is o(n), so K -> 2n asymptotically;
    at the sizes reported here the hull term still matters, so solve
    K = 2n - 2 - (8/3) ln n numerically rather than taking K/2.
    """
    if n_cells < 1:
        return 0
    lo, hi = 3, max(8, 2 * n_cells + 16)
    while lo < hi:
        mid = (lo + hi) // 2
        cells = 2 * mid - 2 - (8.0 / 3.0) * math.log(max(mid, 2))
        if cells < n_cells:
            lo = mid + 1
        else:
            hi = mid
    return lo


def compute_max_mesh_size(
    model_params, training_round_s, secure_key_rate_bps, bits_per_param=None,
    fpga_latency_s=0.005
):
    """Largest mesh whose round fits the training period.

    Under masked cell submission the key demand on a physical edge is
    capped at MAX_MASKS_PER_EDGE_PER_ROUND whatever the mesh size, so
    entangled-key supply sets a floor on the round time but does not
    bound the mesh. What bounds it is the aggregator's single classical
    access port, which must carry one masked vector per cell per round.

    bits_per_param defaults to config.BITS_PER_PARAM (32: the int32
    modular masking ring the implementation actually uses; 16-bit
    BF16-style accounting is not implementable against
    QUANT_SCALE = 2^20 - see config.py).

    Returns:
        max_triangles: Largest cell count whose round fits, 0 if the
            key floor alone already exceeds the round period.
        max_nodes: Node count giving that many cells.
    """
    if bits_per_param is None:
        try:
            from config import BITS_PER_PARAM
        except ImportError:
            from ..config import BITS_PER_PARAM
        bits_per_param = BITS_PER_PARAM
    try:
        from config import CLASSICAL_LINK_BPS
    except ImportError:
        from ..config import CLASSICAL_LINK_BPS
    gradient_bits = model_params * bits_per_param
    if gradient_bits <= 0 or training_round_s <= 0:
        return 0, 0

    # Floor: the busiest edge's masks must distil inside the round, and
    # this does not improve with a smaller mesh.
    key_floor = (MAX_MASKS_PER_EDGE_PER_ROUND * gradient_bits
                 / max(secure_key_rate_bps, 1))
    budget = training_round_s - key_floor - fpga_latency_s
    if budget <= 0:
        # Infeasible at any size. Report 0 rather than a mesh that
        # cannot finish a round.
        return 0, 0

    max_triangles = int(budget * CLASSICAL_LINK_BPS / gradient_bits)
    if max_triangles < 1:
        return 0, 0
    return max_triangles, max(3, nodes_for_cell_count(max_triangles))


def analyze_mesh(n_nodes, model_params, secure_key_rate_bps,
                 training_round_s=60, campus_size_m=2000, rng=None):
    """Full mesh analysis: generate topology, build tree, compute latencies."""
    if rng is None:
        rng = np.random.default_rng(42)

    positions, triangles, graph = generate_campus_mesh(n_nodes, campus_size_m, rng)
    tree, depth = build_aggregation_tree(triangles)

    # Per-edge rates: a node splits its source across every incident
    # edge, and each edge runs at its own length, so a flat rate taken
    # from a lone 500 m triangle overstates a mesh's well-connected
    # links several-fold.
    links = sorted({tuple(sorted((int(t[i]), int(t[j]))))
                    for t in triangles for i, j in ((0, 1), (1, 2), (0, 2))})
    degrees = {}
    for a, b in links:
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
    edge_rates = (per_edge_secret_rates(positions, links, degrees)
                  if links else {})

    try:
        from config import BITS_PER_PARAM
    except ImportError:
        from ..config import BITS_PER_PARAM
    gradient_bits = model_params * BITS_PER_PARAM  # int32 modular masking ring
    per_level, total_time = compute_aggregation_latency(
        len(triangles), gradient_bits, secure_key_rate_bps
    )

    utilization = total_time / training_round_s if training_round_s > 0 else float('inf')

    return MeshResult(
        n_nodes=n_nodes,
        n_triangles=len(triangles),
        n_links=graph.number_of_edges(),
        aggregation_depth=depth,
        aggregation_tree=tree,
        node_positions=positions,
        triangles=triangles,
        per_level_latency_s=per_level,
        total_aggregation_time_s=total_time,
        randomness_utilization=min(utilization, 1.0),
        edge_secret_rates=edge_rates,
        node_degrees=degrees,
    )


