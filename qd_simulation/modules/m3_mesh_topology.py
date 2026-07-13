"""
Module 3: Mesh topology and aggregation scheduling.

Generates overlapping 3-clique (triangle) meshes from a Delaunay
triangulation of the campus, constructs an aggregation tree over the
triangle dual graph, and computes per-level latencies under the
shared-bit-rate budget. Triangles are the minimum geometric unit
providing complete pairwise coverage for 3-client Bonawitz masking.
"""
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


def compute_aggregation_latency(
    n_triangles, aggregation_depth, gradient_bits, secure_key_rate_bps,
    fpga_latency_s=0.005
):
    """Compute aggregation latency for the mesh.

    Per-level latency = mask_generation_time + FPGA_processing + communication
    Total = depth × per_level

    Args:
        n_triangles: Number of triangles in the mesh.
        aggregation_depth: Tree depth.
        gradient_bits: Gradient size in bits.
        secure_key_rate_bps: Final secure key rate per link.
        fpga_latency_s: FPGA processing latency (Assumption C).

    Returns:
        per_level_latencies: List of per-level times.
        total_time: Total aggregation time.
    """
    mask_gen_time = gradient_bits / max(secure_key_rate_bps, 1)
    communication_time = gradient_bits / 1e9  # 1 Gbps classical link
    per_level = mask_gen_time + fpga_latency_s + communication_time

    per_level_latencies = [per_level] * max(aggregation_depth, 1)
    total_time = per_level * max(aggregation_depth, 1)

    return per_level_latencies, total_time


def compute_max_mesh_size(
    model_params, training_round_s, secure_key_rate_bps, bits_per_param=None
):
    """Compute maximum mesh size for a given model and training round.

    K_max ≈ (T / t_per_level)² where t_per_level = gradient_bits / key_rate

    bits_per_param defaults to config.BITS_PER_PARAM (32: the int32 modular
    masking ring the implementation actually uses; 16-bit BF16-style
    accounting is not implementable against QUANT_SCALE = 2^20 — see
    config.py).

    Returns:
        max_triangles: Maximum number of triangles.
        max_nodes: Approximate maximum nodes (≈ 2 × max_triangles).
    """
    if bits_per_param is None:
        try:
            from config import BITS_PER_PARAM
        except ImportError:
            from ..config import BITS_PER_PARAM
        bits_per_param = BITS_PER_PARAM
    gradient_bits = model_params * bits_per_param
    t_per_level = gradient_bits / max(secure_key_rate_bps, 1)

    if t_per_level <= 0 or training_round_s <= 0:
        return 0, 0
    # Infeasible: a single level's mask-generation time already exceeds the
    # whole round period. The previous max(1, ...) clamp converted this into
    # a fabricated "2-node feasible mesh" — at honest rates most of exp3b's
    # modeled panel was that artifact (Gate-1 F2). Report 0.
    if t_per_level > training_round_s:
        return 0, 0

    max_depth = max(1, int(training_round_s / t_per_level))
    # NOTE: max_triangles ~ depth^2 is a heuristic motivated by planar Delaunay
    # diameter scaling on uniformly placed points (Euclidean planar mesh
    # diameter grows as sqrt(N)). Not a proved bound — adversarial placements
    # or non-Euclidean topologies will deviate. exp3 measures the empirical
    # exponent over random placements.
    max_triangles = max_depth ** 2
    # A single triangle needs 3 nodes; the 2×triangles heuristic undercounts
    # below that (Gate-1 F2).
    max_nodes = max(3, int(max_triangles * 2))

    return max_triangles, max_nodes


def analyze_mesh(n_nodes, model_params, secure_key_rate_bps,
                 training_round_s=60, campus_size_m=2000, rng=None):
    """Full mesh analysis: generate topology, build tree, compute latencies."""
    if rng is None:
        rng = np.random.default_rng(42)

    positions, triangles, graph = generate_campus_mesh(n_nodes, campus_size_m, rng)
    tree, depth = build_aggregation_tree(triangles)

    try:
        from config import BITS_PER_PARAM
    except ImportError:
        from ..config import BITS_PER_PARAM
    gradient_bits = model_params * BITS_PER_PARAM  # int32 modular masking ring
    per_level, total_time = compute_aggregation_latency(
        len(triangles), depth, gradient_bits, secure_key_rate_bps
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
    )


