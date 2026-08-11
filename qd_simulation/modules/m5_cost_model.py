"""
Module 5: Cost and Performance Model
Parametric CAPEX/OPEX, energy analysis, timing precision, Pareto frontier.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CostResult:
    """Cost analysis results for a given configuration."""
    n_nodes: int
    n_triangles: int
    capex_total: float
    capex_per_node: float
    opex_annual: float
    power_total_w: float
    cost_per_training_round: float
    cost_per_secure_mbit: float
    energy_per_round_kwh: float


def compute_capex(n_triangles, n_nodes, volume=False, params=None):
    """Compute total CAPEX for the quantum overlay.

    Option B co-located architecture:
      - Each node hosts 1 QD per triangle it belongs to.
      - Edge nodes (1 triangle): 1 QD, 1 switch, 1 cryostat, 7 SPADs.
      - Shared nodes (2 triangles): 2 QDs, 2 switches, 1 dual-QD cryostat, 12-14 SPADs.
      - Total QDs = 3 × n_triangles (each triangle claims 1 QD at each of its 3 vertices).
      - Total cryostats = n_nodes (shared nodes use larger dual-QD cryostats).
      - Uniform 80 MHz per link. No bottlenecks.

    Args:
        n_triangles: Number of entangled triangles.
        n_nodes: Number of distinct nodes.
        volume: Whether to apply volume discount.
        params: Cost parameter dict (from config.COST).
    """
    if params is None:
        from config import COST as params

    discount = params.get("volume_discount", 0.6) if volume else 1.0

    # QD sources: 3 per triangle (one per vertex, shared nodes get multiple)
    n_qds = n_triangles * 3
    # Support both old single-price and new split-price config
    qd_unit = params.get("qd_source", params.get("qd_source_cryostat", 75000) - 25000)
    capex_qds = n_qds * qd_unit * discount

    # Cryostats: 1 per node (shared nodes get larger dual-QD cryostat)
    n_shared_nodes = n_qds - n_nodes  # nodes with 2+ QDs
    n_edge_nodes = n_nodes - n_shared_nodes
    cryo_base = params.get("cryostat", 25000)
    cryo_dual_premium = params.get("cryostat_dual_premium", 8000)
    capex_cryostats = (n_edge_nodes * cryo_base +
                       n_shared_nodes * (cryo_base + cryo_dual_premium)) * discount

    # Optical switches: 1 per QD (each QD needs its own 2-way switch)
    capex_switches = n_qds * params.get("optical_switch", 3500) * discount

    # SPADs: edge nodes 7, shared nodes ~13 average
    # Formula: 7 per node base + 6 extra per shared node
    n_spads = 7 * n_nodes + 6 * n_shared_nodes
    capex_spads = n_spads * params["spad_module"] * discount

    # FPGAs: 1 per node (multi-channel for shared nodes)
    capex_fpga = n_nodes * params["fpga_board"] * discount

    # Optics (waveplates, PBS): scales with QD count (1 set per QD)
    capex_optics = n_nodes * params["dac_optics_per_node"] * discount

    # Fiber: approximate unique links
    # In a triangle mesh: n_links ≈ 1.5 × n_edges_per_triangle × n_triangles / sharing
    # Conservative: use n_triangles × 3 (overcounts shared edges slightly)
    n_links = n_triangles * 3
    capex_fiber = n_links * params["fiber_per_link"] * discount

    total = (capex_qds + capex_cryostats + capex_switches + capex_spads +
             capex_fpga + capex_optics + capex_fiber)

    return {
        "total": total,
        "qds": capex_qds,
        "cryostats": capex_cryostats,
        "switches": capex_switches,
        "spads": capex_spads,
        "fpga": capex_fpga,
        "optics": capex_optics,
        "fiber": capex_fiber,
        "per_node": total / max(n_nodes, 1),
        "n_qds": n_qds,
        "n_spads": n_spads,
        "n_shared_nodes": n_shared_nodes,
    }


def compute_opex_annual(n_triangles, n_nodes, params=None):
    """Compute annual OPEX for the quantum overlay.
    Option B co-located: 1 cryostat per node (some dual-QD). Power per cryostat."""
    if params is None:
        from config import COST as params

    # Power - 1 cryostat per node (dual-QD cryostats use ~same power)
    power_cryostats = n_nodes * params["cryostat_power_w"]
    # SPADs: approximate total from Option B formula
    n_qds = n_triangles * 3
    n_shared = n_qds - n_nodes
    n_spads = 7 * n_nodes + 6 * n_shared
    power_spads = n_spads * params["spad_power_w"]
    power_fpga = n_nodes * params["fpga_power_w"]
    total_power_w = power_cryostats + power_spads + power_fpga

    electricity_annual = (
        total_power_w / 1000  # kW
        * params["pue"]
        * 8760  # hours/year
        * params["electricity_eur_per_kwh"]
    )

    maintenance = n_nodes * params.get("annual_maintenance_per_node",
                                        params.get("annual_maintenance_per_triangle", 35000))
    total_opex = electricity_annual + maintenance

    return {
        "total": total_opex,
        "electricity": electricity_annual,
        "maintenance": maintenance,
        "power_w": total_power_w,
    }


def compute_cost_per_round(n_triangles, n_nodes, training_rounds_per_year,
                            params=None):
    """Compute cost per training round (amortized CAPEX + OPEX)."""
    if params is None:
        from config import COST as params

    capex = compute_capex(n_triangles, n_nodes, params=params)
    opex = compute_opex_annual(n_triangles, n_nodes, params=params)

    annual_cost = (capex["total"] / params["equipment_lifetime_years"]) + opex["total"]
    cost_per_round = annual_cost / max(training_rounds_per_year, 1)

    return cost_per_round, annual_cost


def compute_energy_per_round(n_triangles, n_nodes, round_duration_s, params=None):
    """Energy consumed by quantum infrastructure per training round."""
    if params is None:
        from config import COST as params

    opex = compute_opex_annual(n_triangles, n_nodes, params=params)
    power_kw = opex["power_w"] / 1000
    energy_kwh = power_kw * (round_duration_s / 3600)
    return energy_kwh


def compute_classical_baseline_costs(n_nodes, training_rounds_per_year, params=None):
    """Compute costs for classical security alternatives."""
    if params is None:
        from config import COST as params

    lifetime = params["equipment_lifetime_years"]

    # PQC VPN (computational security)
    pqc_capex = n_nodes * params["pqc_vpn_capex_per_site"]
    pqc_opex = n_nodes * params["pqc_vpn_annual_license"]
    pqc_annual = pqc_capex / lifetime + pqc_opex
    pqc_per_round = pqc_annual / max(training_rounds_per_year, 1)

    # HSM cluster
    hsm_capex = n_nodes * params["hsm_capex_per_site"]
    hsm_annual = hsm_capex / lifetime + pqc_opex  # Similar maintenance
    hsm_per_round = hsm_annual / max(training_rounds_per_year, 1)

    # DP (free infrastructure, but accuracy cost)
    dp_annual = 0
    dp_per_round = 0

    # HE (high compute cost)
    # Assume HE requires 100× more compute per round
    base_gpu_cost_per_round = 0.10  # EUR per round on cloud GPU
    he_per_round = base_gpu_cost_per_round * params["he_compute_overhead_factor"]
    he_annual = he_per_round * training_rounds_per_year

    return {
        "pqc": {"annual": pqc_annual, "per_round": pqc_per_round,
                "security": "computational", "accuracy_impact": 0},
        "hsm": {"annual": hsm_annual, "per_round": hsm_per_round,
                "security": "computational", "accuracy_impact": 0},
        "dp_eps1": {"annual": dp_annual, "per_round": dp_per_round,
                    "security": "epsilon-1 DP", "accuracy_impact": 0.15},
        "dp_eps5": {"annual": dp_annual, "per_round": dp_per_round,
                    "security": "epsilon-5 DP", "accuracy_impact": 0.05},
        "dp_eps10": {"annual": dp_annual, "per_round": dp_per_round,
                     "security": "epsilon-10 DP", "accuracy_impact": 0.02},
        "he": {"annual": he_annual, "per_round": he_per_round,
               "security": "computational (HE)", "accuracy_impact": 0},
    }


def build_pareto_frontier(quantum_cost, quantum_accuracy, baselines, dp_accuracies=None):
    """Build the cost-privacy-accuracy Pareto frontier.

    Returns list of (cost, accuracy, security_level, method_name) tuples.
    """
    points = []

    # Quantum
    points.append({
        "method": "Quantum (this system)",
        "annual_cost": quantum_cost,
        "accuracy": quantum_accuracy,
        "security": "information-theoretic",
        "security_rank": 4,  # Highest
    })

    for name, baseline in baselines.items():
        acc = quantum_accuracy - baseline["accuracy_impact"]
        points.append({
            "method": name,
            "annual_cost": baseline["annual"],
            "accuracy": max(acc, 0.5),
            "security": baseline["security"],
            "security_rank": 1 if "DP" in baseline["security"] else 2
                             if "HE" in baseline["security"] else 3,
        })

    return points


def full_cost_analysis(n_triangles, n_nodes, model_params,
                        training_round_s, secure_key_rate_bps,
                        training_rounds_per_year=None, params=None):
    """Complete cost analysis combining all components."""
    if params is None:
        from config import COST as params

    if training_rounds_per_year is None:
        rounds_per_day = 86400 / training_round_s
        training_rounds_per_year = int(rounds_per_day * 365 * 0.5)  # 50% utilization

    capex = compute_capex(n_triangles, n_nodes, params=params)
    opex = compute_opex_annual(n_triangles, n_nodes, params=params)
    cost_per_round, annual = compute_cost_per_round(
        n_triangles, n_nodes, training_rounds_per_year, params
    )
    energy = compute_energy_per_round(n_triangles, n_nodes, training_round_s, params)

    try:
        from config import BITS_PER_PARAM
    except ImportError:
        from ..config import BITS_PER_PARAM
    gradient_bits = model_params * BITS_PER_PARAM  # int32 modular masking ring
    secure_mbits_per_year = secure_key_rate_bps * 8760 * 3600 / 1e6 * 0.5
    cost_per_secure_mbit = annual / max(secure_mbits_per_year, 1)

    return CostResult(
        n_nodes=n_nodes,
        n_triangles=n_triangles,
        capex_total=capex["total"],
        capex_per_node=capex["per_node"],
        opex_annual=opex["total"],
        power_total_w=opex["power_w"],
        cost_per_training_round=cost_per_round,
        cost_per_secure_mbit=cost_per_secure_mbit,
        energy_per_round_kwh=energy,
    )
