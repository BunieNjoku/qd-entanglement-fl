"""
Entanglement-Based Shared Randomness for FL - Configuration.

Parameters anchored to published experimental data where possible.
See system_description.md for source citations.
"""
import os
import numpy as np

# Output directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# MODULE 1: QUANTUM SOURCE PARAMETERS
QUANTUM = {
    "fidelity": 0.987,              # Schimpf 2021
    "fidelity_sweep": np.linspace(0.90, 0.99, 10),
    "source_rep_rate_hz": 80e6,     # 80 MHz pulsed excitation
    "source_brightness": 0.01,      # Pair emission probability per pulse
    "fiber_length_m": 500,          # Campus scale default
    "fiber_length_sweep_m": np.array([100, 250, 500, 1000, 2000, 5000]),
    "fiber_loss_db_per_km_780nm": 3.0,   # Native QD emission wavelength
    "fiber_loss_db_per_km_1550nm": 0.2,  # Telecom C-band after QFC
    "use_qfc": True,                      # Enable quantum frequency conversion to 1550nm
    "qfc_efficiency": 0.35,               # PPLN waveguide conversion efficiency (Weber 2019)
    "qfc_insertion_loss_db": 4.6,         # -10*log10(0.35) ≈ 4.6 dB
    "coupling_efficiency": 0.15,    # Source-to-fiber
    "coupling_sweep": np.linspace(0.05, 0.40, 8),
    "spad_efficiency": 0.65,        # Excelitas SPCM-AQRH
    "local_optics_efficiency": 0.90, # PBS + waveplate loss for local Photon 1 (no fiber)
    "colocated": True,               # Co-located QD architecture (Photon 1 stays local)
    "spad_jitter_ps": 350,          # FWHM
    "spad_dark_count_hz": 50,
    "coincidence_window_ns": 2.0,
    "multi_photon_prob": 8e-5,      # Schimpf 2021
    "pbs_extinction_ratio": 1000,   # 1000:1
    # Fixed conservative allowance for residual birefringence in the
    # ANALYTIC QBER model (m1.compute_qber_analytical). The Monte Carlo
    # derives birefringence from fiber_birefringence_rad_per_km_per_C ×
    # length × ΔT; for campus links (≤3 km) this allowance upper-bounds
    # that contribution. Config-visible so OPERATING_POINT moves with
    # declared physics (Gate-1 F4).
    "analytic_qber_birefringence_allowance": 0.001,
    "fiber_birefringence_rad_per_km_per_C": 0.01,  # Compensated SMF (pol. controller)
    "temperature_variation_C": 2.0,
    # CHSH measurement angles (radians) - optimal for |Φ+⟩
    "chsh_angles_alice": [0, np.pi/4],           # 0°, 45°
    "chsh_angles_bob": [np.pi/8, 3*np.pi/8],     # 22.5°, 67.5°
}

# MODULE 2: POST-PROCESSING PARAMETERS
POST_PROC = {
    "block_size": 50000,            # Bits per EC block
    "block_size_sweep": [10000, 25000, 50000, 100000],
    "ldpc_max_iterations": 50,      # BP decoder iterations
    "ldpc_code_rate": 0.84,         # Optimized for QBER ~0.02
    "security_margin_bits": 128,    # Privacy amplification margin
    "qber_sweep": np.linspace(0.01, 0.06, 11),
    "num_blocks_per_point": 100,    # Statistical sample size
    # FPGA latency estimation (Assumption C)
    "fpga_ldpc_throughput_mbps": 100,     # Conservative estimate
    "fpga_toeplitz_clock_mhz": 300,
    "fpga_latency_safety_factor": 2.0,    # 2× margin
    "bell_test_duty_cycle": 0.02,  # 2% overhead
}

# QKD-FRAMED COMPARISON ARM (exp13)
# Same physics as OPERATING_POINT; only the post-processing framing
# differs. Comparison-only - never feeds headline numbers.
QKD_FRAMING = {
    # Basis keep-fractions p^2 + (1-p)^2 for BBM92-class sifting
    # (Lo, Chau & Ardehali, J. Cryptology 2005).
    "sifting_modes": {
        "symmetric": 0.5,                    # textbook BBM92, p = 0.5
        "efficient-90/10": 0.9**2 + 0.1**2,  # = 0.82, QuNetQFL's split
        "efficient-asymptotic": 1.0,         # asymptotic efficient limit
    },
    # e_ph = e_bit: standard symmetric-channel assumption; exp13 prints
    # the caveat about m1's noise model.
    "phase_error_equals_bit_error": True,
}

# MASKING WIDTH - single source of truth
# The implementation masks int32-quantized gradients in the modular ring
# Z_{2^32} (m4_federated_learning: QUANT_SCALE = 2^20 → 20 fractional bits
# plus sign and dynamic range). Every throughput, cost, and feasibility
# calculation must therefore charge 32 bits of distilled key per parameter
# per edge per round. A 16-bit (BF16-style) payload is NOT implementable
# against this construction: QUANT_SCALE = 2^20 does not fit a 16-bit ring.
# Lowering this constant would require changing the masking ring and
# QUANT_SCALE in modules/m4_federated_learning.py - never change it in
# isolation.
BITS_PER_PARAM = 32

# Classical-network reference bandwidth for communication-latency modeling
# (single source of truth: m3's aggregation-latency model and m4's
# round-latency decomposition both consume this).
CLASSICAL_LINK_BPS = 1e9  # 1 Gbps campus LAN reference

# DERIVED OPERATING POINTS - single source of truth for key rates
# The per-edge secret rate is DERIVED at import time from the m1 device
# model and the m2 post-processing waterfall, evaluated at the
# QUANTUM/POST_PROC parameters above. It moves automatically if the physics
# config changes. Experiments must import OPERATING_POINT (modeled device)
# or PROJECTED_OPERATING_POINT (labeled hypothetical) - never hardcode a
# rate literal.


def _rate_waterfall_at(raw_edge_rate_hz, qber):
    """m2 rate waterfall at this config's post-processing parameters."""
    try:
        from modules.m2_post_processing import compute_rate_waterfall
    except ImportError:
        from .modules.m2_post_processing import compute_rate_waterfall
    return compute_rate_waterfall(
        raw_edge_rate_hz, qber,
        code_rate=POST_PROC['ldpc_code_rate'],
        security_margin=POST_PROC['security_margin_bits'],
        block_size=POST_PROC['block_size'],
        bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
        # GLLP tagged fraction made explicit here so the whole derivation is
        # visible in one place (m2's internal default resolves identically).
        tagged_fraction=QUANTUM['multi_photon_prob'] / QUANTUM['source_brightness'],
    )


def _derive_operating_point():
    """Modeled-device operating point (analytic, deterministic).

    Raw edge rate: the symmetric dual-endpoint edge receives ~R/2 from each
    endpoint's source (m1.edge_key_rate_symmetric), netting one full
    source-link coincidence rate R = m1.compute_coincidence_rate(...) per
    edge; the triangle's three edges distill CONCURRENTLY at this rate, so
    the binding per-round constraint is the per-edge pool-fill time.
    QBER: m1.compute_qber_analytical at the config fidelity/extinction.
    Experiment 1 [1d] cross-validates both with Monte Carlo (15,355 Hz,
    QBER 0.0068 at seed 7 vs analytic ≈15.6 kHz, 0.0085 - the MC includes
    sampled accidentals/dark counts; agreement within sampling noise, and
    the analytic QBER is the conservative one).

    All numbers are simulation-grade (m1 device model), not hardware
    measurements.
    """
    try:
        from modules.m1_quantum_source import (
            compute_coincidence_rate, compute_qber_analytical)
    except ImportError:
        from .modules.m1_quantum_source import (
            compute_coincidence_rate, compute_qber_analytical)

    raw_edge_rate_hz = compute_coincidence_rate(
        QUANTUM['source_rep_rate_hz'], QUANTUM['source_brightness'],
        QUANTUM['coupling_efficiency'], QUANTUM['fiber_length_m'],
        QUANTUM['fiber_loss_db_per_km_1550nm'] if QUANTUM['use_qfc']
        else QUANTUM['fiber_loss_db_per_km_780nm'],
        QUANTUM['spad_efficiency'],
        colocated=QUANTUM['colocated'],
        local_optics_efficiency=QUANTUM['local_optics_efficiency'],
        use_qfc=QUANTUM['use_qfc'],
        qfc_efficiency=QUANTUM['qfc_efficiency'],
    )
    qber = compute_qber_analytical(
        QUANTUM['fidelity'], pbs_extinction=QUANTUM['pbs_extinction_ratio'],
        bire_contribution=QUANTUM['analytic_qber_birefringence_allowance'],
    )
    wf = _rate_waterfall_at(raw_edge_rate_hz, qber)
    return {
        "label": "modeled device (m1+m2 derived, simulation-grade)",
        "raw_edge_rate_hz": raw_edge_rate_hz,
        "qber": qber,
        "secret_rate_bps": wf["final_secure_rate"],
        "tagged_fraction": wf["tagged_fraction"],
        "waterfall": wf,
        # Honesty caveat (Gate-1 F11): the reconciliation stage behind this
        # rate is the statistical placeholder m2.fast_ec_simulate (its FER
        # tiers are asserted, not derived); the repo's real BP decoder
        # currently fails frames at low QBER in exp2d. Rates are
        # simulation-grade pending EC validation.
        "ec_model_caveat": "EC modeled by statistical placeholder "
                           "(m2.fast_ec_simulate); BP-decoder validation "
                           "pending - see exp2d",
    }


def _derive_projected_operating_point():
    """HYPOTHETICAL 1 MHz-class source - NOT produced by the m1 device model.

    Retained as the explicitly-labeled optimistic projection: raw
    coincidence rate 1 MHz at QBER 2% - the assumption formerly smuggled
    into exps 2/3/6/8 as a hardcoded 675000 bps literal. (That legacy
    literal additionally baked in the superseded pre-GLLP waterfall; run
    through the corrected waterfall, the same 1 MHz / 2% assumption yields
    ≈0.73 Mbps - ≈0.81 Mbps before charging the EC tier model's 10% FER,
    which applies because 2% QBER sits in the declared rate-0.84 code's
    degraded band, 0.8×capacity ≤ q < capacity.) Sources of this
    brightness class are plausible for state-of-the-art entangled-pair
    hardware but are hypothetical here: nothing in m1 produces this rate
    at the config above. Every result computed at this point MUST carry
    the "projected" label.
    """
    raw_edge_rate_hz = 1e6   # hypothetical 1 MHz-class raw coincidence rate
    qber = 0.02              # exp2's historical Assumption-D midpoint
    wf = _rate_waterfall_at(raw_edge_rate_hz, qber)
    return {
        "label": "projected 1 MHz-class source (hypothetical, not modeled)",
        "raw_edge_rate_hz": raw_edge_rate_hz,
        "qber": qber,
        "secret_rate_bps": wf["final_secure_rate"],
        "tagged_fraction": wf["tagged_fraction"],
        "waterfall": wf,
        # Same honesty caveat as OPERATING_POINT - and it bites harder
        # here: at 2% QBER the declared rate-0.84 code sits in the tier
        # model's DEGRADED band, so this point is charged FER 0.10
        # (Gate-1 round-2 finding 1).
        "ec_model_caveat": "EC modeled by statistical placeholder "
                           "(m2.fast_ec_simulate tier model; FER 0.10 at "
                           "this QBER - the declared rate-0.84 code is in "
                           "its degraded band at 2%); BP-decoder "
                           "validation pending - see exp2d",
    }


OPERATING_POINT = _derive_operating_point()
PROJECTED_OPERATING_POINT = _derive_projected_operating_point()

# MODULE 3: MESH TOPOLOGY PARAMETERS
MESH = {
    "node_counts_sweep": [3, 6, 9, 12, 18, 24, 30, 40, 50],
    "campus_size_m": 2000,          # 2km × 2km campus
}

# MODULE 4: FEDERATED LEARNING PARAMETERS
FL = {
    "num_clients": 3,               # Matches 1-triangle default
    "num_clients_sweep": [3, 6, 9, 12],
    "num_rounds": 100,              # Training rounds
    "local_epochs": 2,              # Local training epochs per round
    "local_batch_size": 32,
    "learning_rate": 0.01,
    "num_repeats": 10,              # For statistical significance (mean ± std with CI)
    # Model sizes (parameter counts) for throughput sweep
    "model_sizes_sweep": [5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6],
    # Differential privacy baseline
    "dp_epsilon_sweep": [1.0, 5.0, 10.0],
    "dp_delta": 1e-5,
    "dirichlet_alpha": 0.5,         # Non-IID Dirichlet allocation (lower = more heterogeneous)
    # Gradient inversion attack
    "attack_iterations": 3000,
    "attack_lr": 0.1,
    "num_attack_images": 4,
    # CIFAR-10 CNN
    "cifar_rounds": 80,
    "cifar_local_epochs": 1,
}

# MODULE 5: COST MODEL PARAMETERS
COST = {
    # CAPEX (EUR, near-term pricing)
    # Option B co-located: 1 QD per triangle membership per node.
    # Edge nodes (1 triangle): 1 QD, 1 switch, 7 SPADs.
    # Shared nodes (2 triangles): 2 QDs, 2 switches, 12-14 SPADs, 1 dual-QD cryostat.
    # Total QDs = 3 × n_triangles (each triangle claims 1 QD at each vertex).
    # Total cryostats = n_nodes (shared nodes: larger cryostat housing 2 QD chips).
    "qd_source": 55000,             # QD chip + mount (without cryostat)
    "cryostat": 25000,              # 4K cryostat enclosure (single or dual QD)
    "cryostat_dual_premium": 8000,  # Extra cost for dual-QD cryostat vs single
    "optical_switch": 3500,          # EO Pockels cell or MEMS (780nm, ns switching)
    "spad_module": 10000,
    "fpga_board": 12000,
    "dac_optics_per_node": 5000,
    "fiber_per_link": 1500,
    # Volume pricing multiplier
    "volume_discount": 0.6,         # 40% discount at scale
    # OPEX
    "electricity_eur_per_kwh": 0.15,
    "pue": 1.5,
    "cryostat_power_w": 400,        # Per cryostat (dominates; dual-QD adds ~10W laser)
    "spad_power_w": 5,
    "fpga_power_w": 50,
    "annual_maintenance_per_node": 17500,  # Per-node (cryostat service, calibration)
    "equipment_lifetime_years": 5,
    # Comparison baselines
    "pqc_vpn_capex_per_site": 30000,
    "pqc_vpn_annual_license": 5000,
    "hsm_capex_per_site": 100000,
    "he_compute_overhead_factor": 100,
    "dp_cost_per_site": 0,
}

# VISUALIZATION
VIZ = {
    # IEEE single-column = 3.5in, double-column = 7.16in
    # Use double-column width for most plots; single for compact ones
    "figsize": (7.16, 4.5),            # Double-column default
    "figsize_wide": (7.16, 3.5),       # Double-column, shorter (side-by-side subplots)
    "figsize_single": (3.5, 3.0),      # Single-column
    "dpi": 300,                         # Publication quality
    "colors": {
        "quantum": "#1B6B93",
        "prng": "#F4A261",
        "unmasked": "#E76F51",
        "dp": "#8AB17D",
        "pareto": "#2A9D8F",
        "accent": "#E9C46A",
        "dark": "#264653",
    },
    "font_size": 9,                     # Base font for 10pt IEEE body text
}
