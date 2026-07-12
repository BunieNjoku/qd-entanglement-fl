"""
Entanglement-Based Shared Randomness for FL — Configuration.

Parameters anchored to published experimental data where possible.
See system_description.md for source citations.
"""
import os
import numpy as np

# ── Output directories ──
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════
# MODULE 1: QUANTUM SOURCE PARAMETERS
# ══════════════════════════════════════════════
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
    "fiber_birefringence_rad_per_km_per_C": 0.01,  # Compensated SMF (pol. controller)
    "temperature_variation_C": 2.0,
    # CHSH measurement angles (radians) — optimal for |Φ+⟩
    "chsh_angles_alice": [0, np.pi/4],           # 0°, 45°
    "chsh_angles_bob": [np.pi/8, 3*np.pi/8],     # 22.5°, 67.5°
}

# ══════════════════════════════════════════════
# MODULE 2: POST-PROCESSING PARAMETERS
# ══════════════════════════════════════════════
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

# ══════════════════════════════════════════════
# MODULE 3: MESH TOPOLOGY PARAMETERS
# ══════════════════════════════════════════════
MESH = {
    "node_counts_sweep": [3, 6, 9, 12, 18, 24, 30, 40, 50],
    "campus_size_m": 2000,          # 2km × 2km campus
}

# ══════════════════════════════════════════════
# MODULE 4: FEDERATED LEARNING PARAMETERS
# ══════════════════════════════════════════════
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

# ══════════════════════════════════════════════
# MODULE 5: COST MODEL PARAMETERS
# ══════════════════════════════════════════════
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

# ══════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════
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
