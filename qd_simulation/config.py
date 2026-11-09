"""
Entanglement-Based Shared Randomness for FL - Configuration.

Parameters anchored to published experimental data where possible.
Source citations are given inline at each block; the hardware anchors
are listed in the Module 1 header below.
"""
import os
import numpy as np

# Output directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Module 1: quantum source parameters.
#
# Anchor: the rack-based QD-CamNetz entangled photon pair source, Zena
# et al., Adv. Quantum Technol. 9(10), 2026, doi 10.1002/qute.70510
# (arXiv:2604.02024), "Zena 2026". Quantities it does not report come
# from sister devices of the same group: Hopfmann et al., PRB 103,
# 075413 (2021) and Langer et al., APL Photonics 10, 066117 (2025).
# Source and detection parameters are measured; the fiber link and
# everything downstream are modeled.

# Entanglement negativity 2n of the rack source (Zena 2026, Fig. 6b and
# Fig. 5).
_NEGATIVITY_2N = 0.960          # six-hour average, 0.960(4)
_NEGATIVITY_2N_WINDOW = 0.98    # maximum within an 8 ps window, 0.98(1)
_NEGATIVITY_2N_T1AVG = 0.94     # averaged over one X lifetime, 0.94(1)

# Werner mapping: for w|phi+><phi+| + (1-w)I/4, 2N = (3w - 1)/2, so
# w = (2*(2N) + 1)/3 (checked in tests/test_anchored_source.py).
# Charging the measured negativity loss as isotropic Werner noise is
# conservative for the fixed-basis QBER.
_WERNER_W = (2 * _NEGATIVITY_2N + 1) / 3          # 0.9733

# Per-pulse excitation efficiency, 0.076(9) on the sister device
# (Hopfmann 2021), limited by parasitic charging of the dot.
_ETA_EX = 0.076

# Combined X+XX singles rate in the V basis, six-hour average (Zena
# 2026, Fig. 6a); calibrates the per-arm transmission below.
_SINGLES_V_BASIS_HZ = 697e3
_DETECTOR_EFF = 0.85            # Single Quantum SSPD, nominal (Zena 2026)
# Per-arm transmission from dot to detector input, calibrated to the
# measured singles rate. Cross-check: Hopfmann 2021 measures 1.1(1)%
# including the detector; here 0.0108 * 0.85 = 0.92%.
_ARM_TRANSMISSION = _SINGLES_V_BASIS_HZ / (1e9 * _ETA_EX) / _DETECTOR_EFF

# Multi-pair emission probability per pulse from g2_XX(0) = 8.0(6)e-4
# (Langer 2025) via p_multi <= g2(0) * <n>^2 / 2 (Waks, Santori &
# Yamamoto, PRA 66, 042315).
_G2_XX_ZERO = 8.0e-4

QUANTUM = {
    # Bell weight w of m1.werner_state(); the key name is historical.
    "fidelity": _WERNER_W,
    "negativity_2n": _NEGATIVITY_2N,
    "negativity_2n_window": _NEGATIVITY_2N_WINDOW,
    "negativity_2n_t1avg": _NEGATIVITY_2N_T1AVG,
    "fidelity_sweep": np.linspace(0.90, 0.99, 10),
    "source_rep_rate_hz": 1e9,      # 1 GHz clocked TPE (Zena 2026)
    "source_brightness": _ETA_EX,   # per-pulse excitation efficiency
    "singles_rate_v_basis_hz": _SINGLES_V_BASIS_HZ,
    "fiber_length_m": 500,          # campus default
    "fiber_length_sweep_m": np.array([100, 250, 500, 1000, 2000, 5000]),
    # Native 780 nm link fiber (780HP class), no frequency conversion.
    "fiber_loss_db_per_km_780nm": 3.5,
    "coupling_efficiency": _ARM_TRANSMISSION,   # calibrated, ~0.0108
    "detector_efficiency": _DETECTOR_EFF,
    "local_optics_efficiency": 1.0,  # absorbed into calibrated coupling
    "colocated": True,               # rack architecture (Photon 1 local)
    "detector_jitter_rms_ps": 20,    # SSPD single-detector RMS (Zena 2026)
    # Two-photon timing resolution as quoted by Zena 2026 (supplement);
    # used directly by the window model.
    "two_photon_jitter_fwhm_ps": 50,
    "detector_dark_count_hz": 50,    # not reported; conservative SSPD value
    "coincidence_window_ns": 0.5,    # ~3x T1; swept in exp15
    "multi_photon_prob": 0.5 * _G2_XX_ZERO * _ETA_EX**2,  # Waks 2002 bound
    "g2_xx_zero": _G2_XX_ZERO,
    "pbs_extinction_ratio": 1000,   # 1000:1, not reported; standard spec
    # Exciton fine structure and lifetime (Zena 2026): drive the
    # entanglement-vs-delay model behind the exp15 window optimization.
    "fss_uev": 2.54,                # 2.54(12) ueV
    "t1_x_ps": 162.0,               # 162(4) ps
    # Residual-birefringence allowance in m1.compute_qber_analytical; at
    # campus lengths the Monte Carlo contribution is below 2e-4.
    "analytic_qber_birefringence_allowance": 0.0,
    "fiber_birefringence_rad_per_km_per_C": 0.01,  # Compensated SMF (pol. controller)
    "temperature_variation_C": 2.0,  # lab climate 18(2) C (Zena 2026)
    # CHSH measurement angles (radians) - optimal for |Phi+>
    "chsh_angles_alice": [0, np.pi/4],           # 0 deg, 45 deg
    "chsh_angles_bob": [np.pi/8, 3*np.pi/8],     # 22.5 deg, 67.5 deg
}

# Module 2: post-processing parameters.
POST_PROC = {
    "block_size": 50000,            # Bits per EC block
    "block_size_sweep": [10000, 25000, 50000, 100000],
    "ldpc_max_iterations": 50,      # BP decoder iterations
    "ldpc_code_rate": 0.84,         # Optimized for QBER ~0.02
    # Decoding-success efficiency for m2.tier_frame_error_rate; the
    # literature midpoint (Elkouss 2009, Martinez-Mateo 2013). Distinct
    # from the 1.15 leakage envelope.
    "ec_decoder_beta": 1.10,
    "security_margin_bits": 128,    # Privacy amplification margin
    "qber_sweep": np.linspace(0.01, 0.06, 11),
    "num_blocks_per_point": 100,    # Statistical sample size
    # FPGA latency estimation (Assumption C)
    "fpga_ldpc_throughput_mbps": 100,     # Conservative estimate
    "fpga_toeplitz_clock_mhz": 300,
    "fpga_latency_safety_factor": 2.0,    # 2x margin
    # Conduit-mode entanglement-witness duty cycle: one 4-setting CHSH
    # check at 60 s/setting (Zena 2026, Sec. 4) per six-hour stability
    # window (Fig. 6b).
    "bell_test_duty_cycle": (4 * 60) / (6 * 3600),
}

# QKD-framed comparison arm (exp13); same physics as OPERATING_POINT.
QKD_FRAMING = {
    # Basis keep-fractions p^2 + (1-p)^2 (Lo, Chau & Ardehali 2005).
    "sifting_modes": {
        "symmetric": 0.5,
        "efficient-90/10": 0.9**2 + 0.1**2,
        "efficient-asymptotic": 1.0,
    },
    "phase_error_equals_bit_error": True,
}

# BBM92 executing arm (m1.simulate_bbm92 + m2.bbm92_distill, exp14).
BBM92 = {
    # Z-basis probability (Lo-Chau-Ardehali 2005), keep 0.82.
    "p_z": 0.9,
    # Abort threshold on the observed X-basis error rate: the rate-0.84
    # code's design QBER.
    "q_tol": 0.02,
    # TLGR finite-key security parameters (Tomamichel-Lim-Gisin-Renner
    # 2012; values as in Lim et al., PRA 89, 022307).
    "eps_sec": 1e-9,
    "eps_cor": 1e-15,
    # Intercept-resend tap-fraction sweep for exp14.
    "eve_fraction_sweep": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12,
                           0.20, 0.35, 0.50, 0.75, 1.0],
}

# Classical IT-dealer baseline (m7.run_it_dealer, exp9/exp12).
IT_DEALER = {
    # Rounds per courier provisioning visit, for exp12's batch report.
    "courier_rounds_per_visit": 30,
}

# Masking width: int32 modular ring in m4 (QUANT_SCALE = 2^20). Change
# together with the masking ring and QUANT_SCALE.
BITS_PER_PARAM = 32

# Classical-network reference bandwidth for latency modeling (m3, m4).
CLASSICAL_LINK_BPS = 1e9

# Derived operating point: the per-edge secret rate from the m1 device
# model and the m2 waterfall at the parameters above.


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
        tagged_fraction=QUANTUM['multi_photon_prob'] / QUANTUM['source_brightness'],
    )


def _derive_operating_point():
    """Anchored operating point (analytic, deterministic).

    Raw edge rate: one source-link coincidence rate R per edge (each
    endpoint's source contributes ~R/2, m1.edge_key_rate_symmetric).
    QBER: m1.compute_qber_analytical at the config Bell weight and
    extinction.
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
        QUANTUM['fiber_loss_db_per_km_780nm'],
        QUANTUM['detector_efficiency'],
        colocated=QUANTUM['colocated'],
        local_optics_efficiency=QUANTUM['local_optics_efficiency'],
    )
    qber = compute_qber_analytical(
        QUANTUM['fidelity'], pbs_extinction=QUANTUM['pbs_extinction_ratio'],
        bire_contribution=QUANTUM['analytic_qber_birefringence_allowance'],
    )
    wf = _rate_waterfall_at(raw_edge_rate_hz, qber)
    return {
        "label": "QD-CamNetz-anchored (source/detection measured, "
                 "link+distillation modeled)",
        "raw_edge_rate_hz": raw_edge_rate_hz,
        "qber": qber,
        "secret_rate_bps": wf["final_secure_rate"],
        "tagged_fraction": wf["tagged_fraction"],
        "waterfall": wf,
        "ec_model_caveat": "EC FER from the finite-blocklength normal "
                           "approximation at efficiency ec_decoder_beta "
                           "(m2.tier_frame_error_rate), not a measured "
                           "decoder curve - see exp2d",
    }


OPERATING_POINT = _derive_operating_point()

# Module 3: mesh topology parameters.
MESH = {
    "node_counts_sweep": [3, 6, 9, 12, 18, 24, 30, 40, 50],
    "campus_size_m": 2000,          # 2km x 2km campus
}

# Module 4: federated learning parameters.
FL = {
    "num_clients": 3,               # Matches 1-triangle default
    "num_clients_sweep": [3, 6, 9, 12],
    "num_rounds": 100,              # Training rounds
    "local_epochs": 2,              # Local training epochs per round
    "local_batch_size": 32,
    "learning_rate": 0.01,
    "num_repeats": 10,              # repeats for mean +- std and CI
    # Model sizes (parameter counts) for throughput sweep
    "model_sizes_sweep": [5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6],
    "dirichlet_alpha": 0.5,         # Non-IID Dirichlet allocation (lower = more heterogeneous)
    # Gradient inversion attack
    "attack_iterations": 3000,
    "attack_lr": 0.1,
    "num_attack_images": 4,
    # CIFAR-10 CNN
    "cifar_rounds": 80,
    "cifar_local_epochs": 1,
}

# Visualization.
VIZ = {
    # IEEE single-column = 3.5in, double-column = 7.16in
    # Use double-column width for most plots; single for compact ones
    "figsize": (7.16, 4.5),            # Double-column default
    "figsize_wide": (7.16, 3.5),       # Double-column, shorter (side-by-side subplots)
    "figsize_grid": (7.16, 5.2),        # Double-column, one row of three subplots
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
