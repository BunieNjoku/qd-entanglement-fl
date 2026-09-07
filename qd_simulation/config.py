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
#
# Anchor: the rack-based QD-CamNetz entangled photon pair source
# (Zena et al., arXiv:2604.02024 - "Zena 2026"). Where the rack paper
# does not report a quantity, sister-device measurements from the same
# group and excitation protocol are used and marked as such:
# Hopfmann et al., PRB 103, 075413 (2021) - "Hopfmann 2021" - and
# Langer et al., APL Photonics 10, 066117 (2025) - "Langer 2025".
# Source and detection parameters are measured; the two-node fiber link
# and everything downstream remain modeled.

# Measured entanglement negativity 2n of the rack source. The six-hour
# unattended average (Zena 2026, Fig. 6b) is the operating anchor; the
# 8 ps-window maximum and the T1-average bound the acceptance-window
# trade (Fig. 5).
_NEGATIVITY_2N = 0.960          # six-hour average, 0.960(4)
_NEGATIVITY_2N_WINDOW = 0.98    # maximum within an 8 ps window, 0.98(1)
_NEGATIVITY_2N_T1AVG = 0.94     # averaged over one X lifetime, 0.94(1)

# Werner mapping: for w|phi+><phi+| + (1-w)I/4 the negativity satisfies
# 2N = (3w - 1)/2 (partial-transpose eigenvalue (1-3w)/4; standard
# result, pinned numerically in tests/test_anchored_source.py), so
# w = (2*(2N) + 1)/3. Werner is the conservative noise attribution: the
# measured negativity loss is dominated by FSS precession (coherent,
# basis-dependent, absent from the fixed measurement basis), which the
# window model in m1 refines; charging it as isotropic Werner noise
# overstates the fixed-basis QBER rather than understating it.
_WERNER_W = (2 * _NEGATIVITY_2N + 1) / 3          # 0.9733

# Per-pulse excitation efficiency (probability the XX cascade fires):
# measured 0.076(9) on the sister device (Hopfmann 2021), limited by
# parasitic positive charging of the dot, not by the coherent pi-pulse.
_ETA_EX = 0.076

# Measured combined X+XX singles rate in the V basis (Zena 2026,
# Fig. 6a: six-hour average 697(8) kHz at the 1 GHz clock). Under H/V
# balance this equals the average per-arm total singles rate, which
# calibrates the per-arm transmission below.
_SINGLES_V_BASIS_HZ = 697e3
_DETECTOR_EFF = 0.85            # Single Quantum SSPD, nominal (Zena 2026)
# Per-arm transmission dot -> detector input (collection into the fiber
# objective + module/projection optics), calibrated so the model
# reproduces the measured singles rate. Cross-check: Hopfmann 2021
# measures the same-defined per-arm detection efficiency directly as
# 1.1(1)% including the detector; here 0.0108 * 0.85 = 0.92%.
_ARM_TRANSMISSION = _SINGLES_V_BASIS_HZ / (1e9 * _ETA_EX) / _DETECTOR_EFF

# Multi-pair emission probability per pulse from the measured
# second-order autocorrelation g2_XX(0) = 8.0(6)e-4 (Langer 2025) via
# the sub-Poissonian multiphoton bound p_multi <= g2(0) * <n>^2 / 2
# (Waks, Santori & Yamamoto, PRA 66, 042315 (2002)). The GLLP tagged
# fraction consumed by m2 is multi_photon_prob / source_brightness
# = g2 * eta_ex / 2 ~ 3.0e-5.
_G2_XX_ZERO = 8.0e-4

QUANTUM = {
    # m1's werner_state() is parameterized by the Bell weight w; the key
    # keeps its historical name "fidelity".
    "fidelity": _WERNER_W,          # derived from measured 2n = 0.960(4)
    "negativity_2n": _NEGATIVITY_2N,
    "negativity_2n_window": _NEGATIVITY_2N_WINDOW,
    "negativity_2n_t1avg": _NEGATIVITY_2N_T1AVG,
    "fidelity_sweep": np.linspace(0.90, 0.99, 10),
    "source_rep_rate_hz": 1e9,      # 1 GHz clocked TPE (Zena 2026)
    "source_brightness": _ETA_EX,   # per-pulse excitation efficiency
    "singles_rate_v_basis_hz": _SINGLES_V_BASIS_HZ,
    "fiber_length_m": 500,          # Campus scale default (SCENARIO)
    "fiber_length_sweep_m": np.array([100, 250, 500, 1000, 2000, 5000]),
    # Native 780 nm link fiber (780HP class, Langer 2025); spec-class
    # attenuation. No quantum frequency conversion: at campus distances
    # the PPLN insertion loss (~4.6 dB) exceeds the 780 nm fiber loss
    # it would save.
    "fiber_loss_db_per_km_780nm": 3.5,
    "coupling_efficiency": _ARM_TRANSMISSION,   # calibrated, ~0.0108
    # Sweep spans the calibrated value up to the group's device roadmap:
    # theoretical microlens fiber-coupled efficiency 0.21-0.24 (0.37-0.39
    # with AR coating, Langer 2025); simulated free-standing CBG devices
    # 0.27-0.40 (Dhurjati 2026, arXiv:2605.02360).
    "coupling_sweep": np.array([0.005, _ARM_TRANSMISSION, 0.02, 0.05,
                                0.10, 0.21, 0.30, 0.40]),
    "detector_efficiency": _DETECTOR_EFF,
    "local_optics_efficiency": 1.0,  # absorbed into calibrated coupling
    "colocated": True,               # rack architecture (Photon 1 local)
    "detector_jitter_rms_ps": 20,    # SSPD single-detector RMS (Zena 2026)
    "two_photon_jitter_fwhm_ps": 50, # two-photon timing resolution
    "detector_dark_count_hz": 50,    # not reported; conservative SSPD value
    "coincidence_window_ns": 0.5,    # ~3x T1; swept in exp15
    "multi_photon_prob": 0.5 * _G2_XX_ZERO * _ETA_EX**2,  # Waks 2002 bound
    "g2_xx_zero": _G2_XX_ZERO,
    "pbs_extinction_ratio": 1000,   # 1000:1, not reported; standard spec
    # Exciton fine structure and lifetime (Zena 2026): drive the
    # entanglement-vs-delay model behind the exp15 window optimization.
    "fss_uev": 2.54,                # 2.54(12) ueV
    "t1_x_ps": 162.0,               # 162(4) ps
    # Fixed conservative allowance for residual birefringence in the
    # ANALYTIC QBER model (m1.compute_qber_analytical). The Monte Carlo
    # derives birefringence from fiber_birefringence_rad_per_km_per_C ×
    # length × ΔT; for campus links (≤3 km) this allowance upper-bounds
    # that contribution. Config-visible so OPERATING_POINT moves with
    # declared physics.
    "analytic_qber_birefringence_allowance": 0.001,
    "fiber_birefringence_rad_per_km_per_C": 0.01,  # Compensated SMF (pol. controller)
    "temperature_variation_C": 2.0,  # lab climate 18(2) C (Zena 2026)
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
    # Decoding-success efficiency for the finite-blocklength FER model
    # (m2.tier_frame_error_rate): the code corrects while (1-R) covers
    # ec_decoder_beta x h2(q). Literature midpoint for tuned LDPC
    # reconciliation (Elkouss 2009, Martinez-Mateo 2013: ~1.05-1.12).
    # Distinct from the conservative 1.15 leakage envelope. The
    # projected 2%-QBER point is sensitive to this value (see the
    # m2 docstring); the knob exists so that sensitivity is
    # reproducible.
    "ec_decoder_beta": 1.10,
    "security_margin_bits": 128,    # Privacy amplification margin
    "qber_sweep": np.linspace(0.01, 0.06, 11),
    "num_blocks_per_point": 100,    # Statistical sample size
    # FPGA latency estimation (Assumption C)
    "fpga_ldpc_throughput_mbps": 100,     # Conservative estimate
    "fpga_toeplitz_clock_mhz": 300,
    "fpga_latency_safety_factor": 2.0,    # 2× margin
    # Conduit-mode entanglement-witness duty cycle, derived from measured
    # quantities instead of asserted: one 4-setting CHSH check at the
    # rack system's 60 s/setting tomography integration (Zena 2026,
    # Sec. 4) once per measured six-hour hands-off stability window
    # (2n never below 0.95 without active stabilization, Fig. 6b):
    # 4 x 60 s / 6 h = 0.0111. BBM92-mode parameter estimation is priced
    # inside sifting instead and does not use this knob.
    "bell_test_duty_cycle": (4 * 60) / (6 * 3600),
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

# BBM92 EXECUTING ARM (m1.simulate_bbm92 + m2.bbm92_distill, exp14)
# Same m1 physics as the source-model arm; measurement-side protocol
# upgrade to entanglement-based QKD (Bennett-Brassard-Mermin 1992) that
# tolerates an ACTIVE channel adversary. Executing counterpart of the
# exp13 waterfall comparison.
BBM92 = {
    "p_z": 0.9,           # Z-basis probability, efficient asymmetric split
                          # (Lo-Chau-Ardehali 2005) - matches exp13's
                          # efficient-90/10 mode (keep 0.82)
    # Abort threshold on the observed X-basis error rate. Set at the
    # declared rate-0.84 LDPC code's design QBER: beyond it the fixed
    # code fails frames anyway (m2.tier_frame_error_rate), so the
    # protocol has no key to lose by aborting.
    "q_tol": 0.02,
    # Composable security parameters for the TLGR finite-key bound
    # (Tomamichel-Lim-Gisin-Renner 2012, Eq. 2). Values in the range
    # used by Lim et al., PRA 89, 022307 (2014). The resulting PA floor
    # log2(2/(eps_sec^2 eps_cor)) is ~111 bits at run level (~115/block
    # when the eps are split across 3 blocks), comparable to the
    # source-model arm's fixed 128-bit margin.
    "eps_sec": 1e-9,
    "eps_cor": 1e-15,
    # Intercept-resend tap-fraction sweep for exp14 (fine steps below
    # the detection threshold ~2*q_tol, coarse above).
    "eve_fraction_sweep": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12,
                           0.20, 0.35, 0.50, 0.75, 1.0],
}

# CLASSICAL IT-DEALER BASELINE (m7.run_it_dealer, exp9/exp12)
# The strongest classical objection made executable: a trusted dealer
# provisions pairwise one-time pads over classical channels. Comparison
# baseline only.
IT_DEALER = {
    # Illustrative courier cadence for the amortized-batch report in
    # exp12 (one physical provisioning visit per this many rounds; the
    # pads for all interim rounds sit in storage between visits).
    "courier_rounds_per_visit": 30,
}

# MASKING WIDTH - single source of truth
# The implementation masks int32-quantized gradients in the modular ring
# Z_{2^32} (m4_federated_learning: QUANT_SCALE = 2^20 -> 20 fractional bits
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

# DERIVED OPERATING POINT - single source of truth for key rates
# The per-edge secret rate is DERIVED at import time from the m1 device
# model and the m2 post-processing waterfall, evaluated at the
# QUANTUM/POST_PROC parameters above. It moves automatically if the physics
# config changes. Experiments must import OPERATING_POINT - never hardcode
# a rate literal.


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
    """Anchored operating point (analytic, deterministic).

    Raw edge rate: the symmetric dual-endpoint edge receives ~R/2 from each
    endpoint's source (m1.edge_key_rate_symmetric), netting one full
    source-link coincidence rate R = m1.compute_coincidence_rate(...) per
    edge; the triangle's three edges distill CONCURRENTLY at this rate, so
    the binding per-round constraint is the per-edge pool-fill time.
    QBER: m1.compute_qber_analytical at the config Bell weight/extinction.

    Source and detection inputs are measured (Zena 2026; sister-device
    values Hopfmann 2021 / Langer 2025 - see the QUANTUM block); the link
    and distillation are modeled, so the derived rate is still
    simulation-grade, not a hardware measurement. The coincidence rate
    inherits the eta_ex sensitivity noted at source_brightness: at fixed
    measured singles it scales as 1/eta_ex, pending the rack device's own
    coincidence data.
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
        # Honesty caveat: the reconciliation FER behind this rate is
        # the finite-blocklength normal approximation at efficiency
        # ec_decoder_beta (m2.tier_frame_error_rate) - an idealized
        # benchmark of a competently implemented code, not a measured
        # decoder curve; the repo's toy BP decoder underperforms it
        # (exp2d). Rates are simulation-grade.
        "ec_model_caveat": "EC FER from the finite-blocklength normal "
                           "approximation at efficiency ec_decoder_beta "
                           "(m2.tier_frame_error_rate), not a measured "
                           "decoder curve - see exp2d",
    }


OPERATING_POINT = _derive_operating_point()

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
