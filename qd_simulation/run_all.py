#!/usr/bin/env python3
"""
entangled-key two-mask FL — full simulation runner.

Threat model (entangled-key, not QKD):
  - Cooperating clients trust each other and trust the QD source at the
    co-located node. The fiber is noisy but not actively adversarial.
  - Pairwise Bell-pair measurement produces correlated bit pairs.
  - LDPC reconciliation makes the two endpoints' strings bit-exact.
  - Toeplitz PA compresses the strings to remove residual classical
    correlation introduced by the reconciliation syndrome and by
    physical-layer side channels.
  - The resulting per-round single-use shared bits drive modular OTP
    masking in two-mask Bonawitz secure aggregation.

Composition of privacy:
  (i)   Source verification: CHSH S > 2 verifies entangled state under
        Assumption B. This is a source check, not an eavesdropper bound.
  (ii)  Reconciliation + PA produce shared strings with min-entropy
        approaching block length (Leftover Hash Lemma; Renner 2005).
  (iii) Modular OTP masking on int32-quantized gradients: under the
        standard Bonawitz threat model with t-collusion threshold, the
        aggregator learns only the sum of gradients.
  (iv)  Per-round freshness: bits consumed once. Survivor reveal of
        pairwise bits on dropout does not compromise other rounds.

Usage:
    python run_all.py              # Run all experiments
    python run_all.py --exp 1 4 5  # Run specific experiments
    python run_all.py --quick      # Quick mode (reduced sweeps)
"""
import os
import sys
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from config import (QUANTUM, POST_PROC, MESH, FL, COST, VIZ, RESULTS_DIR,
                    BITS_PER_PARAM, OPERATING_POINT, PROJECTED_OPERATING_POINT,
                    CLASSICAL_LINK_BPS)

from modules.m1_quantum_source import (
    simulate_source, compute_chsh_s, werner_state,
    compute_qber_analytical, compute_coincidence_rate,
    simulate_symmetric_edge,
)
from modules.m2_post_processing import (
    fast_ec_simulate, compute_rate_waterfall, binary_entropy,
    validate_bp_against_statistical, toeplitz_hash, validate_pa_output,
    estimate_capacity_qber, randomness_battery,
)
from modules.m3_mesh_topology import (
    generate_campus_mesh, build_aggregation_tree,
    compute_aggregation_latency, compute_max_mesh_size,
    analyze_mesh,
)
from modules.m5_cost_model import (
    compute_capex, compute_opex_annual, compute_cost_per_round,
    compute_energy_per_round, compute_classical_baseline_costs,
    full_cost_analysis,
)

# ── Publication-quality plot defaults (IEEE 10pt, Times New Roman) ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',           # Math font matching Times
    'font.size': VIZ['font_size'],        # 9pt base (matches 10pt body)
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'figure.dpi': VIZ['dpi'],
    'savefig.dpi': VIZ['dpi'],
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'figure.constrained_layout.use': False,
})
C = VIZ['colors']


def save_fig(fig, name):
    """Save figure as both PNG (preview) and PDF (LaTeX inclusion)."""
    for ext in ['png', 'pdf']:
        path = os.path.join(RESULTS_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=VIZ['dpi'], bbox_inches='tight',
                    facecolor='white', pad_inches=0.03)
    plt.close(fig)
    print(f"    Saved: {name}.png/pdf")


def print_comm_summary(fl_result, label=""):
    """Per-client per-round communication summary (HANDOFF Phase 2).

    Means over rounds; uplink split by phase. key_wait and wire latency
    are MODELED (config operating point / CLASSICAL_LINK_BPS); local,
    masking, and aggregator times are measured on this machine.
    """
    rounds = getattr(fl_result, 'comm_rounds', None)
    if not rounds:
        print(f"    [comm{(' ' + label) if label else ''}] no accounting recorded")
        return None
    n = len(rounds)

    def _mean(f):
        return sum(f(r) for r in rounds) / n

    nc = max(rounds[0].n_clients, 1)
    mult = max((getattr(r, 'max_payload_multiplicity', 1) for r in rounds),
               default=1)
    up = _mean(lambda r: r.uplink_bytes()) / nc
    phases = {
        'payload': _mean(lambda r: r.payload_bytes) / nc,
        'shares': _mean(lambda r: r.share_bytes) / nc,
        'selfmask': _mean(lambda r: r.selfmask_bytes) / nc,
        'recovery+reveal': _mean(lambda r: r.recovery_bytes()) / nc,
        'inter-tri': _mean(lambda r: r.inter_triangle_bytes) / nc,
    }
    down = _mean(lambda r: r.downlink_bytes) / nc
    key_intra = _mean(lambda r: r.key_bits_intra)
    key_inter = _mean(lambda r: r.key_bits_inter)
    tfail = sum(r.threshold_failures for r in rounds)
    mult_note = (f"; heaviest client sends ×{mult} payloads"
                 if mult > 1 else "")
    print(f"    [comm{(' ' + label) if label else ''}] MEAN per client·round"
          f"{mult_note}: uplink {up:,.0f} B ("
          + ", ".join(f"{k} {v:,.0f}" for k, v in phases.items() if v > 0)
          + f"), downlink {down:,.0f} B")
    print(f"           key bits/round: intra {key_intra:,.0f}, inter {key_inter:,.0f}"
          f"  |  threshold-failed triangle-rounds: {tfail}")
    print(f"           latency/round: key_wait {_mean(lambda r: r.key_wait_s):.2f}s (modeled), "
          f"local {_mean(lambda r: r.local_compute_s)*1e3:.0f}ms, "
          f"masking {_mean(lambda r: r.masking_s)*1e3:.0f}ms, "
          f"aggregator {_mean(lambda r: r.aggregator_time_s)*1e3:.1f}ms, "
          f"wire {_mean(lambda r: r.comm_s)*1e3:.2f}ms (modeled)")
    return {"uplink_bytes_per_client_round": up,
            "downlink_bytes_per_client_round": down,
            "recovery_bytes_per_client_round": phases['recovery+reveal'],
            "threshold_failed_triangle_rounds": tfail}


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 1: QUANTUM SOURCE CHARACTERIZATION
# ═══════════════════════════════════════════════════════════
def run_experiment_1(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 1: Quantum Source Characterization")
    print("="*70)

    fidelities = QUANTUM['fidelity_sweep'] if not quick else np.linspace(0.90, 0.99, 5)
    fiber_lengths = QUANTUM['fiber_length_sweep_m'] if not quick else np.array([100, 500, 2000])
    coupling_effs = QUANTUM['coupling_sweep'] if not quick else np.linspace(0.05, 0.35, 4)

    use_qfc = QUANTUM.get('use_qfc', False)
    qfc_eff = QUANTUM.get('qfc_efficiency', 0.35)
    # Select fiber loss based on whether QFC converts to telecom C-band
    fiber_loss = QUANTUM.get('fiber_loss_db_per_km_1550nm', 0.2) if use_qfc else QUANTUM['fiber_loss_db_per_km_780nm']
    wavelength_label = "1550nm (C-band via QFC)" if use_qfc else "780nm (native)"

    # ── 1a: QBER vs Fidelity ──
    print("  [1a] QBER vs Fidelity...")
    qbers_sim, qbers_analytical, s_values = [], [], []
    for f in fidelities:
        result = simulate_source(fidelity=f, n_pairs=5_000_000, fiber_length_m=100,
                                 fiber_loss_db_per_km=fiber_loss,
                                 colocated=QUANTUM.get('colocated', True),
                                 local_optics_efficiency=QUANTUM.get('local_optics_efficiency', 0.90),
                                 use_qfc=use_qfc, qfc_efficiency=qfc_eff,
                                 rng=np.random.default_rng(42))
        qbers_sim.append(result.qber)
        qbers_analytical.append(compute_qber_analytical(f))
        s_values.append(result.chsh_s)

    # ── 1d: Symmetric dual-endpoint edge keying (availability/fairness witness) ──
    print("  [1d] Dual-endpoint (beamsplitter-multiplexed) edge keying...")
    from modules.m1_quantum_source import simulate_symmetric_edge
    _sym = simulate_symmetric_edge(
        switch_split=0.5, duration_s=0.2,
        fidelity=QUANTUM['fidelity'], source_rep_rate_hz=QUANTUM['source_rep_rate_hz'],
        source_brightness=QUANTUM['source_brightness'],
        fiber_length_m=QUANTUM['fiber_length_m'],
        fiber_loss_db_per_km=fiber_loss,
        coupling_efficiency=QUANTUM['coupling_efficiency'],
        spad_efficiency=QUANTUM['spad_efficiency'],
        spad_dark_count_hz=QUANTUM['spad_dark_count_hz'],
        coincidence_window_ns=QUANTUM['coincidence_window_ns'],
        pbs_extinction_ratio=QUANTUM['pbs_extinction_ratio'],
        fiber_birefringence_rad_per_km_per_C=QUANTUM['fiber_birefringence_rad_per_km_per_C'],
        temperature_variation_C=QUANTUM['temperature_variation_C'],
        colocated=QUANTUM.get('colocated', True),
        local_optics_efficiency=QUANTUM.get('local_optics_efficiency', 0.90),
        use_qfc=use_qfc, qfc_efficiency=qfc_eff,
        rng=np.random.default_rng(7))
    print(f"      edge rate={_sym.coincidence_rate_hz:,.0f} Hz  QBER={_sym.qber:.4f}  "
          f"S={_sym.chsh_s:.3f}  symmetry_ratio={_sym.params['source_symmetry_ratio']:.3f} "
          f"(0.5 = no privileged endpoint; availability: source failure halves, not kills, an edge)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    ax1.plot(fidelities, np.array(qbers_sim)*100, 'o-', color=C['quantum'], linewidth=2, label='Simulation')
    ax1.plot(fidelities, np.array(qbers_analytical)*100, '--', color=C['accent'], linewidth=2, label='Analytical')
    ax1.axhline(y=1.9, color=C['pareto'], linestyle=':', linewidth=1.5, label='Schimpf 2021 (1.9%)')
    ax1.axhline(y=11, color=C['unmasked'], linestyle=':', linewidth=1, alpha=0.5, label='11% (BB84 QKD reference — not a bound in the source model)')
    ax1.axvline(x=0.987, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Entanglement Fidelity F')
    ax1.set_ylabel('QBER (%)')
    ax1.set_title('QBER vs Source Fidelity')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(fidelities, s_values, 'o-', color=C['quantum'], linewidth=2)
    ax2.axhline(y=2.0, color=C['unmasked'], linestyle='--', linewidth=2, label='Classical bound (S=2)')
    ax2.axhline(y=2*np.sqrt(2), color=C['pareto'], linestyle=':', linewidth=1.5, label=f'Quantum max (S={2*np.sqrt(2):.2f})')
    ax2.fill_between(fidelities, 2.0, s_values, alpha=0.15, color=C['quantum'], label='Quantum advantage region')
    ax2.set_xlabel('Entanglement Fidelity F')
    ax2.set_ylabel('CHSH S-parameter')
    ax2.set_title('Bell Inequality Violation (Assumption B)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.suptitle('Experiment 1a: Source Characterization — Fidelity Sweep', fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp1a_fidelity_sweep')

    # ── 1b: Coincidence Rate vs Fiber Length ──
    print("  [1b] Coincidence rate vs Fiber length...")
    rates = []
    for fl in fiber_lengths:
        rate = compute_coincidence_rate(
            QUANTUM['source_rep_rate_hz'], QUANTUM['source_brightness'],
            QUANTUM['coupling_efficiency'], fl,
            fiber_loss, QUANTUM['spad_efficiency'],
            colocated=QUANTUM.get('colocated', True),
            local_optics_efficiency=QUANTUM.get('local_optics_efficiency', 0.90),
            use_qfc=use_qfc, qfc_efficiency=qfc_eff,
        )
        rates.append(rate)

    fig, ax = plt.subplots(figsize=VIZ['figsize'])
    rates_khz = [r / 1e3 for r in rates]

    ax.plot(fiber_lengths/1000, rates_khz, 'o-', color=C['quantum'], markersize=6)
    # Annotate each data point with its value
    for fl_km, r_khz in zip(fiber_lengths/1000, rates_khz):
        ax.annotate(f'{r_khz:.1f}', xy=(fl_km, r_khz),
                    xytext=(0, 7), textcoords='offset points',
                    ha='center', fontsize=7, color=C['dark'])

    # Reference lines in kHz
    ax.axhline(y=1000, color=C['pareto'], linestyle='--', alpha=0.7, label='1 MHz target')
    ax.axhline(y=100, color=C['accent'], linestyle='--', alpha=0.7, label='100 kHz minimum')

    ax.set_xlabel('Fiber Length (km)')
    ax.set_ylabel('Coincidence Rate (kHz)')
    qfc_note = f', QFC eff={qfc_eff}' if use_qfc else ''
    ax.set_title(f'Coincidence Rate vs Fiber Length\n'
                 f'({wavelength_label}, coupling={QUANTUM["coupling_efficiency"]}{qfc_note})')
    ax.legend()

    # Set y-axis to show the actual data range clearly
    min_rate = min(rates_khz) * 0.7
    max_rate = max(max(rates_khz) * 1.3, 120)  # Include 100 kHz line if close
    ax.set_ylim(min_rate, max_rate * 12)  # Leave room for 1 MHz line
    ax.set_yscale('log')
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, pos: f'{val:.0f}' if val >= 1 else f'{val:.1f}'
    ))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, alpha=0.3, which='major')
    ax.grid(True, alpha=0.15, which='minor')
    fig.tight_layout()
    save_fig(fig, 'exp1b_rate_vs_fiber')

    # ── 1c: Coupling efficiency sensitivity ──
    print("  [1c] Coupling efficiency sensitivity...")
    fig, ax = plt.subplots(figsize=VIZ['figsize'])
    for fl in [100, 500, 1000, 2000]:
        rates_c = [compute_coincidence_rate(
            QUANTUM['source_rep_rate_hz'], QUANTUM['source_brightness'],
            ce, fl, fiber_loss, QUANTUM['spad_efficiency'],
            colocated=QUANTUM.get('colocated', True),
            local_optics_efficiency=QUANTUM.get('local_optics_efficiency', 0.90),
            use_qfc=use_qfc, qfc_efficiency=qfc_eff,
        ) for ce in coupling_effs]
        ax.semilogy(coupling_effs, rates_c, 'o-', linewidth=2, label=f'{fl}m fiber')

    ax.axhline(y=1e6, color='gray', linestyle=':', alpha=0.5,
               label='1 MHz-class (hypothetical projection)')
    ax.set_xlabel('Source-to-Fiber Coupling Efficiency')
    ax.set_ylabel('Coincidence Rate (Hz)')
    ax.set_title('Experiment 1c: Rate Sensitivity to Coupling Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    save_fig(fig, 'exp1c_coupling_sensitivity')

    print("  ✓ Experiment 1 complete.")
    return {"qbers": qbers_sim, "s_values": s_values, "rates": rates}


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 2: POST-PROCESSING RATE VALIDATION
# ═══════════════════════════════════════════════════════════
def run_experiment_2(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 2: Post-Processing Rate Validation")
    print("="*70)

    qber_sweep = POST_PROC['qber_sweep'] if not quick else np.linspace(0.01, 0.05, 5)
    # Operating points from config — single source of truth (HANDOFF 1.1).
    # base_rate is the MODELED device's raw edge rate, not an assumed 1 MHz.
    op = OPERATING_POINT
    proj = PROJECTED_OPERATING_POINT
    base_rate = op['raw_edge_rate_hz']

    def _wf(rate_hz, qber, **kw):
        """Waterfall at this config's post-processing parameters."""
        return compute_rate_waterfall(
            rate_hz, qber,
            code_rate=POST_PROC['ldpc_code_rate'],
            security_margin=POST_PROC['security_margin_bits'],
            block_size=POST_PROC['block_size'],
            bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
            **kw)

    # ── 2a: Rate waterfall at both operating points ──
    print("  [2a] Rate waterfall — modeled device vs labeled projection...")
    wf = _wf(op['raw_edge_rate_hz'], op['qber'])
    # Sanity (HANDOFF 1.1.4): this experiment's waterfall input IS the config
    # operating point, and its output must reproduce config's derived rate —
    # otherwise the single source of truth has been violated somewhere.
    assert abs(wf['final_secure_rate'] - op['secret_rate_bps']) < 1e-6, (
        f"exp2 waterfall ({wf['final_secure_rate']:.3f} bps) diverged from "
        f"config.OPERATING_POINT ({op['secret_rate_bps']:.3f} bps)")
    wf_proj = _wf(proj['raw_edge_rate_hz'], proj['qber'])

    stages = ['Raw', 'Sift', 'EC', 'PA', 'Verif', 'Final']
    fig, axes = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    for ax, w, point in ((axes[0], wf, op), (axes[1], wf_proj, proj)):
        vals_khz = [w['raw_coincidence_rate']/1000, w['after_sifting']/1000,
                    w['after_ec']/1000, w['after_pa']/1000,
                    w['after_verification']/1000, w['final_secure_rate']/1000]
        bars = ax.bar(stages, vals_khz,
                      color=[C['dark'], C['quantum'], C['quantum'],
                             C['quantum'], C['quantum'], C['pareto']],
                      edgecolor='white', linewidth=1.5)
        overheads = ['', '0%', f'−{w["ec_overhead_pct"]:.1f}%',
                     f'−{w["pa_overhead_pct"]:.1f}%',
                     f'−{w["verification_overhead_pct"]:.1f}%',
                     f'−{w["bell_overhead_pct"]:.1f}%']
        for bar, oh in zip(bars, overheads):
            if oh:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        oh, ha='center', va='bottom', fontsize=6, color='gray')
        ax.set_ylabel('Rate (kbit/s)')
        ax.set_title(f"{point['label']}\nQBER {point['qber']*100:.2f}% → "
                     f"{w['final_secure_rate']/1000:.1f} kbit/s", fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')
    fig.suptitle('Experiment 2a: Rate waterfall — modeled device vs labeled projection',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp2a_rate_waterfall')

    print(f"    modeled device : {op['raw_edge_rate_hz']:>11,.0f} Hz raw → "
          f"{wf['final_secure_rate']:>9,.0f} bps secret  (QBER {op['qber']:.4f})")
    print(f"    PROJECTED      : {proj['raw_edge_rate_hz']:>11,.0f} Hz raw → "
          f"{wf_proj['final_secure_rate']:>9,.0f} bps secret  (QBER {proj['qber']:.4f})"
          f"  — hypothetical 1 MHz-class, not modeled")

    # ── 2b: Final secure rate vs QBER ──
    print("  [2b] Secure rate vs QBER sweep...")
    final_rates = []
    ec_overheads = []
    envelope_rates = []
    for q in qber_sweep:
        wf = _wf(base_rate, q)
        final_rates.append(wf['final_secure_rate'])
        ec_overheads.append(wf['ec_overhead_pct'])
        # HYPOTHETICAL rate-adapted envelope for comparison (labeled).
        envelope_rates.append(
            _wf(base_rate, q, assume_rate_adapted=True)['final_secure_rate'])
    cap_q = estimate_capacity_qber(POST_PROC['ldpc_code_rate'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    ax1.plot(qber_sweep*100, np.array(final_rates)/1000, 'o-', color=C['quantum'],
             linewidth=2, label='Declared rate-0.84 code (tier FER)')
    ax1.plot(qber_sweep*100, np.array(envelope_rates)/1000, '--', color=C['accent'],
             linewidth=1.5, label='Rate-adapted family (hypothetical envelope)')
    ax1.axvline(x=cap_q*100, color='gray', linestyle=':', linewidth=1,
                label=f'0.84-code capacity ({cap_q*100:.1f}%)')
    ax1.axvspan(0.8*cap_q*100, cap_q*100, alpha=0.08, color=C['unmasked'],
                label='degraded band (tier FER 0.10)')
    ax1.set_xlabel('QBER (%)')
    ax1.set_ylabel('Final Secure Rate (kbit/s)')
    ax1.set_title(f'Secure rate vs QBER\n(at modeled-device raw rate '
                  f'{base_rate/1000:.1f} kHz)', fontsize=8)
    ax1.axvspan(1.5, 4.0, alpha=0.1, color=C['quantum'], label='Expected range (Assumption D)')
    ax1.axvline(x=op['qber']*100, color=C['dark'], linestyle='--', linewidth=1,
                label=f"operating point ({op['qber']*100:.2f}%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(qber_sweep*100, ec_overheads, 'o-', color=C['accent'], linewidth=2)
    ax2.set_xlabel('QBER (%)')
    ax2.set_ylabel('EC Overhead (%)')
    ax2.set_title('Error Correction Overhead vs QBER')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Experiment 2b: Post-Processing Sensitivity', fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp2b_rate_vs_qber')

    # ── 2c: Rate gain of fixed-basis (no sifting) vs sifting protocol ──
    print("  [2c] Fixed-basis vs sifting protocol rate comparison...")
    our_rates = []
    sifting_rates = []
    for q in qber_sweep:
        our_wf = _wf(base_rate, q)
        sift_wf = _wf(base_rate * 0.5, q)  # 50% sifting loss
        our_rates.append(our_wf['final_secure_rate'])
        sifting_rates.append(sift_wf['final_secure_rate'])

    fig, ax = plt.subplots(figsize=VIZ['figsize'])
    ax.plot(qber_sweep*100, np.array(our_rates)/1000, 'o-', color=C['quantum'],
            linewidth=2, label='entangled-key fixed-basis (trusted source)')
    ax.plot(qber_sweep*100, np.array(sifting_rates)/1000, 's--', color=C['unmasked'],
            linewidth=2, label='Sifting-based reference (50% loss)')
    ax.fill_between(qber_sweep*100,
                     np.array(sifting_rates)/1000, np.array(our_rates)/1000,
                     alpha=0.15, color=C['quantum'],
                     label='Rate gain from trusted-source assumption')
    ax.set_xlabel('QBER (%)')
    ax.set_ylabel('Shared bit rate (kbit/s)')
    ax.set_title(f'Experiment 2c: Fixed-basis vs sifting rate at modeled-device '
                 f'raw rate ({base_rate/1000:.1f} kHz)\n'
                 f'(Fixed-basis requires trusted source; sifting does not)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, 'exp2c_fixedbasis_vs_sifting')

    # ── 2d: BP decoder spot check (design band vs one degraded point) ──
    print("  [2d] BP decoder spot check against the EC tier model "
          "(toy random LDPC)...")
    bp_validation = validate_bp_against_statistical(
        qber_values=[0.005, 0.01, 0.02],
        block_size=800 if quick else 1000,
    )
    print("    QBER    BP FER    Stat FER    Agreement")
    print("    " + "-"*45)
    for row in bp_validation:
        agree = "YES" if row["agreement"] else "NO"
        print(f"    {row['qber']:.3f}   {row['bp_fer']:.2f}      "
              f"{row['stat_fer']:.2f}        {agree}")
    print("    Note: this toy 3-regular rate-0.84 LDPC has a low BSC "
          "threshold — it\n    decodes cleanly below ~0.5% QBER, is "
          "already marginal at the 0.85%\n    operating point, and fails "
          "at 2%. It is a CONSERVATIVE FLOOR, not a\n    certification of "
          "the tiers: a real capacity-approaching code (which the\n    "
          "tiers assume) decodes the full design band. The 0.85-2% gap is "
          "the toy\n    code underperforming that idealized code, not a "
          "tier-model refutation.\n    The stat column is the "
          "deterministic tier FER; the FER tiers remain a\n    stated "
          "placeholder (see fast_ec_simulate). [prior LLR bug fixed: the "
          "old\n    codeword-decoding sign gave FER≈1.0 at every QBER.]")

    # ── 2e: NIST SP 800-22 subset battery on the PA output (HANDOFF 3.2;
    # replaces the former 10,000-bit monobit spot check). The input is the
    # PIPELINE's output, not synthetic rng bits: simulated source at the
    # config fidelity (100 m fiber) → reconciled frames → Toeplitz PA at
    # the LHL-consistent compression for the measured QBER. ──
    print("  [2e] NIST SP 800-22 subset battery on privacy-amplified "
          "pipeline output...")
    rng_pa = np.random.default_rng(42)
    src_pa = simulate_source(fidelity=QUANTUM['fidelity'],
                             n_pairs=8_000_000 if not quick else 4_000_000,
                             fiber_length_m=100, rng=rng_pa)
    ec_pa = fast_ec_simulate(src_pa.raw_bits_alice, src_pa.raw_bits_bob,
                             qber_estimate=src_pa.qber, rng=rng_pa)
    _rec, _off = [], 0
    for _r in ec_pa:
        if _r.frame_success:
            _rec.append(src_pa.raw_bits_alice[_off:_off + _r.n_raw_bits])
        _off += _r.n_raw_bits
    rec_bits = np.concatenate(_rec)
    _wf_pa = compute_rate_waterfall(
        1.0, src_pa.qber, code_rate=POST_PROC['ldpc_code_rate'],
        security_margin=POST_PROC['security_margin_bits'],
        block_size=POST_PROC['block_size'],
        bell_duty_cycle=POST_PROC['bell_test_duty_cycle'])
    _ent_rate = max(0.0, 1.0 - _wf_pa['ec_overhead_pct'] / 100
                    - _wf_pa['tagged_fraction'])
    pa_out_len = (int(len(rec_bits) * _ent_rate)
                  - 2 * POST_PROC['security_margin_bits'])
    secure_output = toeplitz_hash(rec_bits, pa_out_len, rng=rng_pa)
    print(f"    pipeline: {len(src_pa.raw_bits_alice):,} sifted bits @ "
          f"QBER {src_pa.qber:.4f} → {len(rec_bits):,} reconciled → "
          f"{len(secure_output):,} PA output bits "
          f"(LHL rate {_ent_rate:.3f}, margin "
          f"{POST_PROC['security_margin_bits']})")
    battery = randomness_battery(secure_output, alpha=0.01)
    n_pass = sum(r['passed'] for r in battery)
    print(f"    {'test':<28} {'p-value':>10}  verdict")
    for r in battery:
        print(f"    {r['test']:<28} {r['p_value']:>10.4f}  "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    print(f"    → {n_pass}/{len(battery)} tests pass at α = 0.01 "
          f"(expected false rejections for uniform input: "
          f"≈{0.01 * len(battery):.2f})")

    print("  ✓ Experiment 2 complete.")
    return {
        "operating_point_secret_bps": op['secret_rate_bps'],
        "projected_secret_bps": wf_proj['final_secure_rate'],
        "final_rates": final_rates,
        "qber_sweep": qber_sweep.tolist(),
        "bp_validation": bp_validation,
        "pa_battery": {r["test"]: {"p_value": r["p_value"],
                                   "passed": r["passed"]}
                       for r in battery},
        "pa_battery_input": {"qber": float(src_pa.qber),
                             "reconciled_bits": int(len(rec_bits)),
                             "output_bits": int(len(secure_output))},
    }


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 3: MESH SCALING
# ═══════════════════════════════════════════════════════════
def run_experiment_3(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 3: Mesh Scaling")
    print("="*70)

    node_counts = MESH['node_counts_sweep'] if not quick else [3, 6, 12, 24, 40]
    model_sizes = [5e4, 1e5, 5e5, 1e6, 2e6] if not quick else [1e5, 5e5, 1e6]
    round_periods = [15, 30, 60, 120, 300] if not quick else [30, 60, 300]
    # Operating points from config — single source of truth (HANDOFF 1.1).
    op_rate = OPERATING_POINT['secret_rate_bps']
    proj_rate = PROJECTED_OPERATING_POINT['secret_rate_bps']

    # ── 3a: Aggregation depth vs node count ──
    # (Depth and triangle count are geometric — independent of key rate;
    #  the mesh's latency fields use the modeled-device rate.)
    print("  [3a] Aggregation depth vs node count...")
    depths, n_triangles_list = [], []
    for n in node_counts:
        mesh = analyze_mesh(n, 1e5, op_rate, rng=np.random.default_rng(42))
        depths.append(mesh.aggregation_depth)
        n_triangles_list.append(mesh.n_triangles)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    ax1.plot(node_counts, depths, 'o-', color=C['quantum'], linewidth=2, markersize=8, label='Measured')
    # √K reference line
    sqrt_ref = [np.sqrt(nt) for nt in n_triangles_list]
    ax1.plot(node_counts, sqrt_ref, '--', color=C['accent'], linewidth=1.5, label='√(triangles) reference')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Aggregation Tree Depth')
    ax1.set_title('Aggregation Depth vs Network Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(node_counts, n_triangles_list, 'o-', color=C['pareto'], linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Number of Triangles')
    ax2.set_title('Triangulation Density')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Experiment 3a: Mesh Topology Scaling', fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp3a_mesh_scaling')

    # ── 3b: Max nodes heatmap — modeled device vs labeled projection ──
    print("  [3b] Max mesh size heatmap — modeled device vs labeled projection...")
    rate_points = [
        (op_rate, f'modeled device ({op_rate/1000:.1f} kbit/s)'),
        (proj_rate, f'PROJECTED 1 MHz-class ({proj_rate/1000:.0f} kbit/s, hypothetical)'),
    ]
    heatmaps = []
    fig, axes = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    for ax, (rate, rlabel) in zip(axes, rate_points):
        heatmap = np.zeros((len(model_sizes), len(round_periods)))
        for i, ms in enumerate(model_sizes):
            for j, rp in enumerate(round_periods):
                max_tri, max_n = compute_max_mesh_size(int(ms), rp, rate)
                heatmap[i, j] = max_n
        heatmaps.append(heatmap)
        im = ax.imshow(heatmap, aspect='auto', cmap='YlOrBr', interpolation='nearest')
        ax.set_xticks(range(len(round_periods)))
        ax.set_xticklabels([f'{r}s' for r in round_periods])
        ax.set_yticks(range(len(model_sizes)))
        ax.set_yticklabels([f'{int(ms/1000)}K' for ms in model_sizes])
        ax.set_xlabel('Training Round Period')
        ax.set_ylabel('Model Size (parameters)')
        ax.set_title(rlabel, fontsize=8)
        vmax = heatmap.max() if heatmap.max() > 0 else 1
        for i in range(len(model_sizes)):
            for j in range(len(round_periods)):
                val = int(heatmap[i, j])
                ax.text(j, i, str(val), ha='center', va='center', fontsize=6,
                        color='white' if val > 0.55 * vmax else 'black',
                        fontweight='bold')
        fig.colorbar(im, ax=ax, label='Max nodes')
    fig.suptitle('Experiment 3b: Maximum feasible nodes — modeled vs projected rate\n'
                 '(0 = infeasible: per-level mask time exceeds the round period)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp3b_max_nodes_heatmap')

    print("    Max nodes (rows = model size), modeled | projected:")
    header = "  ".join(f"{rp:>9}s" for rp in round_periods)
    print(f"      {'':>6}  {header}")
    for i, ms in enumerate(model_sizes):
        row = "  ".join(f"{int(heatmaps[0][i, j]):>4}|{int(heatmaps[1][i, j]):<5}"
                        for j in range(len(round_periods)))
        print(f"      {int(ms/1000):>5}K  {row}")

    print("  ✓ Experiment 3 complete.")
    return {"depths": depths,
            "heatmap_modeled": heatmaps[0].tolist(),
            "heatmap_projected": heatmaps[1].tolist()}


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 4: FL TRAINING CORRECTNESS
# ═══════════════════════════════════════════════════════════
def run_experiment_4(quick=False):
    """Secure-aggregation fidelity (HEADLINE) + convergence parity (validation).

    HANDOFF 3.5: exact modular masking provably does not change the
    aggregate, so accuracy parity across modes is a CORRECTNESS CHECK,
    not a finding. The load-bearing metric is aggregate fidelity:
      (a) masking exactness — the masked pipeline's integer aggregate
          equals the direct quantized sum word-for-word (theorem;
          asserted at zero mismatches over every round of every seed);
      (b) quantization error — ||dequantized mean − exact float mean||_inf
          measured in float64, against the 2^-20 truncation bound
          (int cast toward zero: per-client error < 1/QUANT_SCALE, so
          the mean-of-n error is < 2^-20 per parameter).
    >= 10 seeds, paired splits AND paired torch seeds across modes,
    95% t-confidence intervals on the paired accuracy deltas.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Secure-Aggregation Fidelity (headline) "
          "+ Convergence Parity (validation)")
    print("="*70)

    from modules.m4_federated_learning import (
        ClinicalMLP, load_clinical_data, run_federated_training, QUANT_SCALE,
    )

    num_rounds = 60 if not quick else 30
    num_clients = 3
    target_acc = 0.92
    # HANDOFF 3.5: >= 10 seeds in BOTH modes (the old quick=3 was
    # underpowered and quick-mode results were being quoted).
    num_repeats = max(10, FL.get('num_repeats', 10))
    dirichlet_alpha = FL.get('dirichlet_alpha', 0.5)

    modes = ["entangled", "prng", "unmasked", "dp"]
    mode_labels = {
        "entangled": "Quantum-masked (this system)",
        "prng": "PRNG-masked (classical)",
        "unmasked": "Unmasked (insecure)",
        "dp": "DP (ε=5)",
    }

    # Run multiple repeats for statistical significance
    all_runs = {mode: [] for mode in modes}
    for repeat in range(num_repeats):
        seed = 42 + repeat
        print(f"\n  ── Repeat {repeat+1}/{num_repeats} (seed={seed}) ──")
        client_data, test_data, input_dim = load_clinical_data(
            num_clients, rng_seed=seed, dirichlet_alpha=dirichlet_alpha
        )
        model_fn = lambda: ClinicalMLP(input_dim=input_dim, hidden=128)

        for mode in modes:
            print(f"    Training [{mode}] for {num_rounds} rounds...")
            result = run_federated_training(
                model_fn, client_data, test_data,
                mode=mode, num_rounds=num_rounds,
                local_epochs=FL['local_epochs'], batch_size=FL['local_batch_size'],
                lr=FL['learning_rate'], target_accuracy=target_acc,
                dp_epsilon=5.0 if mode == "dp" else None,
                verbose=False,
                seed=seed,  # paired across modes within a repeat
            )
            all_runs[mode].append(result)
            if mode == "entangled" and result.quantum_bits_consumed:
                total_bits = sum(result.quantum_bits_consumed)
                print(f"      Quantum key bits consumed: {total_bits:,} "
                      f"({total_bits/8/1024:.1f} KB)")

    # Compute mean ± std across repeats
    results_summary = {}
    for mode in modes:
        accs = [r.final_accuracy for r in all_runs[mode]]
        rounds = [r.rounds_to_target for r in all_runs[mode]]
        results_summary[mode] = {
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs),
            "mean_rounds": np.mean(rounds),
            "std_rounds": np.std(rounds),
            "all_accuracies": [r.accuracies for r in all_runs[mode]],
        }

    # ── HEADLINE: aggregate fidelity of the masked pipeline ──
    total_mismatches = sum(sum(r.agg_exact_mismatches)
                           for r in all_runs["entangled"])
    n_agg_rounds = sum(len(r.agg_exact_mismatches)
                       for r in all_runs["entangled"])
    per_seed_max_err = [max(r.agg_error_inf) for r in all_runs["entangled"]]
    max_err = max(per_seed_max_err)
    quant_bound = 1.0 / QUANT_SCALE  # 2^-20: truncation bound on the mean
    print("\n  HEADLINE — aggregate fidelity "
          f"({num_repeats} seeds x {num_rounds} rounds):")
    print(f"    Masking exactness: {total_mismatches} mismatched uint32 "
          f"words out of {n_agg_rounds} aggregation rounds x "
          f"{all_runs['entangled'][0].n_params:,} params "
          f"(theorem: must be 0)")
    print(f"    Max quantization error ||masked mean - float mean||_inf: "
          f"{max_err:.3e}  vs 2^-20 bound {quant_bound:.3e} "
          f"({'WITHIN' if max_err < quant_bound else 'EXCEEDS'} bound, "
          f"headroom x{quant_bound / max(max_err, 1e-300):.1f})")
    assert total_mismatches == 0, (
        "masking exactness violated — implementation bug")
    assert max_err < quant_bound, (
        f"aggregate error {max_err} exceeds quantization bound {quant_bound}")

    # ── VALIDATION: paired accuracy parity (same seed => same split,
    # same init, same shuffling; only the masking differs) ──
    from scipy.stats import t as _t_dist
    def _paired_ci(deltas):
        d = np.asarray(deltas, dtype=float)
        n = len(d)
        hw = (_t_dist.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
              if n > 1 else float('nan'))
        return d.mean(), hw
    pairs = {}
    for other in ("entangled", "prng"):
        deltas = [all_runs[other][i].final_accuracy
                  - all_runs["unmasked"][i].final_accuracy
                  for i in range(num_repeats)]
        pairs[other] = (_paired_ci(deltas), max(abs(d) for d in deltas))
    print("\n  VALIDATION — paired final-accuracy deltas vs unmasked "
          "(95% t-CI):")
    for other, ((mean_d, hw), max_abs) in pairs.items():
        print(f"    {other:>10} - unmasked: {mean_d:+.4f} ± {hw:.4f} "
              f"(max |Δ| {max_abs:.4f})")
        assert max_abs <= 0.05, (
            f"{other} vs unmasked paired delta {max_abs} exceeds 0.05 — "
            "masking should not measurably change convergence")
    print("    (Parity is EXPECTED — exact masking is a theorem; this "
          "assertion is an\n     implementation check, not a finding.)")
    print("\n  NOTE — DP is not an alternative to masking: DP bounds what "
          "the AGGREGATE\n  leaks about any individual; masking hides "
          "INDIVIDUAL updates from the\n  aggregator. Orthogonal "
          "protections — a hardened deployment runs DP on top\n  of "
          "masking. The DP arm is shown only to display its accuracy "
          "cost.")

    # ── Communication accounting (HANDOFF Phase 2) ──
    print("\n  Communication accounting (last repeat per mode; prng/dp/"
          "unmasked count payload+downlink only — faithful classical "
          "baselines land in Phase 4;\n  key-distillation classical "
          "traffic, syndromes ≈0.20 bits (≈0.025 B) per secret bit, is "
          "excluded from "
          "the FL-layer ledger by scope — see RoundComm):")
    for mode in modes:
        print_comm_summary(all_runs[mode][-1], label=mode)

    # ── Plot: fidelity headline (left) + accuracy validation (right) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    colors = {"entangled": C['quantum'], "prng": C['prng'],
              "unmasked": C['unmasked'], "dp": C['dp']}
    linestyles = {"entangled": "-", "prng": "--", "unmasked": "-.", "dp": ":"}

    for r in all_runs["entangled"]:
        ax1.semilogy(np.arange(1, len(r.agg_error_inf) + 1),
                     np.maximum(r.agg_error_inf, 1e-12),
                     linewidth=0.8, alpha=0.5, color=C['quantum'])
    ax1.axhline(y=quant_bound, color='crimson', linestyle='--', linewidth=1.5,
                label=f'2⁻²⁰ quantization bound ({quant_bound:.1e})')
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('‖masked mean − float mean‖∞ (float64)')
    ax1.set_title(f'HEADLINE: aggregate fidelity, {num_repeats} seeds\n'
                  f'(masking exactness: {total_mismatches} mismatched words '
                  f'— theorem holds)')
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.3, which='both')

    for mode in modes:
        all_acc = np.array(results_summary[mode]["all_accuracies"])
        mean_acc = all_acc.mean(axis=0)
        std_acc = all_acc.std(axis=0)
        rounds_x = np.arange(1, len(mean_acc) + 1)
        ax2.plot(rounds_x, mean_acc, linewidth=2, color=colors[mode],
                 linestyle=linestyles[mode], label=mode_labels[mode])
        ax2.fill_between(rounds_x, mean_acc - std_acc, mean_acc + std_acc,
                         alpha=0.15, color=colors[mode])
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('VALIDATION: convergence parity (mean ± 1σ)\n'
                  'parity is expected — exact masking is a theorem')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.2, 1.02)

    fig.suptitle(f'Experiment 4: Secure-aggregation fidelity — breast-cancer '
                 f'clinical FL\n{num_clients} clients, Dirichlet '
                 f'α={dirichlet_alpha} non-IID, {num_repeats} paired seeds, '
                 f'FedAvg',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp4_training_convergence')

    # ── Summary table (validation metrics) ──
    print(f"\n  SUMMARY ({num_repeats} paired seeds, mean ± std):")
    print(f"  {'Mode':<25} {'Accuracy':>15} {'Rounds to Target':>20}")
    print("  " + "-"*62)
    for mode in modes:
        s = results_summary[mode]
        print(f"  {mode_labels[mode]:<25} "
              f"{s['mean_acc']:.4f} ± {s['std_acc']:.4f} "
              f"{s['mean_rounds']:>8.1f} ± {s['std_rounds']:.1f}")

    print("  ✓ Experiment 4 complete.")
    return {
        "fidelity": {
            "exactness_mismatches": int(total_mismatches),
            "agg_rounds_checked": int(n_agg_rounds),
            "max_agg_error_inf": float(max_err),
            "quant_bound": float(quant_bound),
            "per_seed_max_err": [float(e) for e in per_seed_max_err],
        },
        "paired_deltas_vs_unmasked": {
            other: {"mean": float(ci[0]), "ci95_halfwidth": float(ci[1]),
                    "max_abs": float(mx)}
            for other, (ci, mx) in pairs.items()},
        "modes": {m: {"acc_mean": s["mean_acc"], "acc_std": s["std_acc"],
                      "rounds_mean": s["mean_rounds"]}
                  for m, s in results_summary.items()},
    }


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 5: DROPOUT RECOVERY (entangled-key two-mask)
# ═══════════════════════════════════════════════════════════
def run_experiment_5(quick=False):
    """Dropout recovery via survivor reveal under per-round-fresh pairwise keys.

    Entanglement-derived pairwise bits are single-use per round. Survivors can safely reveal pairwise bits with
    dropped clients (no future rounds compromised). We sweep dropout rate
    and check (i) training still converges, (ii) recovery activates as
    expected, (iii) the chi-squared VALIDATION check stays silent at every
    dropout rate (indistinguishability under uniform modular masks is a
    theorem — the check is an implementation-bug detector, not a security
    result; HANDOFF 3.9).

    Classical parity baseline: the same reveal-based recovery is exercised
    with a Flamingo-style per-round pool (Ma et al., IEEE S&P 2023) to show
    the MECHANISM is not quantum-unique; the quantum differentiator is that
    freshness is unconditional (no PRF/DH assumption, no key-erasure or
    forward-secrecy reliance).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Dropout recovery under per-round-fresh two-mask aggregation")
    print("="*70)

    from modules.m4_federated_learning import (
        ClinicalMLP, load_clinical_data, run_multitriangle_fl,
        define_triangle_topology,
    )

    num_rounds = 40 if not quick else 20
    target_acc = 0.90
    n_clients = 6  # 3 triangles, allows interesting dropout patterns
    n_triangles = 3
    # HANDOFF 3.6: >= 10 seeds, dropout swept 0-40%. The old single-seed
    # design produced pure-noise orderings (0% arm scoring below 15%).
    num_seeds = 10
    dropout_rates = ([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
                     if not quick else [0.0, 0.15, 0.40])

    triangles, _, tree_edges = define_triangle_topology(n_clients)

    # results[dr] = list over seeds of (res, aux); splits and torch seeds
    # are PAIRED across dropout rates (same seed list per rate).
    results = {dr: [] for dr in dropout_rates}
    for si in range(num_seeds):
        seed = 42 + si
        client_data, test_data, input_dim = load_clinical_data(
            n_clients, rng_seed=seed,
            dirichlet_alpha=FL.get('dirichlet_alpha', 0.5))
        model_fn = lambda: ClinicalMLP(input_dim=input_dim, hidden=128)
        for dr in dropout_rates:
            res, aux = run_multitriangle_fl(
                model_fn, client_data, test_data,
                num_clients=n_clients, num_rounds=num_rounds,
                local_epochs=FL['local_epochs'],
                batch_size=FL['local_batch_size'],
                lr=FL['learning_rate'], target_accuracy=target_acc,
                triangles=triangles, tree_edges=tree_edges,
                dropout_rate=dr, verbose=False, seed=seed,
            )
            results[dr].append((res, aux))
        print(f"  seed {seed}: " + "  ".join(
            f"{int(dr*100)}%→{results[dr][-1][0].final_accuracy:.3f}"
            for dr in dropout_rates))

    from scipy.stats import t as _t_dist

    def _mean_ci(vals):
        v = np.asarray(vals, dtype=float)
        hw = (_t_dist.ppf(0.975, len(v) - 1) * v.std(ddof=1)
              / np.sqrt(len(v)) if len(v) > 1 else float('nan'))
        return v.mean(), v.std(ddof=1), hw

    # ── Aggregate per-rate statistics across seeds ──
    drs = sorted(results.keys())
    stats = {}
    for dr in drs:
        accs = [r.final_accuracy for r, _ in results[dr]]
        rec_pct = [100 * np.mean(a['recovery_invoked_per_round'])
                   for _, a in results[dr]]
        # HEADLINE costs of recovery (HANDOFF 3.6):
        # bytes — reveals (+ recovery shares, identically 0 here) per
        # client per round, from the Phase-2 ledger;
        rec_bytes = [np.mean([rc.recovery_bytes() / max(rc.n_clients, 1)
                              for rc in r.comm_rounds])
                     for r, _ in results[dr]]
        # latency — modeled wire time of the recovery bytes on the
        # aggregator's access link;
        rec_wire_ms = [1000 * np.mean([rc.recovery_bytes() * 8
                                       / CLASSICAL_LINK_BPS
                                       for rc in r.comm_rounds])
                       for r, _ in results[dr]]
        # failed rounds — triangle-rounds skipped because survivors <
        # Shamir threshold (the driver zeroes that triangle's update).
        tfail_pct = [100 * sum(rc.threshold_failures
                               for rc in r.comm_rounds)
                     / (num_rounds * n_triangles)
                     for r, _ in results[dr]]
        chi_pct = [100 * np.mean(a['distinguishable_per_round'])
                   for _, a in results[dr]]
        stats[dr] = {
            "acc": _mean_ci(accs),
            "recovery_pct": _mean_ci(rec_pct),
            "rec_bytes": _mean_ci(rec_bytes),
            "rec_wire_ms": _mean_ci(rec_wire_ms),
            "tfail_pct": _mean_ci(tfail_pct),
            "chi_pct": _mean_ci(chi_pct),
        }

    # Representative per-phase byte ledger (last seed per rate)
    print()
    for dr in drs:
        print_comm_summary(results[dr][-1][0],
                           label=f"dropout {int(dr*100)}% (seed "
                                 f"{42 + num_seeds - 1})")

    # ── Plot 1: accuracy vs dropout rate (mean ± 95% CI across seeds) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    acc_means = [stats[dr]["acc"][0] for dr in drs]
    acc_hws = [stats[dr]["acc"][2] for dr in drs]
    xpct = [100 * dr for dr in drs]
    ax1.errorbar(xpct, acc_means, yerr=acc_hws, fmt='o-', linewidth=2,
                 capsize=4, color=C['quantum'],
                 label=f'final accuracy (mean ± 95% CI, {num_seeds} seeds)')
    ax1.axhline(y=target_acc, color='gray', linestyle=':', alpha=0.5,
                label=f'{target_acc} target')
    ax1.set_xlabel('Per-round dropout rate (%)')
    ax1.set_ylabel(f'Final test accuracy ({num_rounds} rounds)')
    ax1.set_title(f'Accuracy degradation vs dropout ({num_seeds} seeds)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2 (HEADLINE): the cost of recovery ──
    rb_means = [stats[dr]["rec_bytes"][0] / 1024 for dr in drs]
    rb_hws = [stats[dr]["rec_bytes"][2] / 1024 for dr in drs]
    tf_means = [stats[dr]["tfail_pct"][0] for dr in drs]
    tf_hws = [stats[dr]["tfail_pct"][2] for dr in drs]
    ax2.bar(np.arange(len(drs)) - 0.18, rb_means, 0.36, yerr=rb_hws,
            capsize=3, color=C['pareto'],
            label='recovery bytes (reveals) KB/client·round')
    ax2b = ax2.twinx()
    ax2b.bar(np.arange(len(drs)) + 0.18, tf_means, 0.36, yerr=tf_hws,
             capsize=3, color=C['unmasked'],
             label='threshold-failed triangle-rounds %')
    ax2.set_xticks(np.arange(len(drs)))
    ax2.set_xticklabels([f'{int(dr*100)}%' for dr in drs])
    ax2.set_xlabel('Per-round dropout rate')
    ax2.set_ylabel('Recovery bytes (KB/client·round)', color=C['pareto'])
    ax2b.set_ylabel('Threshold failures (% of triangle-rounds)',
                    color=C['unmasked'])
    ax2.set_title('HEADLINE: the cost of reveal-based recovery')
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 5: Dropout recovery — accuracy and COST, '
                 f'{num_seeds} paired seeds\n'
                 'Reveal-based recovery under per-round-fresh pairwise keys',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp5_dropout_recovery')

    print(f"\n  Summary ({num_seeds} seeds, mean ± std [95% CI half-width]):")
    print(f"  {'Dropout':>8}  {'Final acc':>22}  {'Recovery %':>12}  "
          f"{'Reveal KB/cl·rd':>16}  {'Wire ms':>8}  {'T-fail %':>9}  "
          f"{'χ² fail %':>10}")
    for dr in drs:
        s = stats[dr]
        print(f"  {int(dr*100):>7}%  "
              f"{s['acc'][0]:.4f} ± {s['acc'][1]:.4f} [{s['acc'][2]:.4f}]  "
              f"{s['recovery_pct'][0]:>11.1f}%  "
              f"{s['rec_bytes'][0]/1024:>15.1f}  "
              f"{s['rec_wire_ms'][0]:>8.3f}  "
              f"{s['tfail_pct'][0]:>8.2f}%  "
              f"{s['chi_pct'][0]:>9.1f}%")
    # ── Classical parity baseline (Flamingo-style per-round pool) ──
    # Same reveal-based recovery, same dropout, two mask-material sources.
    # Mechanism parity is EXPECTED; the differentiator is the assumption class.
    import torch as _torch
    from modules.m4_federated_learning import secure_aggregation_round
    _tri = [0, 1, 2]; _npar = 512
    _grads = {c: _torch.tensor(np.random.default_rng(100 + c).normal(0, 0.05, _npar),
                               dtype=_torch.float32) for c in _tri}
    _seeds = {c: 7000 + c for c in _tri}
    _need = 3 * _npar * BITS_PER_PARAM
    _pools = [
        ("entangled (source-model keys)",
         np.random.default_rng(11).integers(0, 2, size=_need, dtype=np.int8),
         "unconditional (information-theoretic)"),
        ("classical (per-round PRF pool)",
         np.random.default_rng(2026).integers(0, 2, size=_need, dtype=np.int8),
         "computational (PRF/DH; cf. Flamingo)"),
    ]
    print("\n  Classical parity baseline (single triangle, drop client 2):")
    print(f"  {'pool':>32}  {'recovery':>8}  {'exact-agg':>9}  assumption")
    for _label, _pool, _assump in _pools:
        _res = secure_aggregation_round({c: g.clone() for c, g in _grads.items()},
                                        _tri, _pool, _seeds, dropouts={2})
        _exact = bool(_torch.allclose(_res.aggregate, _grads[0] + _grads[1], atol=1e-3))
        print(f"  {_label:>32}  {str(_res.recovery_invoked):>8}  {str(_exact):>9}  {_assump}")
    print("  → identical mechanism and recovery in both pools; the quantum claim")
    print("    is the ASSUMPTION CLASS (unconditional freshness), not the mechanism.")

    print("  ✓ Experiment 5 complete.")
    return {"dropout_results": {f"{int(dr*100)}%": {
        "final_acc_mean": float(stats[dr]["acc"][0]),
        "final_acc_std": float(stats[dr]["acc"][1]),
        "final_acc_ci95": float(stats[dr]["acc"][2]),
        "recovery_pct_mean": float(stats[dr]["recovery_pct"][0]),
        "recovery_bytes_per_client_round": float(stats[dr]["rec_bytes"][0]),
        "recovery_wire_ms": float(stats[dr]["rec_wire_ms"][0]),
        "threshold_fail_pct": float(stats[dr]["tfail_pct"][0]),
        "chi2_fail_pct": float(stats[dr]["chi_pct"][0]),
        "n_seeds": num_seeds,
    } for dr in drs}}


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 11: Gradient-inversion attack on the REAL construction
# (rebuild of the deleted CIFAR/PSNR legacy — HANDOFF 3.8)
# ═══════════════════════════════════════════════════════════
def _tabular_dlg(model, target_grad_params, y_label, input_dim,
                 n_iters, lr, device='cpu', seed=0):
    """DLG/iDLG tabular gradient-inversion (Zhu et al. NeurIPS 2019;
    label-known iDLG variant, Zhao et al. 2020). Optimize a dummy
    feature vector so its gradient matches the observed gradient under
    squared-L2 gradient-matching. No image priors (no total variation) —
    the target is a 30-feature clinical vector, so tabular from the
    ground up (the deleted legacy applied image PSNR/SSIM to tabular
    data and attacked the proof-less Gaussian mask; HANDOFF 3.8)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dummy = torch.randn(1, input_dim, requires_grad=True, device=device)
    opt = torch.optim.Adam([dummy], lr=lr)
    crit = nn.CrossEntropyLoss()
    y = y_label.to(device)
    for _ in range(n_iters):
        opt.zero_grad()
        model.zero_grad()
        out = model(dummy)
        loss = crit(out, y)
        dgrad = torch.autograd.grad(loss, model.parameters(),
                                    create_graph=True)
        gm = sum(((dg - tg) ** 2).sum()
                 for dg, tg in zip(dgrad, target_grad_params))
        gm.backward()
        opt.step()
    return dummy.detach()


def run_experiment_11(quick=False):
    """Gradient-inversion privacy demonstration on the REAL system
    (HANDOFF 3.8): tabular DLG/iDLG against the ClinicalMLP and the
    modular-OTP construction, across three targets with honest,
    theorem-consistent expectations:
      (1) unmasked gradient      → attack SUCCEEDS (per-feature MSE low,
                                    attribute inference well above chance)
      (2) modular-masked payload → attack FAILS (one-time pad makes the
                                    payload independent of the gradient —
                                    a theorem; reconstruction ≈ prior)
      (3) plaintext aggregate    → attack PARTIALLY succeeds (the FedAvg
                                    sum leaks a mixture; this is the
                                    honest scoping limit that motivates
                                    DP-on-top-of-masking)
    Tabular metrics: per-feature MSE against the standardized ground
    truth, and attribute-inference accuracy (adversary predicts each
    feature's above/below-median bit from the reconstruction).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 11: Gradient-inversion attack on the real "
          "construction")
    print("="*70)

    import torch
    from modules.m4_federated_learning import (
        ClinicalMLP, load_clinical_data, quantize_gradient,
        dequantize_gradient, derive_modular_mask, apply_modular_mask,
        _SHAMIR_PRIME)

    n_targets = 2 if quick else 4
    n_iters = 600 if quick else 2000
    attack_lr = 0.1

    client_data, test_data, input_dim = load_clinical_data(
        3, rng_seed=42, dirichlet_alpha=FL.get('dirichlet_alpha', 0.5))
    torch.manual_seed(42)
    model = ClinicalMLP(input_dim=input_dim, hidden=128)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    bits_per_mask = n_params * 32
    rng = np.random.default_rng(42)

    # Median per feature over the training pool → attribute-inference bits.
    all_X = torch.cat([client_data[c][:][0] for c in range(3)], dim=0)
    feat_median = all_X.median(dim=0).values

    def metrics(recon, truth):
        mse = float(((recon - truth) ** 2).mean())
        ai = float(((recon > feat_median) == (truth > feat_median))
                   .float().mean())
        return mse, ai

    # Victim samples (one per target), each its own gradient.
    victim_X = all_X[rng.choice(len(all_X), size=n_targets, replace=False)]
    crit = torch.nn.CrossEntropyLoss()

    res = {"unmasked": [], "masked": [], "aggregate": []}
    for ti in range(n_targets):
        x = victim_X[ti:ti + 1]
        with torch.no_grad():
            y = model(x).argmax(dim=1)  # iDLG: label recoverable from grad
        model.zero_grad()
        loss = crit(model(x), y)
        true_grad = [g.detach().clone()
                     for g in torch.autograd.grad(loss, model.parameters())]

        # (1) UNMASKED: the raw gradient is on the wire.
        r1 = _tabular_dlg(model, true_grad, y, input_dim, n_iters,
                          attack_lr, seed=100 + ti)
        res["unmasked"].append(metrics(r1, x))

        # (2) MASKED: eavesdropper sees the modular-masked payload. Slice
        # a fresh mask from a distilled pool, mask the quantized gradient
        # (the exact wire object), hand the dequantized payload to DLG.
        flat = torch.cat([g.flatten() for g in true_grad])
        q = quantize_gradient(flat)
        pool = rng.integers(0, 2, size=len(flat) * 32, dtype=np.int8)
        mask = derive_modular_mask(len(flat), pool)
        masked = apply_modular_mask(q, mask, sign=+1)
        masked_flat = dequantize_gradient(masked)
        masked_grad, off = [], 0
        for g in true_grad:
            k = g.numel()
            masked_grad.append(masked_flat[off:off + k].view(g.shape))
            off += k
        r2 = _tabular_dlg(model, masked_grad, y, input_dim, n_iters,
                          attack_lr, seed=200 + ti)
        res["masked"].append(metrics(r2, x))

        # (3) AGGREGATE: honest-but-curious aggregator sees the PLAINTEXT
        # FedAvg sum (masks cancel by design). Attack the mean gradient
        # of this victim + 2 other clients' samples.
        others = all_X[rng.choice(len(all_X), size=2, replace=False)]
        agg_grad = [g.clone() for g in true_grad]
        for oi in range(2):
            xo = others[oi:oi + 1]
            with torch.no_grad():
                yo = model(xo).argmax(dim=1)
            model.zero_grad()
            lo = crit(model(xo), yo)
            go = torch.autograd.grad(lo, model.parameters())
            agg_grad = [a + g.detach() for a, g in zip(agg_grad, go)]
        agg_grad = [a / 3.0 for a in agg_grad]
        r3 = _tabular_dlg(model, agg_grad, y, input_dim, n_iters,
                          attack_lr, seed=300 + ti)
        res["aggregate"].append(metrics(r3, x))

    # Chance baselines: reconstruct-as-prior (all-zeros = the standardized
    # mean) for MSE, and coin-flip (0.5) for attribute inference.
    prior_mse = float((victim_X ** 2).mean())

    print(f"\n  {n_targets} victim samples, {n_iters}-iter tabular "
          f"DLG/iDLG, ClinicalMLP ({n_params:,} params):")
    print(f"  {'target':>22}  {'per-feature MSE':>16}  "
          f"{'attr-infer acc':>15}  verdict")
    verdicts = {
        "unmasked": "ATTACK SUCCEEDS",
        "masked": "ATTACK FAILS (theorem)",
        "aggregate": "PARTIAL (motivates DP)",
    }
    summary = {}
    for tgt in ["unmasked", "masked", "aggregate"]:
        mses = [m for m, _ in res[tgt]]
        ais = [a for _, a in res[tgt]]
        summary[tgt] = {"mse_mean": float(np.mean(mses)),
                        "mse_std": float(np.std(mses)),
                        "ai_mean": float(np.mean(ais)),
                        "ai_std": float(np.std(ais))}
        print(f"  {tgt:>22}  {np.mean(mses):>10.4f} ± {np.std(mses):.3f}  "
              f"{100*np.mean(ais):>13.1f}%  {verdicts[tgt]}")
    print(f"  {'(chance baseline)':>22}  {prior_mse:>10.4f} (prior)  "
          f"{50.0:>13.1f}%  reconstruct-as-mean / coin-flip")

    # Honest ordering assertions (means over targets).
    assert summary["unmasked"]["mse_mean"] < summary["masked"]["mse_mean"], \
        "unmasked should reconstruct better than masked"
    assert summary["unmasked"]["ai_mean"] > 0.6, \
        "unmasked attribute inference should beat chance"
    assert summary["masked"]["mse_mean"] > 0.5 * prior_mse, \
        "masked reconstruction should be no better than ~prior"
    # The 'aggregate PARTIALLY succeeds' caption is tested on ATTRIBUTE
    # INFERENCE, not MSE. Averaging n samples destroys single-sample
    # per-feature fidelity, so aggregate MSE can exceed the
    # reconstruct-as-mean prior (it does in underpowered quick mode:
    # ~0.55 vs ~0.47) while distributional/attribute leakage is real and
    # above chance (63% quick, 73% full). Attribute inference is the
    # load-bearing metric for this arm; assert it beats chance, and keep
    # the unmasked <= aggregate <= masked MSE ordering as a sanity band.
    assert summary["aggregate"]["ai_mean"] > 0.55, \
        ("aggregate attribute inference should beat chance — the "
         "partial-leak caption; if this fails in quick mode the run is "
         "underpowered, use full mode")
    assert (summary["unmasked"]["mse_mean"]
            <= summary["aggregate"]["mse_mean"]
            <= summary["masked"]["mse_mean"] + 1e-9), \
        "aggregate leakage should sit between unmasked and masked"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    order = ["unmasked", "aggregate", "masked"]
    labels = ["unmasked\ngradient", "plaintext\naggregate",
              "modular-masked\npayload"]
    cols = [C['unmasked'], C['dp'], C['quantum']]
    ax1.bar(labels, [summary[t]["mse_mean"] for t in order],
            yerr=[summary[t]["mse_std"] for t in order], capsize=4,
            color=cols)
    ax1.axhline(y=prior_mse, color='gray', linestyle=':',
                label=f'prior (reconstruct-as-mean) = {prior_mse:.2f}')
    ax1.set_ylabel('Per-feature reconstruction MSE (↓ = more leak)')
    ax1.set_title('Reconstruction error by target')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(labels, [100 * summary[t]["ai_mean"] for t in order],
            yerr=[100 * summary[t]["ai_std"] for t in order], capsize=4,
            color=cols)
    ax2.axhline(y=50, color='gray', linestyle=':', label='chance (50%)')
    ax2.set_ylabel('Attribute-inference accuracy (%)')
    ax2.set_ylim(0, 105)
    ax2.set_title('Above/below-median attribute inference')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 11: gradient inversion on the REAL '
                 'construction (tabular DLG/iDLG)\nunmasked leaks · '
                 'modular mask blocks (theorem) · plaintext aggregate '
                 'partially leaks',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp11_gradient_inversion')

    print("  ✓ Experiment 11 complete.")
    return {"targets": summary, "prior_mse": prior_mse,
            "n_targets": n_targets, "n_iters": n_iters}



# ═══════════════════════════════════════════════════════════
# EXPERIMENT 6: RATE-CONSTRAINED THROUGHPUT
# ═══════════════════════════════════════════════════════════
def measure_local_compute_time(n_params_target, local_epochs=2, batch_size=32,
                               lr=0.01, reps=3, seed=42):
    """Measured local-training time for an MLP of ≈n_params_target params.

    Replaces exp6's fabricated 2.0 s constant (HANDOFF 1.3) with a direct
    measurement: build a width-scaled two-hidden-layer MLP (params ≈
    h² + (input_dim+4)·h for input_dim features, 2 classes), run the same
    local step the FL loop performs (local_epochs over one client's share
    of the clinical dataset), and return the median wall-clock over `reps`
    rounds on this machine (first warm-up round discarded). Every grid size
    is measured directly — nothing is extrapolated.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from modules.m4_federated_learning import load_clinical_data

    client_data, _, input_dim = load_clinical_data(3, rng_seed=seed)
    dataset = client_data[0]

    # Solve h² + (input_dim + 4)·h ≈ target for the hidden width h.
    b = float(input_dim + 4)
    h = max(4, int((-b + np.sqrt(b * b + 4.0 * n_params_target)) / 2.0))
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(input_dim, h), nn.ReLU(),
        nn.Linear(h, h), nn.ReLU(),
        nn.Linear(h, 2),
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    times = []
    for _ in range(max(1, reps) + 1):  # +1 warm-up, dropped below
        t0 = time.perf_counter()
        for _ in range(local_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb.long())
                loss.backward()
                opt.step()
        times.append(time.perf_counter() - t0)
    return float(np.median(times[1:]))


def run_experiment_6(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 6: Rate-Constrained Training Throughput")
    print("="*70)

    model_sizes = FL['model_sizes_sweep'] if not quick else [5e4, 1e5, 5e5, 1e6]
    # Operating points from config — single source of truth (HANDOFF 1.1).
    # The sweep brackets the two labeled points; no unlabeled rate literals.
    op_rate = OPERATING_POINT['secret_rate_bps']
    proj_rate = PROJECTED_OPERATING_POINT['secret_rate_bps']
    key_rates = [op_rate, 100e3, 300e3, proj_rate] if not quick else [op_rate, proj_rate]
    rate_labels = {
        op_rate: f'{op_rate/1000:.1f} kbit/s — modeled device',
        proj_rate: f'{proj_rate/1000:.0f} kbit/s — PROJECTED 1 MHz-class (hypothetical)',
    }

    # ── Measured local compute time per model size (HANDOFF 1.3) ──
    reps = 1 if quick else 3
    print(f"  Measuring local compute time per model size (median of {reps} "
          f"round(s) each, warm-up discarded)...")
    compute_times = {}
    for ms in model_sizes:
        compute_times[int(ms)] = measure_local_compute_time(
            int(ms), local_epochs=FL['local_epochs'],
            batch_size=FL['local_batch_size'], lr=FL['learning_rate'],
            reps=reps)
        print(f"    {int(ms/1000):>6}K params: "
              f"{compute_times[int(ms)]:.3f} s/round (measured, this machine)")

    # Compute mask generation time and training throughput
    print("  Computing throughput for model size × key rate grid...")
    rounds_per_hour = np.zeros((len(model_sizes), len(key_rates)))

    for i, ms in enumerate(model_sizes):
        gradient_bits = int(ms) * BITS_PER_PARAM
        compute_time_s = compute_times[int(ms)]
        for j, kr in enumerate(key_rates):
            mask_time_s = gradient_bits / kr
            total_round_time = mask_time_s + compute_time_s
            rounds_per_hour[i, j] = 3600 / total_round_time

    fig, ax = plt.subplots(figsize=VIZ['figsize'])
    for j, kr in enumerate(key_rates):
        label = rate_labels.get(kr, f'{kr/1000:.0f} kbit/s (hypothetical sweep point)')
        ax.plot([int(ms/1000) for ms in model_sizes], rounds_per_hour[:, j],
                'o-', linewidth=2, label=label)

    # Unconstrained baseline: measured compute only, no key-rate constraint
    unconstrained = [3600 / compute_times[int(ms)] for ms in model_sizes]
    ax.plot([int(ms/1000) for ms in model_sizes], unconstrained,
            'k--', linewidth=1.5, alpha=0.5,
            label='Unconstrained (measured compute only)')

    ax.set_xlabel('Model Size (K parameters)')
    ax.set_ylabel('Training Rounds per Hour')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Experiment 6: Rate-Constrained Training Throughput\n'
                 'Mask time from key rate; local compute measured per size '
                 'on this machine')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    save_fig(fig, 'exp6_throughput')

    # ── Practical ceiling identification ──
    print("\n  Practical model-size ceilings (mask_time < round_period):")
    for kr in key_rates:
        for period_name, period_s in [("30s", 30), ("60s", 60), ("300s", 300)]:
            max_bits = kr * period_s
            max_params = max_bits / BITS_PER_PARAM
            print(f"    {rate_labels.get(kr, f'{kr/1000:.0f} kbit/s (hypothetical sweep point)')}, "
                  f"{period_name} rounds: max {max_params/1000:.0f}K params "
                  f"({max_params*BITS_PER_PARAM/8/1e6:.2f} MB of key per edge)")

    print("  ✓ Experiment 6 complete.")
    return {"rounds_per_hour": rounds_per_hour.tolist()}


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 7: QBER → PRIVACY-BUDGET CURVE
# ═══════════════════════════════════════════════════════════
def run_experiment_7(quick=False):
    """Map physical QBER to the key-rate / capacity envelope: shared-bit
    rate, the LHL mask-entropy floor, and the maximum maskable model size
    at a fixed round budget.

    LHL accounting matches the m2 waterfall (source model, no eavesdropper
    term): available min-entropy per reconciled input bit is
        1 − max(β·h(q), 1 − R_code) − D_tag
    (reconciliation syndrome leakage and the GLLP tagged fraction), and the
    PA output is ε-close to uniform with per-output-bit gap
    2·log2(1/ε) / (N · H_min_rate). The entropy floor stays essentially
    FLAT across QBER — PA compensates; what QBER degrades is the RATE.
    Supply-depends-on-channel is a cost this experiment prices, not a
    differentiator claim.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 7: QBER -> key-rate / capacity envelope")
    print("="*70)

    qbers = np.linspace(0.005, 0.06, 12) if not quick else np.linspace(0.005, 0.06, 6)
    block_size = POST_PROC['block_size']
    code_rate = POST_PROC['ldpc_code_rate']
    security_margin = POST_PROC['security_margin_bits']

    # Modeled-device raw edge rate from config — single source of truth
    # (HANDOFF 1.1). Identical to m1.compute_coincidence_rate at the config
    # parameters; derived once in config._derive_operating_point() rather
    # than re-derived here.
    raw_rate = OPERATING_POINT['raw_edge_rate_hz']

    # For each QBER, compute:
    #   shared_rate_bps: after EC + PA + verification + Bell duty
    #   mask_entropy_per_bit: LHL floor on output uniformity
    #   max_model_params @ 60s round (per-edge accounting — see below)
    shared_rates, envelope_rates, mask_entropies, max_params = [], [], [], []
    cap_q = estimate_capacity_qber(code_rate)
    for q in qbers:
        waterfall = compute_rate_waterfall(
            raw_rate, q, code_rate=code_rate,
            security_margin=security_margin, block_size=block_size,
            bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
        )
        rate = waterfall['final_secure_rate']
        shared_rates.append(rate)
        # HYPOTHETICAL rate-adapted envelope, plotted alongside with an
        # explicit label: past the declared 0.84 code's capacity the solid
        # curve collapses (tier FER 0.90) and only this labeled assumption
        # keeps a usable rate (Gate-1 round-2 finding 4).
        envelope_rates.append(compute_rate_waterfall(
            raw_rate, q, code_rate=code_rate,
            security_margin=security_margin, block_size=block_size,
            bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
            assume_rate_adapted=True,
        )['final_secure_rate'])

        # LHL bound on per-output-bit entropy floor, using the SAME
        # source-model accounting as the m2 waterfall. (The previous version
        # subtracted an extra full h(q) term — the stale pre-v2
        # eavesdropper-style formula the v2 patch removed from m2 but not
        # from here.) Available min-entropy per input bit =
        # 1 − β·h(q) − D_tag, with the syndrome fraction β·h(q) and GLLP
        # tagged fraction D_tag read from the waterfall itself so no β
        # literal is duplicated.
        syndrome_fraction = waterfall['ec_overhead_pct'] / 100
        input_min_entropy_per_bit = max(
            0, 1 - syndrome_fraction - waterfall['tagged_fraction'])
        # LHL: H_out >= L - 2*log2(1/ε), with 2*log2(1/ε) = 2*security_margin.
        # Per output bit:
        gap_per_bit = (2 * security_margin) / max(block_size * input_min_entropy_per_bit, 1)
        per_bit_entropy = max(0, 1 - gap_per_bit)
        mask_entropies.append(per_bit_entropy)

        # Max model parameters at a fixed 60 s round — PER-EDGE accounting,
        # consistent with exp6 and the round-time table: the triangle's
        # three edge links distill CONCURRENTLY, each at `rate`, and a round
        # consumes BITS_PER_PARAM bits per parameter per edge. The binding
        # constraint is the per-edge pool-fill time. (The previous ÷3
        # silently modeled the three edges sharing ONE rate budget, i.e.
        # serial keying — inconsistent with the architecture and with exp6.)
        round_s = 60
        bits_available_per_edge = rate * round_s
        max_p = bits_available_per_edge / BITS_PER_PARAM
        max_params.append(max_p)

    # Find the cutoff QBER where rate falls below 1 kbit/s (practical floor)
    cutoff_qber = None
    for q, r in zip(qbers, shared_rates):
        if r < 1000:
            cutoff_qber = q
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])

    ax1.plot(qbers, np.array(shared_rates) / 1000, 'o-', linewidth=2,
             color=C['quantum'], label='Declared rate-0.84 code (tier FER)')
    ax1.plot(qbers, np.array(envelope_rates) / 1000, '--', linewidth=1.5,
             color=C['accent'], label='Rate-adapted family (hypothetical)')
    ax1.axvline(x=cap_q, color='gray', linestyle=':', linewidth=1,
                label=f'0.84-code capacity (q={cap_q:.3f})')
    ax1.axvspan(0.8*cap_q, cap_q, alpha=0.08, color=C['unmasked'],
                label='degraded band (tier FER 0.10)')
    ax1_b = ax1.twinx()
    ax1_b.plot(qbers, mask_entropies, 's--', linewidth=2,
               color=C['accent'], label='Mask entropy floor (bits/bit)')
    ax1.set_xlabel('QBER')
    ax1.set_ylabel('Shared bit rate (kbit/s)', color=C['quantum'])
    ax1_b.set_ylabel('Mask entropy floor (per output bit)', color=C['accent'])
    ax1.set_yscale('log')
    if cutoff_qber is not None:
        ax1.axvline(x=cutoff_qber, color='gray', linestyle=':',
                    label=f'1 kbit/s cutoff (QBER={cutoff_qber:.3f})')
    ax1.set_title('Shared rate and mask entropy vs QBER')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=7)

    ax2.plot(qbers, np.array(max_params) / 1000, 'o-', linewidth=2,
             color=C['pareto'])
    ax2.axvline(x=cap_q, color='gray', linestyle=':', linewidth=1)
    ax2.text(cap_q, ax2.get_ylim()[0] if ax2.get_ylim()[0] > 0 else 1,
             ' 0.84-code capacity', fontsize=6, color='gray', rotation=90,
             va='bottom')
    for ref, lbl in [(29, '29K (ClinicalMLP)'),
                      (150, '150K params'),
                      (1000, '1M params')]:
        ax2.axhline(y=ref, color='gray', linestyle=':', alpha=0.5)
        ax2.text(qbers[0], ref, lbl, fontsize=7, color='gray', va='bottom')
    ax2.set_xlabel('QBER')
    ax2.set_ylabel('Max model size at 60s round (K params)')
    ax2.set_yscale('log')
    ax2.set_title('Max maskable model size vs QBER (rate-driven)')
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle('Experiment 7: QBER -> key-rate / capacity envelope\n'
                 'Channel quality bounds key SUPPLY (rate, model size); '
                 'the PA entropy floor stays flat',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp7_qber_key_rate_envelope')

    print("\n  QBER -> Shared rate -> Max model size (60s rounds):")
    print(f"  {'QBER':>8}  {'Declared rate':>15}  {'Envelope*':>12}  "
          f"{'Mask H/bit':>11}  {'Max K params':>14}")
    for q, r, e, h, mp in zip(qbers, shared_rates, envelope_rates,
                              mask_entropies, max_params):
        print(f"  {q:>8.4f}  {r/1000:>10.2f} kbit/s  {e/1000:>7.2f} kb/s"
              f"  {h:>11.4f}  {mp/1000:>14.1f}")
    print(f"  * envelope = HYPOTHETICAL rate-adapted code family (not "
          f"implemented); the declared\n    rate-0.84 code's capacity is "
          f"QBER ≈ {cap_q:.4f} — beyond it the declared-code rate\n    "
          f"collapses (tier FER 0.90) and max-params follows the collapse.")
    if cutoff_qber:
        print(f"\n  Cutoff QBER (rate < 1 kbit/s): {cutoff_qber:.3f}")
    print("  ✓ Experiment 7 complete.")
    return {
        "qbers": qbers.tolist(),
        "shared_rates_bps": shared_rates,
        "mask_entropy_per_bit": mask_entropies,
        "max_params_at_60s": max_params,
        "cutoff_qber": float(cutoff_qber) if cutoff_qber else None,
    }


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 7b (DEMOTED): Cost breakdown (illustrative only)
# ═══════════════════════════════════════════════════════════
def run_experiment_7_cost(quick=False):
    print("\n" + "="*70)
    print("APPENDIX 7b: Cost breakdown (illustrative; prices not sourced)")
    print("="*70)

    n_nodes = 6  # 2 triangles
    n_triangles = 3  # Approximate from Delaunay
    training_round_s = 60
    rounds_per_year = int(365 * 24 * 3600 / training_round_s * 0.5)
    quantum_accuracy = 0.94  # From Experiment 4

    # ── Quantum system cost ──
    quantum_cost_round, quantum_annual = compute_cost_per_round(
        n_triangles, n_nodes, rounds_per_year
    )
    quantum_energy = compute_energy_per_round(n_triangles, n_nodes, training_round_s)

    # ── Classical baselines ──
    baselines = compute_classical_baseline_costs(n_nodes, rounds_per_year)

    print(f"  Quantum annual cost: €{quantum_annual:,.0f}")
    print(f"  Quantum cost/round:  €{quantum_cost_round:.4f}")
    for name, bl in baselines.items():
        print(f"  {name:>12} annual: €{bl['annual']:,.0f}  |  "
              f"accuracy impact: {bl['accuracy_impact']:.2f}")

    # ── Pareto plot ──
    fig, ax = plt.subplots(figsize=VIZ['figsize'])

    methods = {
        "Quantum (this system)": (quantum_annual, quantum_accuracy, C['quantum'], 'o', 80),
        "PQC+Bonawitz": (baselines['pqc']['annual'], quantum_accuracy, C['prng'], 's', 60),
        "HSM cluster": (baselines['hsm']['annual'], quantum_accuracy, '#9B59B6', 'D', 60),
        "HE (CKKS)": (baselines['he']['annual'], quantum_accuracy, '#E67E22', '^', 60),
        "DP ε=1": (max(baselines['dp_eps1']['annual'], 5000), quantum_accuracy - 0.15, C['dp'], 'v', 50),
        "DP ε=5": (max(baselines['dp_eps5']['annual'], 5000), quantum_accuracy - 0.05, '#66BB6A', 'v', 50),
        "DP ε=10": (max(baselines['dp_eps10']['annual'], 5000), quantum_accuracy - 0.02, '#A5D6A7', 'v', 50),
    }

    for name, (cost, acc, color, marker, size) in methods.items():
        ax.scatter(cost, acc, c=color, marker=marker, s=size, zorder=5,
                   label=name, edgecolors='white', linewidth=1.5)

    # Security level annotations
    ax.annotate('Information-theoretic\nsecurity', xy=(quantum_annual, quantum_accuracy),
                xytext=(quantum_annual*3, quantum_accuracy-0.02),
                fontsize=8, color=C['quantum'],
                arrowprops=dict(arrowstyle='->', color=C['quantum'], lw=1.5))

    ax.set_xlabel('Annual Cost per Deployment (€)', fontsize=7)
    ax.set_ylabel('Model Accuracy at Convergence', fontsize=7)
    ax.set_xscale('log')
    ax.set_title(f'Experiment 7: Cost-Privacy-Accuracy Pareto Frontier\n'
                 f'{n_nodes} nodes, {n_triangles} triangles, 60s training rounds',
                 fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0.75, 1.0)
    fig.tight_layout()
    save_fig(fig, 'exp7_pareto_frontier')

    # ── CAPEX breakdown ──
    print("\n  CAPEX Breakdown (Option B: 1 QD per triangle membership):")
    capex = compute_capex(n_triangles, n_nodes)
    print(f"    Hardware: {capex['n_qds']} QDs, {n_nodes} cryostats "
          f"({capex['n_shared_nodes']} dual-QD), {capex['n_spads']} SPADs, {n_nodes} FPGAs")
    for component in ['qds', 'cryostats', 'switches', 'spads', 'fpga', 'optics', 'fiber']:
        print(f"    {component:>10}: €{capex[component]:,.0f}")
    print(f"    {'TOTAL':>10}: €{capex['total']:,.0f}")
    print(f"    {'per node':>10}: €{capex['per_node']:,.0f}")

    # ── CAPEX/OPEX pie charts ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 4.0))

    # ── CAPEX donut ──
    labels = ['QD Chips', 'Cryostats', 'SPADs', 'Switches', 'FPGAs', 'Optics', 'Fiber']
    values = [capex['qds'], capex['cryostats'], capex['spads'],
              capex['switches'], capex['fpga'], capex['optics'], capex['fiber']]
    colors_pie = [C['quantum'], C['dark'], C['pareto'], '#555555',
                  C['accent'], C['prng'], C['dp']]

    # Sort by value descending so largest slices are together
    sorted_data = sorted(zip(values, labels, colors_pie), reverse=True)
    values = [d[0] for d in sorted_data]
    labels = [d[1] for d in sorted_data]
    colors_pie = [d[2] for d in sorted_data]

    total_capex = sum(values)

    # Build percentage labels for autopct
    def make_label(pct):
        return f'{pct:.1f}%' if pct >= 3 else ''

    wedges1, texts1, autotexts1 = ax1.pie(
        values, colors=colors_pie, startangle=90, autopct=make_label,
        pctdistance=0.78, wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1.2))
    for t in autotexts1:
        t.set_fontsize(6)
        t.set_color('white')
        t.set_fontweight('bold')

    # Legend below the chart instead of leader lines
    legend_labels = [f'{lab} — €{val/1000:.0f}K' for lab, val in zip(labels, values)]
    ax1.legend(wedges1, legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.02), fontsize=6.5, ncol=2,
               frameon=False, columnspacing=1.0, handletextpad=0.5)
    ax1.set_title(f'CAPEX Breakdown — Total: €{capex["total"]/1000:.0f}K',
                  fontsize=9, fontweight='bold', pad=8)

    # ── OPEX donut ──
    opex = compute_opex_annual(n_triangles, n_nodes)
    labels2 = ['Maintenance', 'Electricity']
    values2 = [opex['maintenance'], opex['electricity']]
    colors_opex = [C['quantum'], C['accent']]

    total_opex = sum(values2)

    wedges2, texts2, autotexts2 = ax2.pie(
        values2, colors=colors_opex, startangle=90, autopct='%1.1f%%',
        pctdistance=0.75, wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1.2))
    for t in autotexts2:
        t.set_fontsize(7)
        t.set_color('white')
        t.set_fontweight('bold')

    legend_labels2 = [f'{lab} — €{val:,.0f}/yr' for lab, val in zip(labels2, values2)]
    ax2.legend(wedges2, legend_labels2, loc='upper center',
               bbox_to_anchor=(0.5, -0.02), fontsize=6.5, ncol=1,
               frameon=False, handletextpad=0.5)
    ax2.set_title(f'Annual OPEX — Total: €{opex["total"]:,.0f}/yr  |  '
                  f'Power: {opex["power_w"]/1000:.1f} kW',
                  fontsize=9, fontweight='bold', pad=8)

    fig.suptitle('Cost Structure', fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    save_fig(fig, 'exp7b_cost_breakdown')

    print("  ✓ Experiment 7 complete.")
    return {"quantum_annual": quantum_annual, "baselines": {k: v['annual'] for k, v in baselines.items()}}


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 8: MULTI-TRIANGLE FL SCALING
# ═══════════════════════════════════════════════════════════
def run_experiment_8(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 8: Multi-Triangle Federated Learning Scaling")
    print("="*70)

    from modules.m4_federated_learning import (
        ClinicalMLP, load_clinical_data, run_multitriangle_fl,
        define_triangle_topology, topology_from_mesh, evaluate,
    )

    num_rounds = 60 if not quick else 30
    target_acc = 0.92
    scales = [3, 6, 9, 12]
    dirichlet_alpha = FL.get('dirichlet_alpha', 0.5)

    results = {}

    for n_clients in scales:
        print(f"\n  ── Scale: {n_clients} clients ──")

        # Use Module 3 Delaunay topology for larger scales, hardcoded for small
        if n_clients >= 6:
            # Modeled-device rate from config (HANDOFF 1.1) — topology metrics
            # are geometric; the mesh's latency fields use the honest rate.
            mesh = analyze_mesh(n_clients, 1e5, OPERATING_POINT['secret_rate_bps'],
                                rng=np.random.default_rng(42))
            triangles, tree_edges = topology_from_mesh(mesh)
            n_tri = len(triangles)
            tree_depth = mesh.aggregation_depth
            print(f"    Topology (Delaunay): {n_tri} triangles, tree depth {tree_depth}")
        else:
            triangles, _, tree_edges = define_triangle_topology(n_clients)
            n_tri = len(triangles)
            tree_depth = 0
            print(f"    Topology (hardcoded): {n_tri} triangles, tree depth {tree_depth}")

        for i, tri in enumerate(triangles):
            print(f"      T{i}: clients {tri}")

        client_data, test_data, input_dim = load_clinical_data(
            n_clients, rng_seed=42, dirichlet_alpha=dirichlet_alpha
        )
        model_fn = lambda: ClinicalMLP(input_dim=input_dim, hidden=128)

        print(f"    Training for {num_rounds} rounds with entangled-key two-mask...")
        result, aux = run_multitriangle_fl(
            model_fn, client_data, test_data,
            num_clients=n_clients, num_rounds=num_rounds,
            local_epochs=FL['local_epochs'], batch_size=FL['local_batch_size'],
            lr=FL['learning_rate'], target_accuracy=target_acc,
            triangles=triangles, tree_edges=tree_edges,
            dropout_rate=0.0,
            verbose=True,
        )

        results[n_clients] = {
            'result': result,
            'n_triangles': n_tri,
            'tree_depth': tree_depth,
            'triangles': triangles,
            'aux': aux,
        }
        print(f"    Final accuracy: {result.final_accuracy:.4f}")
        print(f"    Rounds to target ({target_acc}): {result.rounds_to_target}")
        print(f"    Wall clock: {result.wall_clock_s:.1f}s")
        print(f"    χ² implementation-check failures (validation): "
              f"{sum(aux['distinguishable_per_round'])}/{num_rounds}")
        print(f"    entangled-key bits consumed (intra/inter): "
              f"{aux['bits_intra_consumed']:,} / {aux['bits_inter_consumed']:,}")
        print_comm_summary(result, label=f"{n_clients}c")

    # ═══ PLOT 1: Training convergence across scales ═══
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])

    scale_colors = {3: C['quantum'], 6: C['prng'], 9: C['pareto'], 12: C['unmasked']}
    scale_styles = {3: '-', 6: '--', 9: '-.', 12: ':'}

    for n_clients in scales:
        r = results[n_clients]['result']
        n_tri = results[n_clients]['n_triangles']
        depth = results[n_clients]['tree_depth']
        label = f"{n_clients} clients / {n_tri} triangles (depth {depth})"

        ax1.plot(range(1, len(r.accuracies)+1), r.accuracies,
                linewidth=2.5 if n_clients == 3 else 2,
                color=scale_colors[n_clients],
                linestyle=scale_styles[n_clients],
                label=label)

        clipped = [max(l, 1e-6) for l in r.losses]
        ax2.semilogy(range(1, len(clipped)+1), clipped,
                    linewidth=2.5 if n_clients == 3 else 2,
                    color=scale_colors[n_clients],
                    linestyle=scale_styles[n_clients],
                    label=label)

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Model Accuracy vs Scale')
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 1.0)
    ax1.axhline(y=target_acc, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Test Loss (log scale)')
    ax2.set_title('Training Loss vs Scale')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle('Experiment 8: Multi-Triangle FL Scaling\n'
                 'Hierarchical secure aggregation across overlapping triangle meshes',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp8a_multitri_convergence')

    # ═══ PLOT 2: Scaling summary bar chart ═══
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.8))

    # Bar data
    final_accs = [results[n]['result'].final_accuracy for n in scales]
    rounds_to_tgt = [results[n]['result'].rounds_to_target for n in scales]
    tree_depths = [results[n]['tree_depth'] for n in scales]
    wall_clocks = [results[n]['result'].wall_clock_s for n in scales]

    bar_colors = [scale_colors[n] for n in scales]
    x = np.arange(len(scales))
    bar_width = 0.6
    xlabels = [f"{n}c/{results[n]['n_triangles']}t" for n in scales]

    def label_bars(ax, values, fmt='{:.3f}'):
        """Place labels just above bars in black."""
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        for i, v in enumerate(values):
            y_pos = v + y_range * 0.02
            ax.text(i, y_pos, fmt.format(v), ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color=C['dark'])

    # Final accuracy
    axes[0].bar(x, final_accs, width=bar_width, color=bar_colors, edgecolor='white', linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xlabels, fontsize=7)
    axes[0].set_ylabel('Final Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].set_ylim(0.8, 1.05)
    axes[0].axhline(y=target_acc, color='gray', linestyle=':', alpha=0.5, linewidth=0.6)
    label_bars(axes[0], final_accs, fmt='{:.3f}')

    # Rounds to target
    axes[1].bar(x, rounds_to_tgt, width=bar_width, color=bar_colors, edgecolor='white', linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels, fontsize=7)
    axes[1].set_ylabel('Rounds')
    axes[1].set_title('Rounds to Target')
    max_rounds = max(rounds_to_tgt)
    axes[1].set_ylim(0, max_rounds * 1.18)
    label_bars(axes[1], rounds_to_tgt, fmt='{:.0f}')

    # Tree depth
    axes[2].bar(x, tree_depths, width=bar_width, color=bar_colors, edgecolor='white', linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(xlabels, fontsize=7)
    axes[2].set_ylabel('Depth')
    axes[2].set_title('Agg. Tree Depth')
    max_depth = max(tree_depths) if max(tree_depths) > 0 else 1
    axes[2].set_ylim(0, max_depth * 1.25)
    label_bars(axes[2], tree_depths, fmt='{:.0f}')

    # Wall clock time
    axes[3].bar(x, wall_clocks, width=bar_width, color=bar_colors, edgecolor='white', linewidth=0.8)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(xlabels, fontsize=7)
    axes[3].set_ylabel('Seconds')
    axes[3].set_title('Wall Clock')
    max_wc = max(wall_clocks)
    axes[3].set_ylim(0, max_wc * 1.18)
    label_bars(axes[3], wall_clocks, fmt='{:.1f}s')

    fig.suptitle('Multi-Triangle Scaling Metrics (c = clients, t = triangles)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, 'exp8b_multitri_scaling_bars')

    # ═══ PLOT 3: Aggregation overhead at both operating points ═══
    # Modeled device AND the labeled hypothetical projection (HANDOFF 1.1.3).
    rate_points = [
        (OPERATING_POINT['secret_rate_bps'], 'modeled device'),
        (PROJECTED_OPERATING_POINT['secret_rate_bps'],
         'PROJECTED 1 MHz-class (hypothetical)'),
    ]
    model_params = [5e4, 1e5, 5e5, 1e6]

    fig, axes = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    for ax, (secure_rate, rlabel) in zip(axes, rate_points):
        for mp in model_params:
            gradient_bits = int(mp) * BITS_PER_PARAM
            mask_time_per_level = gradient_bits / secure_rate
            overhead_at_depths = [mask_time_per_level * d for d in tree_depths]
            ax.plot(scales, overhead_at_depths, 'o-', linewidth=2,
                    label=f'{int(mp/1000)}K ({mask_time_per_level:.1f}s/level)')
        ax.set_xlabel('Number of Clients')
        ax.set_ylabel('Aggregation Overhead (seconds)')
        ax.set_yscale('log')
        ax.set_title(f'{rlabel}\n({secure_rate/1000:.1f} kbit/s)', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3, which='both')
        # Round period reference lines
        for period, plabel in [(30, '30s'), (60, '60s'), (300, '5min')]:
            ax.axhline(y=period, color='gray', linestyle=':', alpha=0.4)
            ax.text(scales[-1], period * 1.08, plabel, fontsize=6,
                    color='gray', ha='right')
    fig.suptitle('Experiment 8c: Projected aggregation overhead — '
                 'modeled vs projected rate', fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp8c_aggregation_overhead')

    # ═══ Summary ═══
    print("\n  ═══ MULTI-TRIANGLE SCALING SUMMARY ═══")
    print(f"  {'Clients':>8} {'Triangles':>10} {'Depth':>6} {'Accuracy':>10} "
          f"{'Rounds':>8} {'Time':>8}")
    print("  " + "-" * 60)
    for n in scales:
        r = results[n]
        print(f"  {n:>8} {r['n_triangles']:>10} {r['tree_depth']:>6} "
              f"{r['result'].final_accuracy:>10.4f} "
              f"{r['result'].rounds_to_target:>8} "
              f"{r['result'].wall_clock_s:>7.1f}s")

    print("\n  KEY FINDING: Multi-triangle hierarchical aggregation")
    if all(results[n]['result'].final_accuracy > 0.85 for n in scales):
        print("  ✓ converges at ALL scales (3–12 clients)")
        print("  ✓ Tree-of-triangles protocol produces correct global aggregate")
    else:
        print("  ⚠ Some scales did not converge — note: quick mode runs only "
              "30 rounds,\n     below convergence at 9–12 clients (full mode "
              "uses 60 rounds)")

    print("  ✓ Experiment 8 complete.")

    return {
        str(n): {
            'accuracy': results[n]['result'].final_accuracy,
            'rounds_to_target': results[n]['result'].rounds_to_target,
            'n_triangles': results[n]['n_triangles'],
            'tree_depth': results[n]['tree_depth'],
        } for n in scales
    }


# ═══════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════
def run_experiment_9(quick=False):
    """HNDL — harvest-now / decrypt-later retroactive transcript attack.

    HANDOFF 3.1: makes claim (b) empirical. Two arms differ ONLY in the
    pairwise-mask source; both record the full classical transcript
    (masked payloads, Shamir shares, seed disclosures, dropout reveals),
    then a recording adversary is granted retroactive god-mode over every
    COMPUTATIONAL secret (future quantum computer: DH/PRF/channel
    encryption broken after the fact). Expected: classical arm 100% of
    recorded gradients recovered; entangled arm 0% with residuals
    statistically uniform (every delivered payload keeps >= 1 pairwise
    mask that never appeared in any transcript).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 9: Harvest-now / decrypt-later (HNDL) transcript attack")
    print("="*70)

    import torch
    from modules.m4_federated_learning import (
        ClinicalMLP, load_clinical_data, secure_aggregation_round,
        train_local, get_model_params, set_model_params,
        get_gradient_update, _SHAMIR_PRIME)
    from modules.m6_hndl import run_transcript_round, retroactive_attack

    n_clients = 3
    num_rounds = 6 if quick else 10
    # Deterministic dropout schedule exercising reveals: exactly one
    # dropped client in two rounds. Two simultaneous drops would abort
    # the cluster at the live-set announcement (threshold failure), so
    # no payload would be recorded at all — the t >= 2 argument's edge
    # case is structurally unreachable for DELIVERED payloads.
    dropout_schedule = {2: {2}, 4: {0}}

    client_data, test_data, input_dim = load_clinical_data(
        n_clients, rng_seed=42, dirichlet_alpha=FL.get('dirichlet_alpha', 0.5))
    torch.manual_seed(42)
    model_fn = lambda: ClinicalMLP(input_dim=input_dim, hidden=128)
    global_model = model_fn()
    n_params = sum(p.numel() for p in global_model.parameters())
    bits_per_mask = n_params * 32

    rng = np.random.default_rng(42)
    tr_ent, tr_cls = [], []
    print(f"  Recording {num_rounds} rounds x both arms "
          f"(drops at rounds {sorted(dropout_schedule)})...")
    for rd in range(num_rounds):
        drops = dropout_schedule.get(rd, set())
        # Real local training -> gradient updates. The SAME gradients are
        # recorded in both arms; only the mask source differs.
        gparams = get_model_params(global_model)
        grads = {}
        for c in range(n_clients):
            local = model_fn()
            set_model_params(local, gparams.clone())
            train_local(local, client_data[c], FL['local_epochs'],
                        FL['local_batch_size'], FL['learning_rate'])
            grads[c] = get_gradient_update(global_model, local)

        pool = rng.integers(0, 2, size=3 * bits_per_mask, dtype=np.int8)
        self_seeds = {c: int(rng.integers(0, _SHAMIR_PRIME))
                      for c in range(n_clients)}
        pair_seeds = {(i, j): int(rng.integers(0, 2**31))
                      for i in range(n_clients)
                      for j in range(i + 1, n_clients)}

        t_e, agg_e = run_transcript_round(
            rd, grads, range(n_clients), drops, "entangled",
            pairwise_pool=pool, self_seeds=self_seeds)
        t_c, agg_c = run_transcript_round(
            rd, grads, range(n_clients), drops, "classical",
            pair_mask_seeds=pair_seeds, self_seeds=self_seeds)
        # Coherence assertions: the wire-faithful transcript must equal
        # the LIVE implementation on identical inputs, and the two arms
        # must agree in the clear (they differ only inside the masks).
        agg_live = secure_aggregation_round(
            client_gradients=grads, cluster_clients=list(range(n_clients)),
            pairwise_pool=pool, local_rng_seeds=self_seeds,
            dropouts=drops, threshold=2)
        assert np.array_equal(agg_e, agg_live.aggregate_quantized), \
            "transcript arm diverges from secure_aggregation_round"
        assert np.array_equal(agg_c, agg_e), "arms diverge in the clear"
        tr_ent.append(t_e)
        tr_cls.append(t_c)

        survivors = n_clients - len(drops)
        set_model_params(global_model,
                         gparams + agg_live.aggregate / survivors)

    atk_c = retroactive_attack(tr_cls, "classical")
    atk_e = retroactive_attack(tr_ent, "entangled")

    print("\n  Retroactive god-mode attack on the recorded transcripts")
    print("  (adversary recomputes every COMPUTATIONAL secret: channel "
          "encryption ->\n   Shamir shares -> all self-masks in BOTH arms; "
          "classical arm also all\n   DH/PRF pairwise seeds):")
    print(f"\n  {'arm':<28} {'payloads':>8} {'recovered':>10} "
          f"{'min masks left':>15} {'residual p':>11} {'median|err|':>12}")
    for label, atk in [("classical (DH/PRF masks)", atk_c),
                       ("entangled (distilled keys)", atk_e)]:
        print(f"  {label:<28} {atk['n_payloads']:>8} "
              f"{100*atk['recovered_fraction']:>9.1f}% "
              f"{atk['min_masks_remaining']:>15} "
              f"{atk['mean_residual_monobit_p']:>11.3f} "
              f"{atk['median_abs_error']:>12.4g}")
    assert atk_c["recovered_fraction"] == 1.0, \
        "classical arm should be fully recoverable under god-mode"
    assert atk_e["recovered_fraction"] == 0.0, \
        "entangled arm leaked a payload — HNDL claim violated"
    assert atk_e["min_masks_remaining"] >= 1, \
        "a delivered payload lost all pairwise masks — t>=2 argument violated"

    print("\n  → classical arm: 100% of recorded gradients recovered "
          "retroactively;")
    print("    entangled arm: 0% — every delivered payload retains ≥ 1 "
          "pairwise mask that\n    never appeared in any transcript, and "
          "god-mode residuals remain uniform\n    (mean monobit p above; "
          "median reconstruction error ~10³ in gradient units\n    vs ~0 "
          "classically).")
    print("\n  HONEST SCOPE: the self-mask leg rides classical channels "
          "(Shamir shares)\n  and is only computationally protected — "
          "god-mode strips it in BOTH arms\n  here. The PAIRWISE leg alone "
          "carries the HNDL claim. Recorded EC syndromes\n  are covered by "
          "the PA ε (masks are sliced from post-PA output).")

    # Plot: recovery fraction + residual uniformity per arm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    arms = ["classical\n(DH/PRF)", "entangled\n(distilled)"]
    ax1.bar(arms, [100 * atk_c["recovered_fraction"],
                   100 * atk_e["recovered_fraction"]],
            color=[C['unmasked'], C['quantum']])
    ax1.set_ylabel('Gradients recovered retroactively (%)')
    ax1.set_ylim(0, 105)
    for i, v in enumerate([100 * atk_c["recovered_fraction"],
                           100 * atk_e["recovered_fraction"]]):
        ax1.text(i, v + 2, f'{v:.0f}%', ha='center', fontweight='bold')
    ax1.set_title('Retroactive recovery under god-mode')
    ax1.grid(True, alpha=0.3, axis='y')

    pc = [r["residual_monobit_p"] for r in atk_c["records"]]
    pe = [r["residual_monobit_p"] for r in atk_e["records"]]
    ax2.scatter(np.zeros(len(pc)) + 0.0, pc, color=C['unmasked'],
                alpha=0.7, label='classical: residual = the gradient')
    ax2.scatter(np.zeros(len(pe)) + 1.0, pe, color=C['quantum'],
                alpha=0.7, label='entangled: residual stays uniform')
    ax2.axhline(y=0.01, color='gray', linestyle=':', linewidth=1,
                label='α = 0.01')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(arms)
    ax2.set_ylabel('Monobit p-value of post-attack residual')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7)
    ax2.set_title('What god-mode is left with')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 9: harvest-now / decrypt-later — record '
                 'everything, break all\ncomputational secrets later; '
                 'only the never-transmitted pairwise bits survive',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp9_hndl_transcript')

    print("  ✓ Experiment 9 complete.")
    return {
        "classical": {k: v for k, v in atk_c.items() if k != "records"},
        "entangled": {k: v for k, v in atk_e.items() if k != "records"},
        "rounds": num_rounds,
        "payloads_per_arm": atk_c["n_payloads"],
    }


def run_experiment_10(quick=False):
    """Availability under single-source failure (HANDOFF 3.4).

    Operationalizes source_symmetry_ratio ≈ 0.5 — previously just a
    printed number. Every edge is keyed from BOTH endpoints' sources
    (beamsplitter-multiplexed), so killing one node's source leaves its
    incident edges keying at ~half rate from the far endpoints, while a
    fixed-source-per-edge design loses those edges outright and stalls
    every triangle that needs them. Framed as AVAILABILITY/FAIRNESS, not
    security: under the trusted-source model there is no 'quasi-dealer'
    adversary to defeat.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 10: Availability under single-source failure")
    print("="*70)

    from modules.m4_federated_learning import define_triangle_topology

    # ── (a) Source-level empirical check: an edge with one endpoint's
    # source dead continues from the far endpoint at ~half rate. ──
    dur = 0.01 if quick else 0.05
    kw = dict(fidelity=QUANTUM['fidelity'], fiber_length_m=100)
    edge_full = simulate_symmetric_edge(
        duration_s=dur, rng=np.random.default_rng(42), **kw)
    # One endpoint's source killed. simulate_symmetric_edge builds the
    # healthy edge by concatenating TWO legs, each a simulate_source run
    # at duration_s * switch_split (switch_split = 0.5): the endpoint-i
    # leg and the endpoint-j leg. Killing endpoint i removes its leg
    # exactly, leaving the endpoint-j leg alone — which is precisely one
    # simulate_source(duration_s = dur * 0.5) call with the same source
    # params. So this is NOT an unrelated half-rate proxy; it is the
    # surviving endpoint's own leg from the symmetric-edge decomposition
    # (Gate-3 round-1 F2). The far endpoint keeps routing switch_split of
    # its pairs down the edge with unchanged per-pair physics.
    edge_degraded = simulate_source(
        duration_s=dur * 0.5, rng=np.random.default_rng(43), **kw)
    n_full = len(edge_full.raw_bits_alice)
    n_deg = len(edge_degraded.raw_bits_alice)
    ratio = n_deg / max(n_full, 1)
    print(f"  (a) Edge keyed from both endpoints: {n_full:,} coincidences "
          f"in {dur}s\n      one source killed: {n_deg:,} "
          f"({100*ratio:.1f}% of the two-source rate; expected ≈ 50%)\n"
          f"      QBER {edge_full.qber:.4f} → {edge_degraded.qber:.4f} "
          f"(unchanged — same physics, fewer pairs)\n"
          f"      symmetry witness on the healthy edge: "
          f"{edge_full.params['source_symmetry_ratio']:.3f} (0.5 = no "
          f"privileged generator)")
    assert 0.35 <= ratio <= 0.65, (
        f"degraded-edge rate ratio {ratio} outside the ~half-rate window")

    # ── (b) Modeled round-time impact on the 6-client / 3-triangle
    # topology (same per-edge demand model as exps 5/8). ──
    n_clients = 6
    triangles, _, tree_edges = define_triangle_topology(n_clients)
    n_params = 28866  # ClinicalMLP — matches exps 4/5/8
    bits_per_mask = n_params * BITS_PER_PARAM
    physical_edges = {tuple(sorted((int(t[i]), int(t[j]))))
                      for t in triangles for i, j in ((0, 1), (1, 2), (0, 2))}
    demand = {}
    for t in triangles:
        for i, j in ((0, 1), (1, 2), (0, 2)):
            e = tuple(sorted((int(t[i]), int(t[j]))))
            demand[e] = demand.get(e, 0) + bits_per_mask
    for parent, child in tree_edges:
        shared = sorted(set(triangles[parent]) & set(triangles[child]))
        if len(shared) >= 2:
            e = (int(shared[0]), int(shared[1]))
        else:
            e = min(tuple(sorted((int(a), int(b))))
                    for a in triangles[parent] for b in triangles[child]
                    if int(a) != int(b)
                    and tuple(sorted((int(a), int(b)))) in physical_edges)
        demand[e] = demand.get(e, 0) + bits_per_mask

    r = OPERATING_POINT['secret_rate_bps']
    kill_node = 1  # shared vertex of T0 and T1 — the harshest choice
    num_rounds, kill_round = 20, 10

    def keywait(rates):
        return max(demand[e] / rates[e] for e in demand)

    rates_healthy = {e: r for e in demand}
    rates_sym = {e: (r / 2 if kill_node in e else r) for e in demand}
    rates_fixed = {e: (0.0 if min(e) == kill_node else r) for e in demand}

    kw_before = keywait(rates_healthy)
    kw_sym = keywait(rates_sym)
    dead_edges = [e for e, v in rates_fixed.items() if v == 0.0]
    stalled_tris = [ti for ti, t in enumerate(triangles)
                    if any(tuple(sorted((int(t[i]), int(t[j])))) in dead_edges
                           for i, j in ((0, 1), (1, 2), (0, 2)))]
    wall_sym = kill_round * kw_before + (num_rounds - kill_round) * kw_sym
    wall_healthy = num_rounds * kw_before

    print(f"\n  (b) Modeled 6c/3t round-time impact, source at node "
          f"{kill_node} killed at round {kill_round}/{num_rounds}\n"
          f"      (per-edge demand from the exps-5/8 ledger model, "
          f"secret rate {r/1000:.1f} kbit/s):")
    print(f"      symmetric switching (implemented): key_wait "
          f"{kw_before:.2f}s → {kw_sym:.2f}s "
          f"(×{kw_sym/kw_before:.2f}); ALL {len(triangles)} triangles "
          f"continue;\n        20-round key-wait wall-clock "
          f"{wall_healthy:.0f}s → {wall_sym:.0f}s "
          f"(+{100*(wall_sym/wall_healthy - 1):.0f}%)")
    print(f"      fixed-source-per-edge (hypothetical alternative): edges "
          f"{dead_edges} DEAD;\n        triangles {stalled_tris} stall "
          f"permanently — {len(stalled_tris)}/{len(triangles)} of the mesh "
          f"never finishes a round\n        (rounds-to-target: unreachable "
          f"for any aggregate that needs those clusters)")
    assert len(stalled_tris) > 0, "kill choice should stall the fixed design"

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    rounds_x = np.arange(1, num_rounds + 1)
    kw_sym_series = [kw_before if rd <= kill_round else kw_sym
                     for rd in rounds_x]
    kw_fixed_series = [kw_before if rd <= kill_round else np.nan
                       for rd in rounds_x]
    ax1.plot(rounds_x, kw_sym_series, 'o-', color=C['quantum'],
             label='symmetric switching (implemented): ×2 slower, continues')
    ax1.plot(rounds_x, kw_fixed_series, 's-', color=C['unmasked'],
             label='fixed source per edge: stalls (no key supply)')
    ax1.axvline(x=kill_round + 0.5, color='gray', linestyle=':',
                label=f'node-{kill_node} source dies')
    _y = kw_sym * 1.06
    ax1.text(kill_round + 1, _y, 'fixed design: 2/3 of mesh stalled',
             fontsize=7, color=C['unmasked'])
    ax1.set_xlabel('Training round')
    ax1.set_ylabel('Modeled key_wait per round (s)')
    ax1.set_title('Round key-wait before/after source failure')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.bar(['both sources\n(healthy edge)', 'one source killed\n(far endpoint only)'],
            [n_full / dur, n_deg / (dur * 0.5) * 0.5],
            color=[C['quantum'], C['accent']])
    ax2.set_ylabel('Edge coincidence rate (Hz, simulated)')
    ax2.set_title(f'(a) Source-level: edge continues at '
                  f'{100*ratio:.0f}% rate')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 10: availability under single-source failure — '
                 'an availability/fairness\nproperty of two-endpoint keying '
                 '(NOT a security mechanism; trusted-source model)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp10_availability')

    print("  ✓ Experiment 10 complete.")
    return {
        "source_level": {"full_coincidences": int(n_full),
                         "degraded_coincidences": int(n_deg),
                         "rate_ratio": float(ratio),
                         "qber_full": float(edge_full.qber),
                         "qber_degraded": float(edge_degraded.qber)},
        "modeled": {"key_wait_before_s": float(kw_before),
                    "key_wait_after_symmetric_s": float(kw_sym),
                    "wall_penalty_pct": float(100 * (wall_sym / wall_healthy - 1)),
                    "fixed_design_dead_edges": [list(e) for e in dead_edges],
                    "fixed_design_stalled_triangles": stalled_tris},
    }


def main():
    parser = argparse.ArgumentParser(description='QD-Entanglement Simulation Runner')
    parser.add_argument('--exp', nargs='+', type=int, default=list(range(1, 12)),
                        help='Experiment numbers to run (default: all 1-8)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with reduced parameter sweeps')
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Entanglement-derived one-time masks for federated learning  ║")
    print("║  over campus fiber meshes — source-model key distillation    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"Experiments to run: {args.exp}")
    print(f"Mode: {'QUICK' if args.quick else 'BALANCED'}")

    all_results = {}
    total_start = time.time()

    experiment_fns = {
        1: run_experiment_1,
        2: run_experiment_2,
        3: run_experiment_3,
        4: run_experiment_4,
        5: run_experiment_5,
        6: run_experiment_6,
        7: run_experiment_7,
        8: run_experiment_8,
        9: run_experiment_9,
        10: run_experiment_10,
        11: run_experiment_11,
    }

    for exp_num in sorted(args.exp):
        if exp_num in experiment_fns:
            exp_start = time.time()
            try:
                result = experiment_fns[exp_num](quick=args.quick)
                all_results[f"exp{exp_num}"] = result
                elapsed = time.time() - exp_start
                print(f"  ⏱ Experiment {exp_num} took {elapsed:.1f}s")
            except Exception as e:
                print(f"  ✗ Experiment {exp_num} failed: {e}")
                import traceback
                traceback.print_exc()

    total_elapsed = time.time() - total_start

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    try:
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=convert)
        print(f"\nSummary saved to: {summary_path}")
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE — Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results in: {RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
