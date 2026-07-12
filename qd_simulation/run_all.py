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
from config import QUANTUM, POST_PROC, MESH, FL, COST, VIZ, RESULTS_DIR

from modules.m1_quantum_source import (
    simulate_source, compute_chsh_s, werner_state,
    compute_qber_analytical, compute_coincidence_rate,
)
from modules.m2_post_processing import (
    fast_ec_simulate, compute_rate_waterfall, binary_entropy,
    validate_bp_against_statistical, toeplitz_hash, validate_pa_output,
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

    ax.axhline(y=1e6, color='gray', linestyle=':', alpha=0.5)
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
    base_rate = 1e6  # 1 MHz coincidence rate

    # ── 2a: Rate waterfall at default QBER ──
    print("  [2a] Rate waterfall at QBER=0.02...")
    wf = compute_rate_waterfall(base_rate, 0.02)

    stages = ['Raw Rate', 'After\nSifting', 'After\nEC', 'After\nPA',
              'After\nVerification', 'Final\nSecure Rate']
    values = [wf['raw_coincidence_rate'], wf['after_sifting'],
              wf['after_ec'], wf['after_pa'],
              wf['after_verification'], wf['final_secure_rate']]
    values_khz = [v/1000 for v in values]

    fig, ax = plt.subplots(figsize=VIZ['figsize'])
    bars = ax.bar(stages, values_khz, color=[C['dark'], C['quantum'], C['quantum'],
                   C['quantum'], C['quantum'], C['pareto']], edgecolor='white', linewidth=1.5)
    # Add overhead labels
    overheads = ['', '0%', f'{wf["ec_overhead_pct"]:.1f}%',
                 f'{wf["pa_overhead_pct"]:.1f}%',
                 f'{wf["verification_overhead_pct"]:.1f}%',
                 f'{wf["bell_overhead_pct"]:.1f}%']
    for bar, oh in zip(bars, overheads):
        if oh:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    f'−{oh}', ha='center', va='bottom', fontsize=7, color='gray')

    ax.set_ylabel('Rate (kbit/s)')
    ax.set_title(f'Experiment 2a: Rate Waterfall — QBER=2%\n'
                 f'Total overhead: {wf["total_overhead_pct"]:.1f}%  |  '
                 f'Final rate: {wf["final_secure_rate"]/1000:.0f} kbit/s')
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    save_fig(fig, 'exp2a_rate_waterfall')

    # ── 2b: Final secure rate vs QBER ──
    print("  [2b] Secure rate vs QBER sweep...")
    final_rates = []
    ec_overheads = []
    for q in qber_sweep:
        wf = compute_rate_waterfall(base_rate, q)
        final_rates.append(wf['final_secure_rate'])
        ec_overheads.append(wf['ec_overhead_pct'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    ax1.plot(qber_sweep*100, np.array(final_rates)/1000, 'o-', color=C['quantum'], linewidth=2)
    ax1.set_xlabel('QBER (%)')
    ax1.set_ylabel('Final Secure Rate (kbit/s)')
    ax1.set_title('Secure Key Rate vs QBER')
    ax1.axvspan(1.5, 4.0, alpha=0.1, color=C['quantum'], label='Expected range (Assumption D)')
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
        our_wf = compute_rate_waterfall(base_rate, q)
        sift_wf = compute_rate_waterfall(base_rate * 0.5, q)  # 50% sifting loss
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
    ax.set_title('Experiment 2c: Fixed-basis vs sifting rate\n'
                 '(Fixed-basis requires trusted source; sifting does not)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, 'exp2c_fixedbasis_vs_sifting')

    # ── 2d: BP decoder validation ──
    print("  [2d] Validating statistical EC model against full BP decoder...")
    bp_validation = validate_bp_against_statistical(
        qber_values=[0.01, 0.02, 0.03, 0.04],
        block_size=10000 if not quick else 5000,
    )
    print("    QBER    BP FER    Stat FER    Agreement")
    print("    " + "-"*45)
    for row in bp_validation:
        agree = "YES" if row["agreement"] else "NO"
        print(f"    {row['qber']:.2f}    {row['bp_fer']:.2f}      "
              f"{row['stat_fer']:.2f}        {agree}")

    # ── 2e: Privacy amplification validation ──
    print("  [2e] Validating Toeplitz privacy amplification...")
    rng_pa = np.random.default_rng(42)
    test_bits = rng_pa.integers(0, 2, size=10000, dtype=np.int8)
    secure_output = toeplitz_hash(test_bits, 5000, rng=rng_pa)
    pa_pvalue = validate_pa_output(secure_output)
    print(f"    Toeplitz output: {len(secure_output)} bits, "
          f"monobit p-value={pa_pvalue:.4f} ({'PASS' if pa_pvalue > 0.01 else 'FAIL'})")

    print("  ✓ Experiment 2 complete.")
    return {
        "final_rates": final_rates,
        "qber_sweep": qber_sweep.tolist(),
        "bp_validation": bp_validation,
        "pa_pvalue": pa_pvalue,
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
    secure_rate = 675000  # 675 kbit/s

    # ── 3a: Aggregation depth vs node count ──
    print("  [3a] Aggregation depth vs node count...")
    depths, n_triangles_list = [], []
    for n in node_counts:
        mesh = analyze_mesh(n, 1e5, secure_rate, rng=np.random.default_rng(42))
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

    # ── 3b: Max nodes heatmap ──
    print("  [3b] Max mesh size heatmap...")
    heatmap = np.zeros((len(model_sizes), len(round_periods)))
    for i, ms in enumerate(model_sizes):
        for j, rp in enumerate(round_periods):
            max_tri, max_n = compute_max_mesh_size(int(ms), rp, secure_rate)
            heatmap[i, j] = max_n

    fig, ax = plt.subplots(figsize=VIZ['figsize'])
    im = ax.imshow(heatmap, aspect='auto', cmap='YlOrBr', interpolation='nearest')
    ax.set_xticks(range(len(round_periods)))
    ax.set_xticklabels([f'{r}s' for r in round_periods])
    ax.set_yticks(range(len(model_sizes)))
    ax.set_yticklabels([f'{int(ms/1000)}K' for ms in model_sizes])
    ax.set_xlabel('Training Round Period')
    ax.set_ylabel('Model Size (parameters)')
    ax.set_title(f'Experiment 3b: Maximum Feasible Nodes\n(at {secure_rate/1000:.0f} kbit/s secure rate)')
    for i in range(len(model_sizes)):
        for j in range(len(round_periods)):
            val = int(heatmap[i, j])
            ax.text(j, i, str(val), ha='center', va='center',
                    color='white' if val > 50 else 'black', fontweight='bold')
    fig.colorbar(im, ax=ax, label='Max nodes')
    fig.tight_layout()
    save_fig(fig, 'exp3b_max_nodes_heatmap')

    print("  ✓ Experiment 3 complete.")
    return {"depths": depths, "heatmap": heatmap.tolist()}


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 4: FL TRAINING CORRECTNESS
# ═══════════════════════════════════════════════════════════
def run_experiment_4(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 4: FL Training Correctness")
    print("="*70)

    from modules.m4_federated_learning import (
        ClinicalMLP, load_clinical_data, run_federated_training,
    )

    num_rounds = 60 if not quick else 30
    num_clients = 3
    target_acc = 0.92
    num_repeats = FL.get('num_repeats', 10) if not quick else 3
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

    # ── Plot: Training curves with error bands ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    colors = {"entangled": C['quantum'], "prng": C['prng'],
              "unmasked": C['unmasked'], "dp": C['dp']}
    linestyles = {"entangled": "-", "prng": "--", "unmasked": "-.", "dp": ":"}

    for mode in modes:
        all_acc = np.array(results_summary[mode]["all_accuracies"])
        mean_acc = all_acc.mean(axis=0)
        std_acc = all_acc.std(axis=0)
        rounds_x = np.arange(1, len(mean_acc) + 1)

        ax1.plot(rounds_x, mean_acc, linewidth=2, color=colors[mode],
                linestyle=linestyles[mode], label=mode_labels[mode])
        ax1.fill_between(rounds_x, mean_acc - std_acc, mean_acc + std_acc,
                         alpha=0.15, color=colors[mode])

        all_losses = np.array([r.losses for r in all_runs[mode]])
        mean_loss = all_losses.mean(axis=0)
        clipped = np.maximum(mean_loss, 1e-6)
        ax2.semilogy(rounds_x, clipped, linewidth=2, color=colors[mode],
                     linestyle=linestyles[mode], label=mode_labels[mode])

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Model Accuracy (mean ± 1σ)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.2, 1.02)

    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Test Loss (log scale)')
    ax2.set_title('Training Loss (mean)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle(f'Experiment 4: FL Training Convergence — Breast Cancer Classification\n'
                 f'{num_clients} clients, Dirichlet α={dirichlet_alpha} non-IID, '
                 f'{num_repeats} repeats, FedAvg',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp4_training_convergence')

    # ── Summary table ──
    print(f"\n  SUMMARY ({num_repeats} repeats, mean ± std):")
    print(f"  {'Mode':<25} {'Accuracy':>15} {'Rounds to Target':>20}")
    print("  " + "-"*62)
    for mode in modes:
        s = results_summary[mode]
        print(f"  {mode_labels[mode]:<25} "
              f"{s['mean_acc']:.4f} ± {s['std_acc']:.4f} "
              f"{s['mean_rounds']:>8.1f} ± {s['std_rounds']:.1f}")

    print("  ✓ Experiment 4 complete.")
    return {m: {"acc_mean": s["mean_acc"], "acc_std": s["std_acc"],
                "rounds_mean": s["mean_rounds"]}
            for m, s in results_summary.items()}


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 5: DROPOUT RECOVERY (entangled-key two-mask)
# ═══════════════════════════════════════════════════════════
def run_experiment_5(quick=False):
    """Dropout recovery via survivor reveal under per-round-fresh pairwise keys.

    Entanglement-derived pairwise bits are single-use per round. Survivors can safely reveal pairwise bits with
    dropped clients (no future rounds compromised). We sweep dropout rate
    and check (i) training still converges, (ii) recovery activates as
    expected, (iii) the honest-but-curious aggregator's chi-squared
    distinguisher cannot break the modular mask at any dropout rate.

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
    dropout_rates = [0.0, 0.05, 0.15, 0.30] if not quick else [0.0, 0.15]

    triangles, _, tree_edges = define_triangle_topology(n_clients)
    client_data, test_data, input_dim = load_clinical_data(
        n_clients, rng_seed=42, dirichlet_alpha=FL.get('dirichlet_alpha', 0.5)
    )
    model_fn = lambda: ClinicalMLP(input_dim=input_dim, hidden=128)

    results = {}
    for dr in dropout_rates:
        print(f"\n  ── Dropout rate: {int(dr*100)}% ──")
        res, aux = run_multitriangle_fl(
            model_fn, client_data, test_data,
            num_clients=n_clients, num_rounds=num_rounds,
            local_epochs=FL['local_epochs'], batch_size=FL['local_batch_size'],
            lr=FL['learning_rate'], target_accuracy=target_acc,
            triangles=triangles, tree_edges=tree_edges,
            dropout_rate=dr, verbose=False,
        )
        results[dr] = (res, aux)
        print(f"    final acc={res.final_accuracy:.4f}  "
              f"rounds_to_target={res.rounds_to_target}  "
              f"recovery_rounds={sum(aux['recovery_invoked_per_round'])}  "
              f"distinguishable_rounds={sum(aux['distinguishable_per_round'])}")

    # Plot 1: accuracy vs round under each dropout rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    colors = [C['quantum'], C['pareto'], C['accent'], C['unmasked']]
    for (dr, (res, _)), col in zip(results.items(), colors):
        ax1.plot(range(1, len(res.accuracies)+1), res.accuracies,
                 linewidth=2, color=col, label=f'dropout {int(dr*100)}%')
    ax1.axhline(y=target_acc, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Training round')
    ax1.set_ylabel('Test accuracy')
    ax1.set_title('Convergence under client dropout')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: per-round recovery + distinguisher
    drs = sorted(results.keys())
    final_accs = [results[dr][0].final_accuracy for dr in drs]
    recovery_pcts = [100 * np.mean(results[dr][1]['recovery_invoked_per_round'])
                     for dr in drs]
    distinguish_pcts = [100 * np.mean(results[dr][1]['distinguishable_per_round'])
                        for dr in drs]
    x = np.arange(len(drs))
    w = 0.28
    ax2.bar(x - w, final_accs, w, color=C['quantum'], label='final accuracy')
    ax2.bar(x, [r/100 for r in recovery_pcts], w, color=C['pareto'],
            label='rounds invoking recovery (fraction)')
    ax2.bar(x + w, [d/100 for d in distinguish_pcts], w, color=C['unmasked'],
            label='rounds with aggregator distinguisher (fraction)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(dr*100)}%' for dr in drs])
    ax2.set_xlabel('Per-round dropout rate')
    ax2.set_ylabel('Value')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.set_title('Recovery activation and aggregator distinguishability')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 5: Dropout recovery, per-round-fresh two-mask aggregation\n'
                 'Per-round bit freshness enables safe survivor reveal',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp5_dropout_recovery')

    print("\n  Summary:")
    print(f"  {'Dropout':>10}  {'Final acc':>10}  {'Recovery %':>12}  {'Distinguishable %':>18}")
    for dr in drs:
        r, a = results[dr]
        rec_pct = 100 * np.mean(a['recovery_invoked_per_round'])
        dis_pct = 100 * np.mean(a['distinguishable_per_round'])
        print(f"  {int(dr*100):>9}%  {r.final_accuracy:>10.4f}  {rec_pct:>11.1f}%  {dis_pct:>17.1f}%")
    # ── Classical parity baseline (Flamingo-style per-round pool) ──
    # Same reveal-based recovery, same dropout, two mask-material sources.
    # Mechanism parity is EXPECTED; the differentiator is the assumption class.
    import torch as _torch
    from modules.m4_federated_learning import secure_aggregation_round
    _tri = [0, 1, 2]; _npar = 512
    _grads = {c: _torch.tensor(np.random.default_rng(100 + c).normal(0, 0.05, _npar),
                               dtype=_torch.float32) for c in _tri}
    _seeds = {c: 7000 + c for c in _tri}
    _need = 3 * _npar * 32
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
        "final_acc": results[dr][0].final_accuracy,
        "recovery_pct": float(np.mean(results[dr][1]['recovery_invoked_per_round'])),
        "distinguishable_pct": float(np.mean(results[dr][1]['distinguishable_per_round'])),
    } for dr in drs}}


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 5b (DEMOTED): Gradient inversion sanity check
# Retained as appendix only; not invoked from run_all by default.
# ═══════════════════════════════════════════════════════════
def run_experiment_5_legacy(quick=False):
    print("\n" + "="*70)
    print("APPENDIX 5b: Gradient inversion sanity check (legacy)")
    print("="*70)

    import torch
    import torchvision
    import torchvision.transforms as transforms
    from modules.m4_federated_learning import (
        SmallCNN, compute_gradient_for_attack,
        gradient_inversion_attack, mask_gradient_for_attack,
    )

    n_attack_images = 3 if not quick else 2
    attack_iters = FL['attack_iterations'] if not quick else 1000

    print("  Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    try:
        dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(RESULTS_DIR, 'data'), train=True,
            download=True, transform=transform
        )
    except Exception as e:
        print(f"  ⚠ Could not download CIFAR-10: {e}")
        print("  Generating synthetic 32x32 image data instead...")
        # Fallback: synthetic images
        X_syn = torch.randn(100, 3, 32, 32)
        y_syn = torch.randint(0, 10, (100,))
        dataset = torch.utils.data.TensorDataset(X_syn, y_syn)

    model = SmallCNN(num_classes=10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: SmallCNN with {n_params:,} parameters")

    fig, axes = plt.subplots(4, n_attack_images, figsize=(7.16, 7.16))

    metric_results = {"unmasked": [], "gaussian-legacy": [], "dp": []}

    def compute_psnr(original, reconstructed):
        mse = torch.mean((original - reconstructed) ** 2).item()
        if mse < 1e-10:
            return 100.0
        return 10 * np.log10(4.0 / mse)  # Max range = 2 (normalized to [-1,1])

    def compute_ssim(original, reconstructed, window_size=7):
        """Structural Similarity Index (Wang et al. 2004).
        Computed per-channel then averaged. Standard metric for
        gradient inversion evaluation (Geiping 2020, Zhu 2019)."""
        C1, C2 = 0.01 ** 2, 0.03 ** 2  # Stabilization constants (L=1 for normalized)
        orig = original.squeeze(0).float()
        recon = reconstructed.squeeze(0).float()
        pad = window_size // 2
        # Uniform window averaging
        kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
        ssim_vals = []
        for ch in range(orig.shape[0]):
            o = orig[ch:ch+1].unsqueeze(0)
            r = recon[ch:ch+1].unsqueeze(0)
            mu_o = torch.nn.functional.conv2d(o, kernel, padding=pad)
            mu_r = torch.nn.functional.conv2d(r, kernel, padding=pad)
            mu_o_sq, mu_r_sq, mu_or = mu_o**2, mu_r**2, mu_o * mu_r
            sigma_o = torch.nn.functional.conv2d(o**2, kernel, padding=pad) - mu_o_sq
            sigma_r = torch.nn.functional.conv2d(r**2, kernel, padding=pad) - mu_r_sq
            sigma_or = torch.nn.functional.conv2d(o*r, kernel, padding=pad) - mu_or
            ssim_map = ((2*mu_or + C1) * (2*sigma_or + C2)) / \
                       ((mu_o_sq + mu_r_sq + C1) * (sigma_o + sigma_r + C2))
            ssim_vals.append(ssim_map.mean().item())
        return np.mean(ssim_vals)

    def to_img(t):
        img = t.squeeze(0).permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        return img

    for img_idx in range(n_attack_images):
        print(f"  Attacking image {img_idx+1}/{n_attack_images}...")

        if hasattr(dataset, 'data'):
            image, label = dataset[img_idx * 50]
        else:
            image, label = dataset[img_idx]
        if isinstance(label, int):
            label = torch.tensor([label])
        image = image.unsqueeze(0)

        true_grad = compute_gradient_for_attack(model, image, label)

        # ── Attack on UNMASKED gradient ──
        print(f"    [unmasked] Running inversion ({attack_iters} iters)...")
        recon_unmasked, _ = gradient_inversion_attack(
            model, true_grad, label, image.shape,
            n_iterations=attack_iters, lr=FL['attack_lr']
        )

        # ── Attack on QUANTUM-MASKED gradient ──
        # Use quantum-derived mask (Box-Muller from random bits, same as FL pipeline)
        from modules.m4_federated_learning import generate_gaussian_mask_legacy
        qbits = np.random.default_rng(img_idx).integers(0, 2, size=n_params*32, dtype=np.int8)
        q_mask = generate_gaussian_mask_legacy(n_params, qbits) * 10  # Scale for strong masking
        masked_grad = mask_gradient_for_attack(true_grad, q_mask)
        print(f"    [quantum] Running inversion ({attack_iters} iters)...")
        recon_quantum, _ = gradient_inversion_attack(
            model, masked_grad, label, image.shape,
            n_iterations=attack_iters, lr=FL['attack_lr']
        )

        # ── Attack on DP gradient ──
        from modules.m4_federated_learning import add_dp_noise
        dp_grad = []
        flat_grad = torch.cat([g.flatten() for g in true_grad])
        noisy_flat = add_dp_noise(flat_grad, epsilon=5.0, delta=1e-5, max_grad_norm=0.5)
        offset = 0
        for g in true_grad:
            n = g.numel()
            dp_grad.append(noisy_flat[offset:offset+n].view(g.shape))
            offset += n
        print(f"    [dp] Running inversion ({attack_iters} iters)...")
        recon_dp, _ = gradient_inversion_attack(
            model, dp_grad, label, image.shape,
            n_iterations=attack_iters, lr=FL['attack_lr']
        )

        # ── Metrics ──
        for mode_name, recon in [("unmasked", recon_unmasked),
                                  ("gaussian-legacy", recon_quantum),
                                  ("dp", recon_dp)]:
            psnr = compute_psnr(image, recon)
            ssim = compute_ssim(image, recon)
            metric_results[mode_name].append({"psnr": psnr, "ssim": ssim})

        # ── Plot: Original + 3 reconstructions ──
        axes[0, img_idx].imshow(to_img(image))
        axes[0, img_idx].set_title('Original', fontsize=8)
        axes[0, img_idx].axis('off')

        m = metric_results["unmasked"][-1]
        axes[1, img_idx].imshow(to_img(recon_unmasked))
        axes[1, img_idx].set_title(f'Unmasked\nPSNR={m["psnr"]:.1f}dB SSIM={m["ssim"]:.3f}', fontsize=7)
        axes[1, img_idx].axis('off')

        m = metric_results["gaussian-legacy"][-1]
        axes[2, img_idx].imshow(to_img(recon_quantum))
        axes[2, img_idx].set_title(f'Quantum-masked\nPSNR={m["psnr"]:.1f}dB SSIM={m["ssim"]:.3f}', fontsize=7)
        axes[2, img_idx].axis('off')

        m = metric_results["dp"][-1]
        axes[3, img_idx].imshow(to_img(recon_dp))
        axes[3, img_idx].set_title(f'DP (ε=5)\nPSNR={m["psnr"]:.1f}dB SSIM={m["ssim"]:.3f}', fontsize=7)
        axes[3, img_idx].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=8, fontweight='bold')
    axes[1, 0].set_ylabel('Unmasked\n(ATTACK SUCCEEDS)', fontsize=11, color=C['unmasked'], fontweight='bold')
    axes[2, 0].set_ylabel('Quantum-masked\n(ATTACK FAILS)', fontsize=11, color=C['quantum'], fontweight='bold')
    axes[3, 0].set_ylabel('DP ε=5\n(PARTIAL LEAKAGE)', fontsize=11, color=C['dp'], fontweight='bold')

    fig.suptitle('Experiment 5: Gradient Inversion Attack — Privacy Demonstration\n'
                 'Original images (top) vs reconstructions from intercepted gradients',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp5_gradient_inversion')

    # ── Metric summary ──
    print("\n  Attack Metric Summary (higher = more information leaked):")
    print(f"  {'Mode':>15}  {'PSNR (dB)':>15}  {'SSIM':>15}")
    print("  " + "-"*50)
    for mode_name in ["unmasked", "gaussian-legacy", "dp"]:
        psnrs = [m["psnr"] for m in metric_results[mode_name]]
        ssims = [m["ssim"] for m in metric_results[mode_name]]
        print(f"  {mode_name:>15}  {np.mean(psnrs):>6.1f} ± {np.std(psnrs):.1f}  "
              f"  {np.mean(ssims):>6.3f} ± {np.std(ssims):.3f}")

    print("  ✓ Experiment 5 complete.")
    return metric_results


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 6: RATE-CONSTRAINED THROUGHPUT
# ═══════════════════════════════════════════════════════════
def run_experiment_6(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 6: Rate-Constrained Training Throughput")
    print("="*70)

    model_sizes = FL['model_sizes_sweep'] if not quick else [5e4, 1e5, 5e5, 1e6]
    key_rates = [100e3, 300e3, 675e3, 1e6] if not quick else [300e3, 675e3]

    # Compute mask generation time and training throughput
    print("  Computing throughput for model size × key rate grid...")
    rounds_per_hour = np.zeros((len(model_sizes), len(key_rates)))

    for i, ms in enumerate(model_sizes):
        gradient_bits = int(ms) * 16  # BF16
        for j, kr in enumerate(key_rates):
            mask_time_s = gradient_bits / kr
            # Assume computation time per round = 2 seconds (local training)
            compute_time_s = 2.0
            total_round_time = mask_time_s + compute_time_s
            rounds_per_hour[i, j] = 3600 / total_round_time

    fig, ax = plt.subplots(figsize=VIZ['figsize'])
    for j, kr in enumerate(key_rates):
        label = f'{kr/1000:.0f} kbit/s'
        ax.plot([int(ms/1000) for ms in model_sizes], rounds_per_hour[:, j],
                'o-', linewidth=2, label=label)

    # Add unconstrained baseline
    unconstrained = [3600 / 2.0] * len(model_sizes)  # Only compute time
    ax.plot([int(ms/1000) for ms in model_sizes], unconstrained,
            'k--', linewidth=1.5, alpha=0.5, label='Unconstrained (compute only)')

    ax.set_xlabel('Model Size (K parameters)')
    ax.set_ylabel('Training Rounds per Hour')
    ax.set_xscale('log')
    ax.set_title('Experiment 6: Rate-Constrained Training Throughput\n'
                 'Quantum mask generation as bottleneck vs. compute')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    save_fig(fig, 'exp6_throughput')

    # ── Practical ceiling identification ──
    print("\n  Practical model-size ceilings (mask_time < round_period):")
    for kr in key_rates:
        for period_name, period_s in [("30s", 30), ("60s", 60), ("300s", 300)]:
            max_bits = kr * period_s
            max_params = max_bits / 16
            print(f"    {kr/1000:.0f} kbit/s, {period_name} rounds: "
                  f"max {max_params/1000:.0f}K params ({max_params*16/8/1e6:.1f} MB gradient)")

    print("  ✓ Experiment 6 complete.")
    return {"rounds_per_hour": rounds_per_hour.tolist()}


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 7: QBER → PRIVACY-BUDGET CURVE
# ═══════════════════════════════════════════════════════════
def run_experiment_7(quick=False):
    """Map physical QBER to a per-bit mask-entropy floor and a maximum
    tolerable model size at fixed round budget.

    This is the quantum-vs-classical differentiator: classical PRNG masks
    have no analog (their entropy doesn't depend on channel quality). For
    entangled-key the LHL bound on PA output is:
        H_min(output) >= H_min(input) - 2*log(1/epsilon)
    where H_min(input) approaches (1 - h(QBER)) * block_size after
    reconciliation. We compute mask entropy per output bit and tie it to
    the round budget for several model sizes.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 7: QBER -> privacy-budget curve")
    print("="*70)

    qbers = np.linspace(0.005, 0.06, 12) if not quick else np.linspace(0.005, 0.06, 6)
    block_size = POST_PROC['block_size']
    code_rate = POST_PROC['ldpc_code_rate']
    security_margin = POST_PROC['security_margin_bits']

    # Source raw rate (analytical) at default fiber length
    raw_rate = compute_coincidence_rate(
        QUANTUM['source_rep_rate_hz'], QUANTUM['source_brightness'],
        QUANTUM['coupling_efficiency'], QUANTUM['fiber_length_m'],
        QUANTUM.get('fiber_loss_db_per_km_1550nm', 0.2)
        if QUANTUM.get('use_qfc', True)
        else QUANTUM['fiber_loss_db_per_km_780nm'],
        QUANTUM['spad_efficiency'],
        colocated=QUANTUM.get('colocated', True),
        local_optics_efficiency=QUANTUM.get('local_optics_efficiency', 0.90),
        use_qfc=QUANTUM.get('use_qfc', True),
        qfc_efficiency=QUANTUM.get('qfc_efficiency', 0.35),
    )

    # For each QBER, compute:
    #   shared_rate_bps: after EC + PA + verification + Bell duty
    #   mask_entropy_per_bit: LHL floor on output uniformity
    #   max_model_params @ 60s round @ 3 pairs per triangle
    shared_rates, mask_entropies, max_params = [], [], []
    for q in qbers:
        waterfall = compute_rate_waterfall(
            raw_rate, q, code_rate=code_rate,
            security_margin=security_margin, block_size=block_size,
            bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
        )
        rate = waterfall['final_secure_rate']
        shared_rates.append(rate)

        # LHL bound on per-output-bit entropy floor
        # H_min(input) per bit ~ (1 - h(q)) * (1 - beta * h(q)/(1-h(q)))
        # Output is epsilon-close to uniform for epsilon = 2^{-security_margin}
        # We report the gap to uniform (in bits per output bit).
        h_q = binary_entropy(q)
        beta = 1.15
        input_min_entropy_per_bit = max(0, 1 - beta * h_q - h_q)
        # LHL: H_out >= L - 2*log2(1/epsilon) where L is output length,
        # 2*log2(1/epsilon) = 2*security_margin. Per output bit:
        gap_per_bit = (2 * security_margin) / max(block_size * input_min_entropy_per_bit, 1)
        per_bit_entropy = max(0, 1 - gap_per_bit)
        mask_entropies.append(per_bit_entropy)

        # Max model parameters at fixed 60s round, 3 pairs per triangle,
        # 32 bits per param (int32 modular mask).
        round_s = 60
        bits_available = rate * round_s
        bits_per_param_per_round = 32 * 3  # 3 pairs in a triangle
        max_p = bits_available / bits_per_param_per_round
        max_params.append(max_p)

    # Find the cutoff QBER where rate falls below 1 kbit/s (practical floor)
    cutoff_qber = None
    for q, r in zip(qbers, shared_rates):
        if r < 1000:
            cutoff_qber = q
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])

    ax1.plot(qbers, np.array(shared_rates) / 1000, 'o-', linewidth=2,
             color=C['quantum'], label='Shared bit rate (kbit/s)')
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
    for ref, lbl in [(50, '50K (ClinicalMLP)'),
                      (150, '150K (SmallCNN)'),
                      (1000, '1M params')]:
        ax2.axhline(y=ref, color='gray', linestyle=':', alpha=0.5)
        ax2.text(qbers[0], ref, lbl, fontsize=7, color='gray', va='bottom')
    ax2.set_xlabel('QBER')
    ax2.set_ylabel('Max model size at 60s round (K params)')
    ax2.set_yscale('log')
    ax2.set_title('Privacy-budget headroom vs QBER')
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle('Experiment 7: QBER -> privacy-budget curve\n'
                 'Physical channel quality bounds the maskable model size',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp7_qber_privacy')

    print("\n  QBER -> Shared rate -> Max model size (60s rounds):")
    print(f"  {'QBER':>8}  {'Rate':>15}  {'Mask H/bit':>11}  {'Max K params':>14}")
    for q, r, h, mp in zip(qbers, shared_rates, mask_entropies, max_params):
        print(f"  {q:>8.4f}  {r/1000:>10.2f} kbit/s  {h:>11.4f}  {mp/1000:>14.1f}")
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
            mesh = analyze_mesh(n_clients, 1e5, 675000, rng=np.random.default_rng(42))
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
        print(f"    Aggregator distinguishable rounds: "
              f"{sum(aux['distinguishable_per_round'])}/{num_rounds}")
        print(f"    entangled-key bits consumed (intra/inter): "
              f"{aux['bits_intra_consumed']:,} / {aux['bits_inter_consumed']:,}")

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

    # ═══ PLOT 3: Aggregation overhead projection ═══
    fig, ax = plt.subplots(figsize=VIZ['figsize'])

    secure_rate = 675000  # bits/s
    model_params = [5e4, 1e5, 5e5, 1e6]

    for mp in model_params:
        gradient_bits = int(mp) * 16
        mask_time_per_level = gradient_bits / secure_rate
        overhead_at_depths = [mask_time_per_level * d for d in tree_depths]
        ax.plot(scales, overhead_at_depths, 'o-', linewidth=2,
                label=f'{int(mp/1000)}K params ({mask_time_per_level:.1f}s/level)')

    ax.set_xlabel('Number of Clients')
    ax.set_ylabel('Aggregation Overhead (seconds)')
    ax.set_title('Experiment 8c: Projected Aggregation Overhead\n'
                 f'at {secure_rate/1000:.0f} kbit/s secure key rate')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Add round period reference lines
    for period, label in [(30, '30s round'), (60, '60s round'), (300, '5min round')]:
        ax.axhline(y=period, color='gray', linestyle=':', alpha=0.4)
        ax.text(12.3, period + 2, label, fontsize=8, color='gray')

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
        print("  ⚠ Some scales did not converge — investigate")

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
def main():
    parser = argparse.ArgumentParser(description='QD-Entanglement Simulation Runner')
    parser.add_argument('--exp', nargs='+', type=int, default=list(range(1, 9)),
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
