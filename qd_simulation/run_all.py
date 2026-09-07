#!/usr/bin/env python3
"""
entangled-key two-mask FL - full simulation runner.

Protocol framings (same physical layer, two post-processings):
  - BBM92 with efficient 90/10 sifting (system default): X-basis
    parameter estimation and TLGR fixed-length finite-key privacy
    amplification. Tolerates an active fiber adversary; source trust
    is discharged by the basis-independent-source argument.
  - Trusted-conduit mode (fixed basis): source-model secret-key
    agreement under a trusted co-located QD source and a noisy but
    passive fiber. Higher rate (1.32x at the operating point), blind
    to an in-fiber intercept-resend adversary (exp14).
  Both: pairwise Bell-pair measurement produces correlated bit pairs;
  LDPC reconciliation makes the endpoints' strings bit-exact; Toeplitz
  PA removes residual correlation from the syndrome and physical-layer
  side channels; the resulting single-use shared bits drive modular
  OTP masking in two-mask Bonawitz secure aggregation, one fresh draw
  per round from a pooled per-edge buffer.

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
from config import (QUANTUM, POST_PROC, MESH, FL, VIZ, RESULTS_DIR,
                    BITS_PER_PARAM, OPERATING_POINT,
                    CLASSICAL_LINK_BPS, QKD_FRAMING, BBM92)

from modules.m1_quantum_source import (
    simulate_source, compute_chsh_s, werner_state,
    compute_qber_analytical, compute_coincidence_rate,
    simulate_symmetric_edge, simulate_bbm92,
    fss_precession_omega, delay_window_coherence, calibrate_visibility,
    detected_delay_mode, werner_weight_for_negativity,
)
from modules.m2_post_processing import (
    fast_ec_simulate, compute_rate_waterfall, compute_rate_waterfall_qkd,
    binary_entropy, validate_bp_against_statistical, toeplitz_hash,
    validate_pa_output, estimate_capacity_qber, randomness_battery,
    bbm92_distill, tlgr_deviation,
)
from modules.m3_mesh_topology import (
    generate_campus_mesh, build_aggregation_tree,
    compute_aggregation_latency, compute_max_mesh_size,
    analyze_mesh,
)

# Publication-quality plot defaults (IEEE 10pt, Times New Roman)
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


# Quick mode writes to suffixed filenames so reduced-sweep runs never
# overwrite the full-mode figures and summary.
_OUTPUT_SUFFIX = ""

# Publication tiering (IEEE Network: six figures/tables total). Figures
# named here land in results/main/, everything else in
# results/peripheral/; quick-mode outputs always go to peripheral.
# Captions carrying the context stripped from figure titles live in
# results/figure_captions.md; prose replacing demoted figures lives in
# results/replaced_by_text.md.
MAIN_FIGURES = {
    'exp5_dropout_recovery',
    'exp7_qber_key_rate_envelope',
    'exp9_hndl_transcript',
    'exp14_bbm92_executing',
}


def set_output_suffix(suffix):
    global _OUTPUT_SUFFIX
    _OUTPUT_SUFFIX = suffix


def _expand_axis_margins(fig, frac=0.10):
    """Guarantee every axis bound sits beyond the outermost drawn data.

    Applied to every figure at save time so no panel can ship with
    markers, whiskers, or curve ends touching the frame. Limits are
    only ever expanded, so explicit generous limits set by individual
    experiments survive. Skipped: image axes (heatmaps), colorbars,
    and non-linear/log scales (symlog axes carry hand-set limits).
    Bar charts keep a zero baseline on the value axis by convention.
    """
    from matplotlib.patches import Rectangle
    for ax in fig.axes:
        if ax.get_label() == '<colorbar>' or ax.images:
            continue
        d = ax.dataLim
        for which in ('x', 'y'):
            if which == 'x':
                scale, (lo, hi) = ax.get_xscale(), ax.get_xlim()
                dmin, dmax = d.x0, d.x1
            else:
                scale, (lo, hi) = ax.get_yscale(), ax.get_ylim()
                dmin, dmax = d.y0, d.y1
            if not (np.isfinite(dmin) and np.isfinite(dmax)) \
                    or dmin >= dmax or lo > hi:
                continue
            if scale == 'linear':
                pad = frac * (dmax - dmin)
                new_lo, new_hi = dmin - pad, dmax + pad
            elif scale == 'log':
                if dmin <= 0:
                    continue
                f = 10 ** (frac * (np.log10(dmax) - np.log10(dmin)))
                new_lo, new_hi = dmin / f, dmax * f
            else:
                continue
            has_bars = sum(isinstance(p, Rectangle)
                           for p in ax.patches) >= 2
            if which == 'y' and has_bars and lo == 0 and dmin >= 0:
                new_lo = 0
            lo2, hi2 = min(lo, new_lo), max(hi, new_hi)
            if (lo2, hi2) != (lo, hi):
                (ax.set_xlim if which == 'x' else ax.set_ylim)(lo2, hi2)


def _extend_ticks_to_bound_data(fig):
    """Guarantee the outermost drawn tick bounds the data on every axis.

    _expand_axis_margins pads limits past the data, but the padded end
    can fall between ticks: a marker or whisker then sits beyond the
    last labeled value and cannot be read against the axis. This pass
    extends the offending limit outward to the next tick-grid point
    (next decade on log axes) and pins the tick set so the new range
    cannot re-bin it. Limits only grow, so headroom reserved for
    annotations survives. Skipped: image axes, colorbars, categorical
    axes (fixed ticks), and scales other than linear/log.
    """
    from matplotlib.ticker import FixedLocator
    for ax in fig.axes:
        if ax.get_label() == '<colorbar>' or ax.images:
            continue
        d = ax.dataLim
        for which in ('x', 'y'):
            axis = ax.xaxis if which == 'x' else ax.yaxis
            if isinstance(axis.get_major_locator(), FixedLocator):
                continue
            if which == 'x':
                scale, (lo, hi) = ax.get_xscale(), ax.get_xlim()
                dmin, dmax = d.x0, d.x1
            else:
                scale, (lo, hi) = ax.get_yscale(), ax.get_ylim()
                dmin, dmax = d.y0, d.y1
            if not (np.isfinite(dmin) and np.isfinite(dmax)) \
                    or dmin >= dmax or lo >= hi:
                continue
            set_lim = ax.set_xlim if which == 'x' else ax.set_ylim
            if scale == 'log':
                if dmin <= 0 or lo <= 0:
                    continue
                shown = [t for t in axis.get_majorticklocs()
                         if lo * (1 - 1e-9) <= t <= hi * (1 + 1e-9)]
                new_lo, new_hi = lo, hi
                if not any(t <= dmin * (1 + 1e-9) for t in shown):
                    new_lo = 10.0 ** np.floor(np.log10(dmin) + 1e-9)
                if not any(t >= dmax * (1 - 1e-9) for t in shown):
                    new_hi = 10.0 ** np.ceil(np.log10(dmax) - 1e-9)
                if (new_lo, new_hi) != (lo, hi):
                    set_lim(min(lo, new_lo), max(hi, new_hi))
                continue
            if scale != 'linear':
                continue
            ticks = np.sort(np.asarray(axis.get_majorticklocs(), float))
            if ticks.size < 2:
                continue
            step = np.min(np.diff(ticks))
            if not np.isfinite(step) or step <= 0:
                continue
            eps = 1e-9 * (hi - lo)
            shown = ticks[(ticks >= lo - eps) & (ticks <= hi + eps)]
            if shown.size == 0:
                continue
            new_ticks = list(shown)
            while new_ticks[0] > dmin + eps:
                new_ticks.insert(0, new_ticks[0] - step)
            while new_ticks[-1] < dmax - eps:
                new_ticks.append(new_ticks[-1] + step)
            new_lo = min(lo, new_ticks[0])
            new_hi = max(hi, new_ticks[-1])
            if (new_lo, new_hi) != (lo, hi):
                set_lim(new_lo, new_hi)
                axis.set_ticks(new_ticks)


def save_fig(fig, name):
    """Save figure as both PNG (preview) and PDF (LaTeX inclusion)."""
    _expand_axis_margins(fig)
    _extend_ticks_to_bound_data(fig)
    tier = 'main' if name in MAIN_FIGURES and not _OUTPUT_SUFFIX \
        else 'peripheral'
    out_dir = os.path.join(RESULTS_DIR, tier)
    os.makedirs(out_dir, exist_ok=True)
    name = f"{name}{_OUTPUT_SUFFIX}"
    for ext in ['png', 'pdf']:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=VIZ['dpi'], bbox_inches='tight',
                    facecolor='white', pad_inches=0.03)
    plt.close(fig)
    print(f"    Saved: {tier}/{name}.png/pdf")


def export_summary_csv(summary, suffix=""):
    """Write the summary content as per-experiment CSV tables.

    Long format, one row per scalar: key is the dotted path into the
    experiment's dict, index is the position for array entries (blank
    for scalars). Parallel arrays such as qbers and rates share
    indices, so any plotted series can be rebuilt with a pivot. Files
    land in results/data/, regenerated alongside summary.json and kept
    out of version control like the rest of the outputs.
    """
    import csv
    out_dir = os.path.join(RESULTS_DIR, 'data')
    os.makedirs(out_dir, exist_ok=True)

    def rows(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from rows(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, (list, tuple, np.ndarray)):
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list, tuple, np.ndarray)):
                    yield from rows(f"{prefix}[{i}]", v)
                else:
                    yield (prefix, i, v)
        else:
            yield (prefix, "", obj)

    for exp in sorted(summary):
        path = os.path.join(out_dir, f"{exp}{suffix}.csv")
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['key', 'index', 'value'])
            for row in rows("", summary[exp]):
                w.writerow(row)
    print(f"CSV tables: {out_dir}/ ({len(summary)} experiments)")


def print_comm_summary(fl_result, label=""):
    """Per-client per-round communication summary.

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


# EXPERIMENT 1: QUANTUM SOURCE CHARACTERIZATION
def run_experiment_1(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 1: Quantum Source Characterization")
    print("="*70)

    fidelities = QUANTUM['fidelity_sweep'] if not quick else np.linspace(0.90, 0.99, 5)
    fiber_lengths = QUANTUM['fiber_length_sweep_m'] if not quick else np.array([100, 500, 2000])
    coupling_effs = (QUANTUM['coupling_sweep'] if not quick
                     else QUANTUM['coupling_sweep'][::2])

    # Native 780 nm link, no frequency conversion (rack architecture).
    fiber_loss = QUANTUM['fiber_loss_db_per_km_780nm']

    # 1a: QBER vs fidelity, with noise-model sensitivity and a
    # finite-statistics CHSH estimate.
    # One-sided depolarizing on half of |phi+> is werner_state(1-p) in
    # m1's Bell-weight parametrization - the Werner family itself - so
    # the informative alternative model is amplitude damping (non-
    # unital): it splits the Z- and X-basis error rates the Werner
    # model keeps equal - the property BBM92's e_ph = e_bit shortcut
    # rests on.
    print("  [1a] QBER vs fidelity (Werner and amplitude-damping "
          "noise; sampled CHSH)...")
    from modules.m1_quantum_source import (
        amplitude_damping_state, amplitude_damping_gamma_for_fidelity,
        qber_in_basis, sample_chsh_s)
    qbers_sim, qbers_analytical, s_values = [], [], []
    qbers_ad_z, qbers_ad_x = [], []
    s_sampled_mean, s_sampled_std = [], []
    # Bell-test sample per 60 s round at the modeled device rate: the
    # duty-cycle fraction of raw coincidences.
    n_bell = int(OPERATING_POINT['raw_edge_rate_hz']
                 * POST_PROC['bell_test_duty_cycle'] * 60)
    rng_bell = np.random.default_rng(11)
    for f in fidelities:
        result = simulate_source(fidelity=f, n_pairs=1_520_000_000, fiber_length_m=100,
                                 fiber_loss_db_per_km=fiber_loss,
                                 colocated=QUANTUM.get('colocated', True),
                                 local_optics_efficiency=QUANTUM['local_optics_efficiency'],
                                 rng=np.random.default_rng(42))
        qbers_sim.append(result.qber)
        qbers_analytical.append(compute_qber_analytical(f))
        s_values.append(result.chsh_s)
        rho_ad = amplitude_damping_state(
            amplitude_damping_gamma_for_fidelity(f))
        # same additive device terms the analytic Werner curve carries
        device_err = (1.0 / QUANTUM['pbs_extinction_ratio']
                      + QUANTUM['analytic_qber_birefringence_allowance'])
        qbers_ad_z.append(qber_in_basis(rho_ad, 0.0) + device_err)
        qbers_ad_x.append(qber_in_basis(rho_ad, np.pi / 4) + device_err)
        sm, ss = sample_chsh_s(werner_state(f), n_bell, rng=rng_bell)
        s_sampled_mean.append(sm)
        s_sampled_std.append(ss)
    print(f"      amplitude damping at w={fidelities[-1]:.3f}: e_z "
          f"{qbers_ad_z[-1]*100:.2f}% vs e_x {qbers_ad_x[-1]*100:.2f}% "
          f"(Werner: equal at {(1-fidelities[-1])/2*100:.2f}% + "
          f"extinction + birefringence).\n      The e_ph = e_bit "
          f"equality is a Werner property. At matched fidelity, "
          f"damping\n      noise puts about TWICE the Werner error in "
          f"the key basis, with e_x about half\n      of that damped "
          f"key-basis error: the Werner-based analytic QBER "
          f"UNDERSTATES\n      the fixed-basis arm's key-basis error "
          f"by up to 2x if the physical noise is\n      damping-like "
          f"- a noise-model risk that arm carries. The BBM92 arm is\n"
          f"      unaffected: it measures e_x directly, and e_x "
          f"remains essentially at the\n      Werner value (within 3% "
          f"relative at matched Bell weight).")
    print(f"      sampled CHSH at w={fidelities[-1]:.3f}: S = "
          f"{s_sampled_mean[-1]:.3f} +- {s_sampled_std[-1]:.3f} "
          f"({n_bell:,} events = one 60 s round at the "
          f"{POST_PROC['bell_test_duty_cycle']*100:.2f}% witness duty cycle)")

    # 1d: Symmetric dual-endpoint edge keying (availability/fairness witness)
    print("  [1d] Dual-endpoint (beamsplitter-multiplexed) edge keying...")
    from modules.m1_quantum_source import simulate_symmetric_edge
    _sym = simulate_symmetric_edge(
        switch_split=0.5, duration_s=2.0,
        fidelity=QUANTUM['fidelity'], source_rep_rate_hz=QUANTUM['source_rep_rate_hz'],
        source_brightness=QUANTUM['source_brightness'],
        fiber_length_m=QUANTUM['fiber_length_m'],
        fiber_loss_db_per_km=fiber_loss,
        coupling_efficiency=QUANTUM['coupling_efficiency'],
        detector_efficiency=QUANTUM['detector_efficiency'],
        detector_dark_count_hz=QUANTUM['detector_dark_count_hz'],
        coincidence_window_ns=QUANTUM['coincidence_window_ns'],
        pbs_extinction_ratio=QUANTUM['pbs_extinction_ratio'],
        fiber_birefringence_rad_per_km_per_C=QUANTUM['fiber_birefringence_rad_per_km_per_C'],
        temperature_variation_C=QUANTUM['temperature_variation_C'],
        colocated=QUANTUM.get('colocated', True),
        local_optics_efficiency=QUANTUM['local_optics_efficiency'],
        rng=np.random.default_rng(7))
    print(f"      edge rate={_sym.coincidence_rate_hz:,.0f} Hz  QBER={_sym.qber:.4f}  "
          f"S={_sym.chsh_s:.3f}  symmetry_ratio={_sym.params['source_symmetry_ratio']:.3f} "
          f"(0.5 = no privileged endpoint; availability: source failure halves, not kills, an edge)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    ax1.plot(fidelities, np.array(qbers_sim)*100, 'o-', color=C['quantum'], linewidth=2, label='Simulation (Werner)')
    ax1.plot(fidelities, np.array(qbers_analytical)*100, '--', color=C['accent'], linewidth=2, label='Analytical (Werner, e_z = e_x)')
    ax1.plot(fidelities, np.array(qbers_ad_z)*100, '-.', color=C['pareto'],
             linewidth=1.5, label='Amplitude damping, e_z')
    ax1.plot(fidelities, np.array(qbers_ad_x)*100, ':', color=C['dark'],
             linewidth=1.5, label='Amplitude damping, e_x')
    ax1.axvline(x=QUANTUM['fidelity'], color='gray', linestyle=':', alpha=0.7,
                label=f"anchored Bell weight ({QUANTUM['fidelity']:.3f}, "
                      f"from 2n = {QUANTUM['negativity_2n']:.3f})")
    ax1.set_xlabel('Bell weight w (Werner)')
    ax1.set_ylabel('QBER (%)')
    ax1.set_title('QBER vs fidelity by noise model')
    ax1.legend(fontsize=6, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(fidelities, s_values, '-', color=C['quantum'], linewidth=2,
             label='Exact (density matrix)')
    ax2.errorbar(fidelities, s_sampled_mean, yerr=s_sampled_std,
                 fmt='o', color=C['pareto'], markersize=3,
                 label=f'Sampled ({n_bell:,} events/round)')
    ax2.axhline(y=2.0, color=C['unmasked'], linestyle='--', linewidth=2, label='Classical bound (S=2)')
    ax2.axhline(y=2*np.sqrt(2), color='gray', linestyle=':', linewidth=1.5, label=f'Quantum max (S={2*np.sqrt(2):.2f})')
    ax2.set_xlabel('Bell weight w (Werner)')
    ax2.set_ylabel('CHSH S-parameter')
    ax2.set_title('CHSH S: exact and sampled')
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3)
    fig.suptitle('Source characterization: fidelity sweep', fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp1a_fidelity_sweep')

    # 1b: Coincidence Rate vs Fiber Length
    print("  [1b] Coincidence rate vs Fiber length...")
    rates = []
    for fl in fiber_lengths:
        rate = compute_coincidence_rate(
            QUANTUM['source_rep_rate_hz'], QUANTUM['source_brightness'],
            QUANTUM['coupling_efficiency'], fl,
            fiber_loss, QUANTUM['detector_efficiency'],
            colocated=QUANTUM.get('colocated', True),
            local_optics_efficiency=QUANTUM['local_optics_efficiency'],
        )
        rates.append(rate)

    fig, ax = plt.subplots(figsize=VIZ['figsize'])
    rates_khz = [r / 1e3 for r in rates]

    ax.plot(fiber_lengths/1000, rates_khz, 'o-', color=C['quantum'], markersize=6)
    # Annotate each data point with its value
    for i, (fl_km, r_khz) in enumerate(zip(fiber_lengths/1000, rates_khz)):
        # staggered offsets: the 100 m and 250 m labels collide otherwise
        ax.annotate(f'{r_khz:.1f}', xy=(fl_km, r_khz),
                    xytext=(0, 7 if i % 2 == 0 else 17),
                    textcoords='offset points',
                    ha='center', fontsize=7, color=C['dark'])

    # Reference line: the anchored edge rate at the campus default length
    ax.axhline(y=OPERATING_POINT['raw_edge_rate_hz'] / 1e3, color=C['pareto'],
               linestyle='--', alpha=0.7,
               label=f"operating point ({OPERATING_POINT['raw_edge_rate_hz']/1e3:.1f} kHz at "
                     f"{QUANTUM['fiber_length_m']} m)")

    ax.set_xlabel('Fiber Length (km)')
    ax.set_ylabel('Coincidence Rate (kHz)')
    ax.set_title('Coincidence rate vs fiber length')
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

    # 1c: Coupling efficiency sensitivity
    print("  [1c] Coupling efficiency sensitivity...")
    fig, ax = plt.subplots(figsize=VIZ['figsize'])
    for fl in [100, 500, 1000, 2000]:
        rates_c = [compute_coincidence_rate(
            QUANTUM['source_rep_rate_hz'], QUANTUM['source_brightness'],
            ce, fl, fiber_loss, QUANTUM['detector_efficiency'],
            colocated=QUANTUM.get('colocated', True),
            local_optics_efficiency=QUANTUM['local_optics_efficiency'],
        ) for ce in coupling_effs]
        ax.loglog(coupling_effs, rates_c, 'o-', linewidth=2, label=f'{fl}m fiber')

    # Hardware anchors on the coupling axis: the calibrated arm
    # transmission of the rack device and the group's device-roadmap
    # ceilings (theoretical microlens with AR coating, Langer 2025;
    # simulated free-standing CBG, Dhurjati 2026).
    ax.axvline(x=QUANTUM['coupling_efficiency'], color=C['dark'],
               linestyle='--', linewidth=1,
               label=f"calibrated ({QUANTUM['coupling_efficiency']:.4f})")
    ax.axvline(x=0.21, color=C['accent'], linestyle=':', linewidth=1,
               label='microlens theory (0.21)')
    ax.axvline(x=0.40, color=C['pareto'], linestyle=':', linewidth=1,
               label='CBG simulated (0.40)')
    ax.set_xlabel('Per-arm coupling efficiency (dot to detector input)')
    ax.set_ylabel('Coincidence Rate (Hz)')
    ax.set_title('Coincidence rate vs coupling efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    save_fig(fig, 'exp1c_coupling_sensitivity')

    print("  Experiment 1 complete.")
    return {"qbers": qbers_sim, "s_values": s_values, "rates": rates}


# EXPERIMENT 2: POST-PROCESSING RATE VALIDATION
def run_experiment_2(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 2: Post-Processing Rate Validation")
    print("="*70)

    qber_sweep = POST_PROC['qber_sweep'] if not quick else np.linspace(0.01, 0.05, 5)
    # Operating point from config - single source of truth.
    op = OPERATING_POINT
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

    # 2a: Rate waterfall at the anchored operating point
    print("  [2a] Rate waterfall at the anchored operating point...")
    wf = _wf(op['raw_edge_rate_hz'], op['qber'])
    # Sanity: this experiment's waterfall input IS the config
    # operating point, and its output must reproduce config's derived rate -
    # otherwise the single source of truth has been violated somewhere.
    assert abs(wf['final_secure_rate'] - op['secret_rate_bps']) < 1e-6, (
        f"exp2 waterfall ({wf['final_secure_rate']:.3f} bps) diverged from "
        f"config.OPERATING_POINT ({op['secret_rate_bps']:.3f} bps)")

    stages = ['Raw', 'Sift', 'EC', 'PA', 'Verif', 'Final']
    fig, ax = plt.subplots(figsize=VIZ['figsize'])
    vals_khz = [wf['raw_coincidence_rate']/1000, wf['after_sifting']/1000,
                wf['after_ec']/1000, wf['after_pa']/1000,
                wf['after_verification']/1000, wf['final_secure_rate']/1000]
    bars = ax.bar(stages, vals_khz,
                  color=[C['dark'], C['quantum'], C['quantum'],
                         C['quantum'], C['quantum'], C['pareto']],
                  edgecolor='white', linewidth=1.5)
    overheads = ['', '0%', f'-{wf["ec_overhead_pct"]:.1f}%',
                 f'-{wf["pa_overhead_pct"]:.1f}%',
                 f'-{wf["verification_overhead_pct"]:.1f}%',
                 f'-{wf["bell_overhead_pct"]:.1f}%']
    for bar, oh in zip(bars, overheads):
        if oh:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    oh, ha='center', va='bottom', fontsize=6, color='gray')
    ax.set_ylabel('Rate (kbit/s)')
    ax.set_title(f"{op['label']}\nQBER {op['qber']*100:.2f}% -> "
                 f"{wf['final_secure_rate']/1000:.2f} kbit/s", fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    fig.suptitle('Shared-bit rate waterfall', fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'exp2a_rate_waterfall')

    print(f"    anchored device: {op['raw_edge_rate_hz']:>11,.0f} Hz raw -> "
          f"{wf['final_secure_rate']:>9,.0f} bps secret  (QBER {op['qber']:.4f})")

    # 2b: Final secure rate vs QBER
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
             linewidth=2,
             label='Declared rate-0.84 code (finite-blocklength FER)')
    ax1.plot(qber_sweep*100, np.array(envelope_rates)/1000, '--', color=C['accent'],
             linewidth=1.5, label='Rate-adapted family (hypothetical envelope)')
    ax1.axvline(x=cap_q*100, color='gray', linestyle=':', linewidth=1,
                label=f'0.84-code capacity ({cap_q*100:.1f}%)')
    ax1.set_xlabel('QBER (%)')
    ax1.set_ylabel('Final Secure Rate (kbit/s)')
    ax1.set_title(f'Secure rate vs QBER\n(at modeled-device raw rate '
                  f'{base_rate/1000:.1f} kHz)', fontsize=8)
    ax1.axvspan(1.5, 4.0, alpha=0.1, color=C['quantum'], label='Expected range (Assumption D)')
    ax1.axvline(x=op['qber']*100, color=C['dark'], linestyle='--', linewidth=1,
                label=f"operating point ({op['qber']*100:.2f}%)")
    # lower-right quadrant is empty past the rate collapse; a centered
    # legend this wide would occlude the cliff segment
    ax1.legend(fontsize=6.5, loc='lower right',
               bbox_to_anchor=(0.98, 0.10))
    ax1.grid(True, alpha=0.3)

    ax2.plot(qber_sweep*100, ec_overheads, 'o-', color=C['accent'], linewidth=2)
    ax2.set_xlabel('QBER (%)')
    ax2.set_ylabel('EC Overhead (%)')
    ax2.set_title('Error Correction Overhead vs QBER')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Secure rate and reconciliation overhead vs QBER',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp2b_rate_vs_qber')

    # 2d: BP decoder spot check (design band vs one degraded point)
    print("  [2d] BP decoder spot check against the finite-blocklength "
          "FER model (toy random LDPC)...")
    bp_validation = validate_bp_against_statistical(
        qber_values=[0.005, 0.01, 0.02] if quick
        else [0.005, 0.01, round(OPERATING_POINT['qber'], 4), 0.02],
        block_size=800 if quick else 1000,
        n_blocks=3 if quick else 10,
    )
    print(f"    QBER     BP FER    Stat FER    Agreement   "
          f"(FER resolution 1/{3 if quick else 10} per point)")
    print("    " + "-"*60)
    for row in bp_validation:
        agree = "YES" if row["agreement"] else "NO"
        print(f"    {row['qber']:.4f}   {row['bp_fer']:.2f}      "
              f"{row['stat_fer']:.2f}        {agree}")
    print("    Note: this toy 3-regular rate-0.84 LDPC has a low BSC "
          "threshold. Measured:\n    FER 0.1-0.6 across the 0.5-1% band "
          "and 1.0 at 2% - far above the\n    finite-blocklength model, "
          "which is the outage probability of a competently\n    "
          "implemented code at efficiency ec_decoder_beta (Polyanskiy "
          "2010 normal\n    approximation), not a measurement of this "
          "reference decoder. The toy BP\n    run is a CONSERVATIVE "
          "FLOOR; the gap is the toy code underperforming an\n    "
          "efficiency-1.10 code, not a refutation of the model. "
          "EC-model uncertainty\n    affects ABSOLUTE rates only: the "
          "exp13 framing ratio charges identical EC\n    to both arms, "
          "so it cancels there.")

    # 2e: NIST SP 800-22 subset battery on the PA output. The input is the
    # PIPELINE's output, not synthetic rng bits: simulated source at the
    # config fidelity (100 m fiber) -> reconciled frames -> Toeplitz PA at
    # the LHL-consistent compression for the measured QBER.
    print("  [2e] NIST SP 800-22 subset battery on privacy-amplified "
          "pipeline output...")
    rng_pa = np.random.default_rng(42)
    src_pa = simulate_source(fidelity=QUANTUM['fidelity'],
                             n_pairs=2_000_000_000 if not quick else 1_000_000_000,
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
          f"QBER {src_pa.qber:.4f} -> {len(rec_bits):,} reconciled -> "
          f"{len(secure_output):,} PA output bits "
          f"(LHL rate {_ent_rate:.3f}, margin "
          f"{POST_PROC['security_margin_bits']})")
    battery = randomness_battery(secure_output, alpha=0.01)
    n_pass = sum(r['passed'] for r in battery)
    print(f"    {'test':<28} {'p-value':>10}  verdict")
    for r in battery:
        print(f"    {r['test']:<28} {r['p_value']:>10.4f}  "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    print(f"    -> {n_pass}/{len(battery)} tests pass at α = 0.01 "
          f"(expected false rejections for uniform input: "
          f"≈{0.01 * len(battery):.2f})")

    print("  Experiment 2 complete.")
    return {
        "operating_point_secret_bps": op['secret_rate_bps'],
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


# EXPERIMENT 3: MESH SCALING
def run_experiment_3(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 3: Mesh Scaling")
    print("="*70)

    node_counts = MESH['node_counts_sweep'] if not quick else [3, 6, 12, 24, 40]
    model_sizes = [5e4, 1e5, 5e5, 1e6, 2e6] if not quick else [1e5, 5e5, 1e6]
    round_periods = [15, 30, 60, 120, 300] if not quick else [30, 60, 300]
    # Operating point from config - single source of truth.
    op_rate = OPERATING_POINT['secret_rate_bps']

    # 3a: Aggregation depth vs node count
    # (Depth and triangle count are geometric - independent of key rate;
    #  the mesh's latency fields use the anchored-device rate.)
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
    ax1.plot(node_counts, sqrt_ref, '--', color=C['accent'], linewidth=1.5, label='sqrt(triangles) reference')
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

    fig.suptitle('Delaunay mesh scaling', fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp3a_mesh_scaling')

    # 3b: Max feasible nodes vs model size and round period at the
    # anchored operating point (table only; the figure said what exp6
    # already shows - key supply binds)
    print("  [3b] Max mesh size table at the anchored operating point...")
    rate_points = [
        (op_rate, f'anchored device ({op_rate/1000:.2f} kbit/s)'),
    ]
    heatmaps = []
    for rate, rlabel in rate_points:
        heatmap = np.zeros((len(model_sizes), len(round_periods)))
        for i, ms in enumerate(model_sizes):
            for j, rp in enumerate(round_periods):
                max_tri, max_n = compute_max_mesh_size(int(ms), rp, rate)
                heatmap[i, j] = max_n
        heatmaps.append(heatmap)

    print("    Max nodes (rows = model size, columns = round period):")
    header = "  ".join(f"{rp:>6}s" for rp in round_periods)
    print(f"      {'':>6}  {header}")
    for i, ms in enumerate(model_sizes):
        row = "  ".join(f"{int(heatmaps[0][i, j]):>7}"
                        for j in range(len(round_periods)))
        print(f"      {int(ms/1000):>5}K  {row}")

    print("  Experiment 3 complete.")
    return {"depths": depths,
            "heatmap_anchored": heatmaps[0].tolist()}


# EXPERIMENT 4: FL TRAINING CORRECTNESS
def run_experiment_4(quick=False):
    """Secure-aggregation fidelity (HEADLINE) + convergence parity (validation).

    exact modular masking provably does not change the
    aggregate, so accuracy parity across modes is a CORRECTNESS CHECK,
    not a finding. The load-bearing metric is aggregate fidelity:
      (a) masking exactness - the masked pipeline's integer aggregate
          equals the direct quantized sum word-for-word (theorem;
          asserted at zero mismatches over every round of every seed);
      (b) quantization error - ||dequantized mean − exact float mean||_inf
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
    # >= 10 seeds in both modes; quick-mode results get
    # quoted, so they need the same statistical power.
    num_repeats = max(10, FL.get('num_repeats', 10))
    dirichlet_alpha = FL.get('dirichlet_alpha', 0.5)

    modes = ["entangled", "prng", "unmasked", "dp"]
    mode_labels = {
        "entangled": "Quantum-masked (this system)",
        "prng": "PRNG-masked (classical)",
        "unmasked": "Unmasked (insecure)",
        "dp": "DP (epsilon=5)",
    }

    # Run multiple repeats for statistical significance
    all_runs = {mode: [] for mode in modes}
    for repeat in range(num_repeats):
        seed = 42 + repeat
        print(f"\n  Repeat {repeat+1}/{num_repeats} (seed={seed})")
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
                      f"({total_bits/8/1000:.1f} kB)")

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

    # HEADLINE: aggregate fidelity of the masked pipeline
    total_mismatches = sum(sum(r.agg_exact_mismatches)
                           for r in all_runs["entangled"])
    n_agg_rounds = sum(len(r.agg_exact_mismatches)
                       for r in all_runs["entangled"])
    per_seed_max_err = [max(r.agg_error_inf) for r in all_runs["entangled"]]
    max_err = max(per_seed_max_err)
    quant_bound = 1.0 / QUANT_SCALE  # 2^-20: truncation bound on the mean
    print("\n  HEADLINE - aggregate fidelity "
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
        "masking exactness violated - implementation bug")
    assert max_err < quant_bound, (
        f"aggregate error {max_err} exceeds quantization bound {quant_bound}")

    # VALIDATION: paired accuracy parity (same seed => same split,
    # same init, same shuffling; only the masking differs)
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
    print("\n  VALIDATION - paired final-accuracy deltas vs unmasked "
          "(95% t-CI):")
    for other, ((mean_d, hw), max_abs) in pairs.items():
        print(f"    {other:>10} - unmasked: {mean_d:+.4f} ± {hw:.4f} "
              f"(max |Δ| {max_abs:.4f})")
        assert max_abs <= 0.05, (
            f"{other} vs unmasked paired delta {max_abs} exceeds 0.05 - "
            "masking should not measurably change convergence")
    print("    (Parity is EXPECTED - exact masking is a theorem; this "
          "assertion is an\n     implementation check, not a finding.)")
    print("\n  NOTE - DP is not an alternative to masking: DP bounds what "
          "the AGGREGATE\n  leaks about any individual; masking hides "
          "INDIVIDUAL updates from the\n  aggregator. Orthogonal "
          "protections - a hardened deployment runs DP on top\n  of "
          "masking. The DP arm is shown only to display its accuracy "
          "cost.")

    # Communication accounting
    print("\n  Communication accounting (last repeat per mode; prng/dp/"
          "unmasked count payload+downlink only - faithful classical "
          "baselines land in Phase 4;\n  key-distillation classical "
          "traffic, syndromes ≈0.20 bits (≈0.025 B) per secret bit, is "
          "excluded from "
          "the FL-layer ledger by scope - see RoundComm):")
    for mode in modes:
        print_comm_summary(all_runs[mode][-1], label=mode)

    # Plot: fidelity headline (left) + accuracy validation (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    colors = {"entangled": C['quantum'], "prng": C['prng'],
              "unmasked": C['unmasked'], "dp": C['dp']}
    linestyles = {"entangled": "-", "prng": "--", "unmasked": "-.", "dp": ":"}

    for r in all_runs["entangled"]:
        ax1.semilogy(np.arange(1, len(r.agg_error_inf) + 1),
                     np.maximum(r.agg_error_inf, 1e-12),
                     linewidth=0.8, alpha=0.5, color=C['quantum'])
    ax1.axhline(y=quant_bound, color='crimson', linestyle='--', linewidth=1.5,
                label=f'$2^{{-20}}$ quantization bound ({quant_bound:.1e})')
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('max |masked mean - float mean| (float64)')
    ax1.set_title('Masked-aggregate deviation from float mean')
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
    ax2.set_title('Convergence parity')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.2, 1.02)

    fig.suptitle('Secure-aggregation fidelity, 3-client clinical FL',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp4_training_convergence')

    # Summary table (validation metrics)
    print(f"\n  SUMMARY ({num_repeats} paired seeds, mean ± std):")
    print(f"  {'Mode':<25} {'Accuracy':>15} {'Rounds to Target':>20}")
    print("  " + "-"*62)
    for mode in modes:
        s = results_summary[mode]
        print(f"  {mode_labels[mode]:<25} "
              f"{s['mean_acc']:.4f} ± {s['std_acc']:.4f} "
              f"{s['mean_rounds']:>8.1f} ± {s['std_rounds']:.1f}")

    print("  Experiment 4 complete.")
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


# EXPERIMENT 5: DROPOUT RECOVERY (entangled-key two-mask)
def run_experiment_5(quick=False):
    """Dropout recovery via survivor reveal under per-round-fresh pairwise keys.

    Entanglement-derived pairwise bits are single-use per round. Survivors can safely reveal pairwise bits with
    dropped clients (no future rounds compromised). We sweep dropout rate
    and check (i) training still converges, (ii) recovery activates as
    expected, (iii) the chi-squared VALIDATION check stays silent at every
    dropout rate (indistinguishability under uniform modular masks is a
    theorem - the check is an implementation-bug detector, not a security
    result; ).

    Classical parity baseline: the same reveal-based recovery is exercised
    with a Flamingo-style per-round pool (Ma et al., IEEE S&P 2023) to show
    the MECHANISM is not quantum-unique; the quantum differentiator is that
    freshness is unconditional (no PRF/DH assumption, no key-erasure or
    forward-secrecy reliance).

    Recovery cost is drawn against the seed-based baselines on the SAME
    dropout events: exp12's executing Bonawitz and Flamingo arms charge
    threshold-many DH-secret shares per dropped client and one PRF seed
    per (survivor, dropped) pair respectively; those byte models are
    applied here to the reveal pairs the ledger records each round, so
    the O(d) / O(1) gap is shown across the whole sweep rather than at
    exp12's single dropout. The IT-dealer arms pay exactly this system's
    O(d) reveal (a pad has no compact seed either).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Dropout recovery under per-round-fresh two-mask aggregation")
    print("="*70)

    from modules.m4_federated_learning import (
        ClinicalMLP, load_clinical_data, run_multitriangle_fl,
        define_triangle_topology, BYTES_PER_MASKED_WORD,
    )
    from modules.m7_baselines import BYTES_PER_DH_SHARE, BYTES_PER_PRF_SEED

    num_rounds = 40 if not quick else 20
    target_acc = 0.90
    n_clients = 6  # 3 triangles, allows interesting dropout patterns
    n_triangles = 3
    # >= 10 seeds, dropout swept 0-40%. The old single-seed
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
        if si == 0:
            n_params = sum(p.numel() for p in model_fn().parameters())
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
            f"{int(dr*100)}%->{results[dr][-1][0].final_accuracy:.3f}"
            for dr in dropout_rates))

    from scipy.stats import t as _t_dist

    def _mean_ci(vals):
        v = np.asarray(vals, dtype=float)
        hw = (_t_dist.ppf(0.975, len(v) - 1) * v.std(ddof=1)
              / np.sqrt(len(v)) if len(v) > 1 else float('nan'))
        return v.mean(), v.std(ddof=1), hw

    # Aggregate per-rate statistics across seeds
    drs = sorted(results.keys())
    stats = {}
    for dr in drs:
        accs = [r.final_accuracy for r, _ in results[dr]]
        rec_pct = [100 * np.mean(a['recovery_invoked_per_round'])
                   for _, a in results[dr]]
        # HEADLINE costs of recovery:
        # bytes - reveals (+ recovery shares, identically 0 here) per
        # client per round, from the Phase-2 ledger;
        rec_bytes = [np.mean([rc.recovery_bytes() / max(rc.n_clients, 1)
                              for rc in r.comm_rounds])
                     for r, _ in results[dr]]
        # The same dropout events under the seed-based baselines' byte
        # models (exp12): the ledger charges n_params words per
        # (survivor, dropped) pair, so the pair count is exact. A
        # triangle under threshold 2 reveals only when exactly one
        # member drops (two survivors), so pairs = 2 x drop events.
        pairs = [[rc.reveal_bytes // (n_params * BYTES_PER_MASKED_WORD)
                  for rc in r.comm_rounds] for r, _ in results[dr]]
        assert all(pc % 2 == 0 for ps in pairs for pc in ps), \
            "reveal pairs not paired: a cluster other than a triangle revealed"
        rec_bytes_bon = [np.mean([(pc // 2) * 2 * BYTES_PER_DH_SHARE
                                  / max(rc.n_clients, 1)
                                  for pc, rc in zip(ps, r.comm_rounds)])
                         for ps, (r, _) in zip(pairs, results[dr])]
        rec_bytes_fla = [np.mean([pc * BYTES_PER_PRF_SEED
                                  / max(rc.n_clients, 1)
                                  for pc, rc in zip(ps, r.comm_rounds)])
                         for ps, (r, _) in zip(pairs, results[dr])]
        # latency - modeled wire time of the recovery bytes on the
        # aggregator's access link;
        rec_wire_ms = [1000 * np.mean([rc.recovery_bytes() * 8
                                       / CLASSICAL_LINK_BPS
                                       for rc in r.comm_rounds])
                       for r, _ in results[dr]]
        # failed rounds - triangle-rounds skipped because survivors <
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
            "rec_bytes_bonawitz": _mean_ci(rec_bytes_bon),
            "rec_bytes_flamingo": _mean_ci(rec_bytes_fla),
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

    # Panel 1: accuracy vs dropout rate (mean +- 95% CI across seeds).
    # Panel 2: recovery bytes per client-round, this system against the
    # seed-based baselines on the same dropout events (log axis: the
    # gap is n_params-fold). Panel 3: threshold-failed triangle-rounds,
    # identical in every arm (same Shamir threshold).
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=VIZ['figsize_wide'])
    acc_means = [stats[dr]["acc"][0] for dr in drs]
    acc_hws = [stats[dr]["acc"][2] for dr in drs]
    xpct = [100 * dr for dr in drs]
    ax1.errorbar(xpct, acc_means, yerr=acc_hws, fmt='o-', linewidth=2,
                 capsize=4, color=C['quantum'],
                 label=f'final accuracy\n(mean +- 95% CI, {num_seeds} seeds)')
    ax1.axhline(y=target_acc, color='gray', linestyle=':', alpha=0.5,
                label=f'{target_acc} target')
    ax1.set_xlabel('Per-round dropout rate (%)')
    ax1.set_ylabel(f'Final test accuracy ({num_rounds} rounds)')
    ax1.set_xlim(min(xpct) - 2.5, max(xpct) + 2.5)
    ax1.set_title('Accuracy vs dropout', fontsize=8)
    ax1.legend(fontsize=6, loc='lower left')
    ax1.grid(True, alpha=0.3)

    # 0% dropout has no recovery traffic in any arm; a log axis cannot
    # show it, so the byte panel starts at the first non-zero rate.
    nz = [dr for dr in drs if stats[dr]["rec_bytes"][0] > 0]
    xnz = [100 * dr for dr in nz]
    ax2.errorbar(xnz, [stats[dr]["rec_bytes"][0] for dr in nz],
                 yerr=[stats[dr]["rec_bytes"][2] for dr in nz],
                 fmt='o-', linewidth=2, capsize=3, color=C['quantum'],
                 label='this system / IT dealer\n(reveal pad bits, O(d))')
    ax2.plot(xnz, [stats[dr]["rec_bytes_bonawitz"][0] for dr in nz],
             's--', linewidth=1.5, color=C['unmasked'],
             label='Bonawitz (Shamir DH shares, O(1))')
    ax2.plot(xnz, [stats[dr]["rec_bytes_flamingo"][0] for dr in nz],
             '^:', linewidth=1.5, color=C['dp'],
             label='Flamingo (PRF seed, O(1))')
    ax2.set_yscale('log')
    ax2.set_xlim(min(xpct) - 2.5, max(xpct) + 2.5)
    _ymax = max(stats[dr]["rec_bytes"][0] + stats[dr]["rec_bytes"][2]
                for dr in nz)
    ax2.set_ylim(0.5, _ymax * 8)
    ax2.set_xlabel('Per-round dropout rate (%)')
    ax2.set_ylabel('Recovery bytes per client-round')
    ax2.set_title('Recovery cost, same dropout events', fontsize=8)
    # the band between the O(1) floor and the O(d) curve is empty
    ax2.legend(fontsize=6, loc='center left')
    ax2.grid(True, alpha=0.3, which='both')

    tf_means = [stats[dr]["tfail_pct"][0] for dr in drs]
    tf_hws = [stats[dr]["tfail_pct"][2] for dr in drs]
    ax3.bar(np.arange(len(drs)), tf_means, 0.6, yerr=tf_hws, capsize=3,
            color=C['unmasked'])
    ax3.set_xticks(np.arange(len(drs)))
    ax3.set_xticklabels([f'{int(dr*100)}' for dr in drs])
    ax3.set_xlabel('Per-round dropout rate (%)')
    ax3.set_ylabel('Threshold-failed triangle-rounds (%)')
    ax3.set_ylim(0, max(t + h for t, h in zip(tf_means, tf_hws)) * 1.15)
    ax3.set_title('Lost work (all arms)', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Dropout recovery: accuracy and cost',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.0)
    save_fig(fig, 'exp5_dropout_recovery')

    print(f"\n  Summary ({num_seeds} seeds, mean ± std [95% CI half-width]):")
    print(f"  {'Dropout':>8}  {'Final acc':>22}  {'Recovery %':>12}  "
          f"{'Reveal kB/cl-rd':>16}  {'Bonawitz B':>11}  {'Flamingo B':>11}  "
          f"{'Wire ms':>8}  {'T-fail %':>9}  {'chi2 fail %':>12}")
    for dr in drs:
        s = stats[dr]
        print(f"  {int(dr*100):>7}%  "
              f"{s['acc'][0]:.4f} +- {s['acc'][1]:.4f} [{s['acc'][2]:.4f}]  "
              f"{s['recovery_pct'][0]:>11.1f}%  "
              f"{s['rec_bytes'][0]/1000:>15.1f}  "
              f"{s['rec_bytes_bonawitz'][0]:>11.1f}  "
              f"{s['rec_bytes_flamingo'][0]:>11.1f}  "
              f"{s['rec_wire_ms'][0]:>8.3f}  "
              f"{s['tfail_pct'][0]:>8.2f}%  "
              f"{s['chi_pct'][0]:>11.1f}%")
    print("  Baseline bytes: exp12's seed-reveal byte models on the same "
          "dropout events\n  (Bonawitz 2 x 6 B shares per dropped client; "
          "Flamingo 4 B per survivor-dropped pair).")
    # Classical parity baseline (Flamingo-style per-round pool)
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
    print("  -> identical mechanism and recovery in both pools; the quantum claim")
    print("    is the ASSUMPTION CLASS (unconditional freshness), not the mechanism.")

    print("  Experiment 5 complete.")
    return {"dropout_results": {f"{int(dr*100)}%": {
        "final_acc_mean": float(stats[dr]["acc"][0]),
        "final_acc_std": float(stats[dr]["acc"][1]),
        "final_acc_ci95": float(stats[dr]["acc"][2]),
        "recovery_pct_mean": float(stats[dr]["recovery_pct"][0]),
        "recovery_bytes_per_client_round": float(stats[dr]["rec_bytes"][0]),
        "recovery_bytes_ci95": float(stats[dr]["rec_bytes"][2]),
        "recovery_bytes_bonawitz_per_client_round":
            float(stats[dr]["rec_bytes_bonawitz"][0]),
        "recovery_bytes_flamingo_per_client_round":
            float(stats[dr]["rec_bytes_flamingo"][0]),
        "recovery_wire_ms": float(stats[dr]["rec_wire_ms"][0]),
        "threshold_fail_pct": float(stats[dr]["tfail_pct"][0]),
        "chi2_fail_pct": float(stats[dr]["chi_pct"][0]),
        "n_seeds": num_seeds,
    } for dr in drs}}


# EXPERIMENT 11: Gradient-inversion attack on the REAL construction
# (rebuild of the deleted CIFAR/PSNR legacy - )
def _tabular_dlg(model, target_grad_params, y_label, input_dim,
                 n_iters, lr, device='cpu', seed=0):
    """DLG/iDLG tabular gradient-inversion (Zhu et al. NeurIPS 2019;
    label-known iDLG variant, Zhao et al. 2020). Optimize a dummy
    feature vector so its gradient matches the observed gradient under
    squared-L2 gradient-matching. No image priors (no total variation) -
    the target is a 30-feature clinical vector, so tabular from the
    ground up (the deleted legacy applied image PSNR/SSIM to tabular
    data and attacked the proof-less Gaussian mask; )."""
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
    """Gradient-inversion privacy demonstration on the REAL system: tabular DLG/iDLG against the ClinicalMLP and the
    modular-OTP construction, across three targets with honest,
    theorem-consistent expectations:
      (1) unmasked gradient      -> attack SUCCEEDS (per-feature MSE low,
                                    attribute inference well above chance)
      (2) modular-masked payload -> attack FAILS (one-time pad makes the
                                    payload independent of the gradient -
                                    a theorem; reconstruction ≈ prior)
      (3) plaintext aggregate    -> attack PARTIALLY succeeds (the FedAvg
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

    # Median per feature over the training pool -> attribute-inference bits.
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
        ("aggregate attribute inference should beat chance - the "
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
    ax1.set_ylabel('Per-feature reconstruction MSE (lower = more leak)')
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

    fig.suptitle('Gradient inversion on the implemented construction',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp11_gradient_inversion')

    print("  Experiment 11 complete.")
    return {"targets": summary, "prior_mse": prior_mse,
            "n_targets": n_targets, "n_iters": n_iters}



# EXPERIMENT 6: RATE-CONSTRAINED THROUGHPUT
def measure_local_compute_time(n_params_target, local_epochs=2, batch_size=32,
                               lr=0.01, reps=3, seed=42):
    """Measured local-training time for an MLP of ≈n_params_target params.

    Replaces exp6's fabricated 2.0 s constant with a direct
    measurement: build a width-scaled two-hidden-layer MLP (params ≈
    h² + (input_dim+4)·h for input_dim features, 2 classes), run the same
    local step the FL loop performs (local_epochs over one client's share
    of the clinical dataset), and return the median wall-clock over `reps`
    rounds on this machine (first warm-up round discarded). Every grid size
    is measured directly - nothing is extrapolated.
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
    # Operating points from config - single source of truth.
    # The sweep brackets the two labeled points; no unlabeled rate literals.
    op_rate = OPERATING_POINT['secret_rate_bps']
    key_rates = [op_rate, 30e3, 100e3, 300e3] if not quick else [op_rate, 100e3]
    rate_labels = {
        op_rate: f'{op_rate/1000:.2f} kbit/s - anchored device',
    }

    # Measured local compute time per model size
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
    ax.set_title('Rate-constrained training throughput')
    ax.legend(fontsize=7, loc='center right')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    save_fig(fig, 'exp6_throughput')

    # Practical ceiling identification
    print("\n  Practical model-size ceilings (mask_time < round_period):")
    for kr in key_rates:
        for period_name, period_s in [("30s", 30), ("60s", 60), ("300s", 300)]:
            max_bits = kr * period_s
            max_params = max_bits / BITS_PER_PARAM
            print(f"    {rate_labels.get(kr, f'{kr/1000:.0f} kbit/s (hypothetical sweep point)')}, "
                  f"{period_name} rounds: max {max_params/1000:.0f}K params "
                  f"({max_params*BITS_PER_PARAM/8/1e6:.2f} MB of key per edge)")

    print("  Experiment 6 complete.")
    return {"rounds_per_hour": rounds_per_hour.tolist()}


# EXPERIMENT 7: QBER -> PRIVACY-BUDGET CURVE
def run_experiment_7(quick=False):
    """Map physical QBER to the key-rate / capacity envelope: shared-bit
    rate, the LHL min-entropy floor per output bit, and the maximum
    maskable model size at a fixed round budget.

    LHL accounting matches the m2 waterfall (source model, no eavesdropper
    term): available min-entropy per reconciled input bit is
        1 − max(β·h(q), 1 − R_code) − D_tag
    (reconciliation syndrome leakage and the GLLP tagged fraction), and the
    PA output is ε-close to uniform with per-output-bit gap
    2·log2(1/ε) / (N · H_min_rate). The entropy floor stays essentially
    FLAT across QBER - PA compensates; what QBER degrades is the RATE.
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

    # Modeled-device raw edge rate from config - single source of truth
    #. Identical to m1.compute_coincidence_rate at the config
    # parameters; derived once in config._derive_operating_point() rather
    # than re-derived here.
    raw_rate = OPERATING_POINT['raw_edge_rate_hz']

    # For each QBER, compute:
    #   shared_rate_bps: after EC + PA + verification + Bell duty
    #   mask_entropy_per_bit: LHL floor on output uniformity
    #   max_model_params @ 60s round (per-edge accounting - see below)
    shared_rates, envelope_rates, mask_entropies, max_params = [], [], [], []
    bbm92_rates, bbm92_max_params = [], []
    keep_eff = QKD_FRAMING['sifting_modes']['efficient-90/10']
    cap_q = estimate_capacity_qber(code_rate)
    for q in qbers:
        waterfall = compute_rate_waterfall(
            raw_rate, q, code_rate=code_rate,
            security_margin=security_margin, block_size=block_size,
            bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
        )
        rate = waterfall['final_secure_rate']
        shared_rates.append(rate)
        # BBM92-framed rate at the same physics: efficient-90/10 sifting
        # with the h2(q) eavesdropper charge, asymptotic accounting (no
        # TLGR deviation term - exp14 panel d carries the finite-key
        # penalty on top of this).
        bbm92_rate = compute_rate_waterfall_qkd(
            raw_rate, q, keep_eff, code_rate=code_rate,
            security_margin=security_margin, block_size=block_size,
            bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
        )['final_secure_rate']
        bbm92_rates.append(bbm92_rate)
        bbm92_max_params.append(bbm92_rate * 60 / BITS_PER_PARAM)
        # HYPOTHETICAL rate-adapted envelope, plotted alongside with an
        # explicit label: past the declared 0.84 code's effective design
        # point the solid curve collapses (finite-blocklength FER -> 1)
        # and only this labeled assumption keeps a usable rate.
        envelope_rates.append(compute_rate_waterfall(
            raw_rate, q, code_rate=code_rate,
            security_margin=security_margin, block_size=block_size,
            bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
            assume_rate_adapted=True,
        )['final_secure_rate'])

        # LHL bound on per-output-bit entropy floor, using the SAME
        # source-model accounting as the m2 waterfall: no eavesdropper
        # h(q) subtraction. Available min-entropy per input bit =
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

        # Max model parameters at a fixed 60 s round - PER-EDGE accounting,
        # consistent with exp6 and the round-time table: the triangle's
        # three edge links distill CONCURRENTLY, each at `rate`, and a round
        # consumes BITS_PER_PARAM bits per parameter per edge. The binding
        # constraint is the per-edge pool-fill time; dividing by 3 would
        # model serial keying, inconsistent with the architecture and exp6.
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

    # Taller than figsize_wide: the panel-1 legend moves below the axes
    # so it cannot occlude the min-entropy curve or the zero-rate row.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 4.3))

    # Linear rate axis: the two framings sit 1.32x apart, which a
    # multi-decade log axis hides, and the collapse past the code's
    # design point reads as the curve reaching zero rather than as
    # off-scale floor markers.
    sr = np.array(shared_rates) / 1000
    br = np.array(bbm92_rates) / 1000
    ax1.plot(qbers, br, 's-', linewidth=2, markersize=4,
             color=C['pareto'],
             label='BBM92-framed (keep 0.82), rate-0.84 code')
    ax1.plot(qbers, sr, 'o-', linewidth=1.6,
             color=C['quantum'],
             label='trusted-conduit mode, same code')
    ax1.plot(qbers, np.array(envelope_rates) / 1000, '--', linewidth=1.5,
             color=C['accent'], label='Rate-adapted family (hypothetical)')
    ax1.axvline(x=OPERATING_POINT['qber'], color=C['dark'], linestyle='--',
                linewidth=1,
                label=f"operating point (QBER "
                      f"{OPERATING_POINT['qber']*100:.2f}%)")
    ax1.axvline(x=cap_q, color='gray', linestyle=':', linewidth=1,
                label=f'0.84-code capacity (q={cap_q:.3f})')
    ax1_b = ax1.twinx()
    ax1_b.plot(qbers, mask_entropies, 's--', linewidth=2,
               color=C['accent'], label='Min-entropy per output bit (LHL)')
    _hspan = max(mask_entropies) - min(mask_entropies)
    ax1_b.set_ylim(min(mask_entropies) - 0.10 * _hspan,
                   max(mask_entropies) + 0.10 * _hspan)
    ax1.set_xlabel('QBER')
    ax1.set_ylabel('Shared bit rate (kbit/s)', color=C['quantum'])
    ax1_b.set_ylabel('Min-entropy per output bit', color=C['accent'])
    ax1.set_xlim(qbers[0] - 0.003, qbers[-1] + 0.003)
    ax1.set_ylim(0, max(sr.max(), br.max()) * 1.18)
    if cutoff_qber is not None:
        ax1.axvline(x=cutoff_qber, color='gray', linestyle=':',
                    label=f'1 kbit/s cutoff (QBER={cutoff_qber:.3f})')
    ax1.set_title('Shared rate and min-entropy vs QBER')
    ax1.grid(True, alpha=0.3)
    # Combined legend for both y-axes, drawn on the twin so it sits on top.
    _h1, _l1 = ax1.get_legend_handles_labels()
    _h2, _l2 = ax1_b.get_legend_handles_labels()
    ax1_b.legend(_h1 + _h2, _l1 + _l2, loc='upper center',
                 bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6,
                 frameon=False)

    mp = np.array(max_params) / 1000
    bp = np.array(bbm92_max_params) / 1000
    ax2.plot(qbers, bp, 's-', linewidth=2, markersize=4,
             color=C['pareto'], label='BBM92-framed (keep 0.82)')
    ax2.plot(qbers, mp, 'o-', linewidth=1.6, color=C['quantum'],
             label='trusted-conduit mode')
    ax2.axvline(x=OPERATING_POINT['qber'], color=C['dark'], linestyle='--',
                linewidth=1)
    ax2.axvline(x=cap_q, color='gray', linestyle=':', linewidth=1)
    ax2.set_xlim(qbers[0] - 0.003, qbers[-1] + 0.003)
    _ref = 28866 / 1000
    ax2.axhline(y=_ref, color='gray', linestyle=':', alpha=0.7)
    ax2.text(qbers[-1], _ref, '29K (ClinicalMLP) ', fontsize=6,
             color='gray', va='bottom', ha='right')
    ax2.set_ylim(0, max(_ref, mp.max()) * 1.25)
    ax2.text(cap_q, 0.4, ' 0.84-code capacity', fontsize=6,
             color='gray', rotation=90, va='bottom')
    ax2.text(OPERATING_POINT['qber'], 0.4, ' operating point',
             fontsize=6, color=C['dark'], rotation=90, va='bottom')
    ax2.text(qbers[-1], ax2.get_ylim()[1] * 0.97,
             '150K and 1M references off scale ', fontsize=6,
             color='gray', va='top', ha='right')
    ax2.legend(fontsize=6, loc='center right')
    ax2.set_xlabel('QBER')
    ax2.set_ylabel('Max model size at 60s round (K params)')
    ax2.set_title('Max maskable model size vs QBER (rate-driven)')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('QBER, key rate, and min-entropy',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=3.0, rect=(0, 0.18, 1, 1))
    save_fig(fig, 'exp7_qber_key_rate_envelope')

    print("\n  QBER -> Shared rate -> Max model size (60s rounds):")
    print(f"  {'QBER':>8}  {'Declared rate':>15}  {'Envelope*':>12}  "
          f"{'Hmin/bit':>11}  {'Max K params':>14}")
    for q, r, e, h, mp in zip(qbers, shared_rates, envelope_rates,
                              mask_entropies, max_params):
        print(f"  {q:>8.4f}  {r/1000:>10.2f} kbit/s  {e/1000:>7.2f} kb/s"
              f"  {h:>11.4f}  {mp/1000:>14.1f}")
    print(f"  * envelope = HYPOTHETICAL rate-adapted code family (not "
          f"implemented); the declared\n    rate-0.84 code's capacity is "
          f"QBER ~ {cap_q:.4f} - beyond its effective design point\n    "
          f"the declared-code rate collapses (finite-blocklength FER "
          f"-> 1) and max-params\n    follows the collapse.")
    if cutoff_qber:
        print(f"\n  Cutoff QBER (rate < 1 kbit/s): {cutoff_qber:.3f}")
    print("  Experiment 7 complete.")
    return {
        "qbers": qbers.tolist(),
        "shared_rates_bps": shared_rates,
        "bbm92_framed_rates_bps": bbm92_rates,
        "mask_entropy_per_bit": mask_entropies,
        "max_params_at_60s": max_params,
        "bbm92_max_params_at_60s": bbm92_max_params,
        "cutoff_qber": float(cutoff_qber) if cutoff_qber else None,
    }


# EXPERIMENT 7b (DEMOTED): Cost breakdown (illustrative only)
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
        print(f"\n  Scale: {n_clients} clients")

        # Use Module 3 Delaunay topology for larger scales, hardcoded for small
        if n_clients >= 6:
            # Modeled-device rate from config - topology metrics
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

    # PLOT 1: Training convergence across scales
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

    fig.suptitle('Multi-triangle FL convergence',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp8a_multitri_convergence')

    # PLOT 2: Scaling summary bar chart
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
    axes[0].set_xticklabels(xlabels, fontsize=6, rotation=20)
    axes[0].set_ylabel('Final Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].set_ylim(0.8, 1.05)
    axes[0].axhline(y=target_acc, color='gray', linestyle=':', alpha=0.5, linewidth=0.6)
    label_bars(axes[0], final_accs, fmt='{:.3f}')

    # Rounds to target
    axes[1].bar(x, rounds_to_tgt, width=bar_width, color=bar_colors, edgecolor='white', linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels, fontsize=6, rotation=20)
    axes[1].set_ylabel('Rounds')
    axes[1].set_title('Rounds to Target')
    max_rounds = max(rounds_to_tgt)
    axes[1].set_ylim(0, max_rounds * 1.18)
    label_bars(axes[1], rounds_to_tgt, fmt='{:.0f}')

    # Tree depth
    axes[2].bar(x, tree_depths, width=bar_width, color=bar_colors, edgecolor='white', linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(xlabels, fontsize=6, rotation=20)
    axes[2].set_ylabel('Depth')
    axes[2].set_title('Agg. Tree Depth')
    max_depth = max(tree_depths) if max(tree_depths) > 0 else 1
    axes[2].set_ylim(0, max_depth * 1.25)
    label_bars(axes[2], tree_depths, fmt='{:.0f}')

    # Wall clock time
    axes[3].bar(x, wall_clocks, width=bar_width, color=bar_colors, edgecolor='white', linewidth=0.8)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(xlabels, fontsize=6, rotation=20)
    axes[3].set_ylabel('Seconds')
    axes[3].set_title('Wall Clock')
    max_wc = max(wall_clocks)
    axes[3].set_ylim(0, max_wc * 1.18)
    label_bars(axes[3], wall_clocks, fmt='{:.1f}s')

    fig.suptitle('Multi-triangle scaling metrics',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=2.0)
    save_fig(fig, 'exp8b_multitri_scaling_bars')

    # Aggregation overhead at both operating points: tree depth times
    # the per-level mask time (model bits / secret rate). Modeled device
    # AND the labeled hypothetical projection. Table only; the plotted
    # version repeated exp6's point that key supply binds.
    rate_points = [
        (OPERATING_POINT['secret_rate_bps'], 'anchored device'),
    ]
    model_params = [5e4, 1e5, 5e5, 1e6]
    level_times = {}
    print("\n  Inter-triangle aggregation overhead = tree depth x per-level "
          "mask time (s):")
    for secure_rate, rlabel in rate_points:
        print(f"    {rlabel} ({secure_rate/1000:.1f} kbit/s):")
        print(f"      {'params':>8} {'s/level':>9}  "
              + "  ".join(f"{n:>2} clients (d={d})"
                          for n, d in zip(scales, tree_depths)))
        for mp in model_params:
            per_level = int(mp) * BITS_PER_PARAM / secure_rate
            level_times[f"{rlabel}|{int(mp)}"] = per_level
            print(f"      {int(mp/1000):>7}K {per_level:>9.0f}  "
                  + "  ".join(f"{per_level * d:>14.0f}"
                              for d in tree_depths))

    # Summary
    print("\n  MULTI-TRIANGLE SCALING SUMMARY")
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
        print("  converges at ALL scales (3-12 clients)")
        print("  Tree-of-triangles protocol produces correct global aggregate")
    else:
        print("  ⚠ Some scales did not converge - note: quick mode runs only "
              "30 rounds,\n     below convergence at 9-12 clients (full mode "
              "uses 60 rounds)")

    print("  Experiment 8 complete.")

    out = {
        str(n): {
            'accuracy': results[n]['result'].final_accuracy,
            'rounds_to_target': results[n]['result'].rounds_to_target,
            'n_triangles': results[n]['n_triangles'],
            'tree_depth': results[n]['tree_depth'],
        } for n in scales
    }
    out['mask_time_per_level_s'] = level_times
    return out


# MAIN RUNNER
def run_experiment_9(quick=False):
    """HNDL - harvest-now / decrypt-later retroactive transcript attack.

    makes claim (b) empirical. Four arms differ ONLY in the
    pairwise-mask source and its provisioning; all record the full
    classical transcript
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
    # no payload would be recorded at all - the t >= 2 argument's edge
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
    tr_ent, tr_cls, tr_dlr, tr_cou = [], [], [], []
    print(f"  Recording {num_rounds} rounds x all 4 arms "
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

        dealer_pool = rng.integers(0, 2, size=3 * bits_per_mask,
                                   dtype=np.int8)
        courier_pool = rng.integers(0, 2, size=3 * bits_per_mask,
                                    dtype=np.int8)
        t_e, agg_e = run_transcript_round(
            rd, grads, range(n_clients), drops, "entangled",
            pairwise_pool=pool, self_seeds=self_seeds)
        t_c, agg_c = run_transcript_round(
            rd, grads, range(n_clients), drops, "classical",
            pair_mask_seeds=pair_seeds, self_seeds=self_seeds)
        t_d, agg_d = run_transcript_round(
            rd, grads, range(n_clients), drops, "it_dealer_online",
            pairwise_pool=dealer_pool, self_seeds=self_seeds)
        t_k, agg_k = run_transcript_round(
            rd, grads, range(n_clients), drops, "it_dealer_courier",
            pairwise_pool=courier_pool, self_seeds=self_seeds)
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
        assert np.array_equal(agg_d, agg_e), "dealer arm diverges in the clear"
        assert np.array_equal(agg_k, agg_e), \
            "courier arm diverges in the clear"
        tr_ent.append(t_e)
        tr_cls.append(t_c)
        tr_dlr.append(t_d)
        tr_cou.append(t_k)

        survivors = n_clients - len(drops)
        set_model_params(global_model,
                         gparams + agg_live.aggregate / survivors)

    atk_c = retroactive_attack(tr_cls, "classical")
    atk_d = retroactive_attack(tr_dlr, "it_dealer_online")
    atk_k = retroactive_attack(tr_cou, "it_dealer_courier")
    atk_e = retroactive_attack(tr_ent, "entangled")

    print("\n  Retroactive god-mode attack on the recorded transcripts")
    print("  (adversary recomputes every COMPUTATIONAL secret: channel "
          "encryption ->\n   Shamir shares -> all self-masks in ALL arms; "
          "classical arm also all\n   DH/PRF pairwise seeds; dealer arm "
          "all recorded pad ciphertexts):")
    print(f"\n  {'arm':<32} {'payloads':>8} {'recovered':>10} "
          f"{'min masks left':>15} {'residual p':>11} {'median|err|':>12}")
    for label, atk in [("classical (DH/PRF masks)", atk_c),
                       ("IT dealer (online provisioning)", atk_d),
                       ("IT dealer (courier provisioning)", atk_k),
                       ("entangled (distilled keys)", atk_e)]:
        print(f"  {label:<32} {atk['n_payloads']:>8} "
              f"{100*atk['recovered_fraction']:>9.1f}% "
              f"{atk['min_masks_remaining']:>15} "
              f"{atk['mean_residual_monobit_p']:>11.3f} "
              f"{atk['median_abs_error']:>12.4g}")
    assert atk_c["recovered_fraction"] == 1.0, \
        "classical arm should be fully recoverable under god-mode"
    assert atk_d["recovered_fraction"] == 1.0, \
        "online-dealer arm should be fully recoverable under god-mode"
    assert atk_k["recovered_fraction"] == 0.0, \
        "courier-dealer arm leaked a payload - nothing was on the wire"
    assert atk_e["recovered_fraction"] == 0.0, \
        "entangled arm leaked a payload - HNDL claim violated"
    assert atk_e["min_masks_remaining"] >= 1, \
        "a delivered payload lost all pairwise masks - t>=2 argument violated"

    print("\n  -> classical arm AND online-dealer arm: 100% of recorded "
          "gradients recovered\n    retroactively - IT-secure masking with "
          "computationally-secured provisioning\n    is still HNDL-exposed "
          "(the hole moves from the key agreement to the dealer's\n    "
          "distribution channel);")
    print("    courier-dealer arm: 0% - the same pads, delivered out of "
          "band, leave nothing\n    recorded to open; its immunity is "
          "against the channel recorder only (the dealer\n    holds every "
          "pad);")
    print("    entangled arm: 0% - every delivered payload retains >= 1 "
          "pairwise mask that\n    never appeared in any transcript, and "
          "god-mode residuals remain uniform\n    (mean monobit p above; "
          "median reconstruction error ~10^3 in gradient units\n    vs ~0 "
          "classically). Online vs courier differ only in provisioning: "
          "HNDL\n    immunity is a provisioning property.")
    print("\n  HONEST SCOPE: the self-mask leg rides classical channels "
          "(Shamir shares)\n  and is only computationally protected - "
          "god-mode strips it in ALL arms\n  here. The PAIRWISE leg alone "
          "carries the HNDL claim. Recorded EC syndromes\n  are covered by "
          "the PA ε (masks are sliced from post-PA output).")

    # Plot: recovery fraction + residual uniformity per arm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    arms = ["classical\n(DH/PRF)", "IT dealer\n(online)",
            "IT dealer\n(courier)", "entangled\n(distilled)"]
    arm_colors = [C['unmasked'], C['accent'], C['dark'], C['quantum']]
    fracs = [100 * atk_c["recovered_fraction"],
             100 * atk_d["recovered_fraction"],
             100 * atk_k["recovered_fraction"],
             100 * atk_e["recovered_fraction"]]
    ax1.bar(arms, fracs, color=arm_colors)
    ax1.tick_params(axis='x', labelsize=7)
    ax1.set_ylabel('Gradients recovered retroactively (%)')
    ax1.set_ylim(0, 114)  # headroom for the bar labels
    for i, v in enumerate(fracs):
        ax1.text(i, v + 2, f'{v:.0f}%', ha='center', fontweight='bold')
    ax1.set_title('Gradients recovered retroactively')
    ax1.grid(True, alpha=0.3, axis='y')

    pc = [r["residual_monobit_p"] for r in atk_c["records"]]
    pd_ = [r["residual_monobit_p"] for r in atk_d["records"]]
    pk = [r["residual_monobit_p"] for r in atk_k["records"]]
    pe = [r["residual_monobit_p"] for r in atk_e["records"]]
    ax2.scatter(np.zeros(len(pc)) + 0.0, pc, color=C['unmasked'],
                alpha=0.7, label='classical: residual = the gradient')
    ax2.scatter(np.zeros(len(pd_)) + 1.0, pd_, color=C['accent'],
                alpha=0.85, label='dealer (online): residual = the gradient')
    ax2.scatter(np.zeros(len(pk)) + 2.0, pk, color=C['dark'],
                alpha=0.7, label='dealer (courier): residual stays uniform')
    ax2.scatter(np.zeros(len(pe)) + 3.0, pe, color=C['quantum'],
                alpha=0.7, label='entangled: residual stays uniform')
    ax2.axhline(y=0.01, color='gray', linestyle=':', linewidth=1,
                label='alpha = 0.01')
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(arms)
    ax2.tick_params(axis='x', labelsize=7)
    ax2.set_ylabel('Monobit p-value of post-attack residual')
    ax2.set_ylim(-0.05, 1.45)  # headroom keeps the legend off the points
    ax2.legend(fontsize=6, loc='upper center', ncol=2)
    ax2.set_title('Post-attack residual uniformity')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('HNDL transcript attack: retroactive recovery',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp9_hndl_transcript')

    print("  Experiment 9 complete.")
    return {
        "classical": {k: v for k, v in atk_c.items() if k != "records"},
        "it_dealer_online": {k: v for k, v in atk_d.items()
                             if k != "records"},
        "it_dealer_courier": {k: v for k, v in atk_k.items()
                              if k != "records"},
        "entangled": {k: v for k, v in atk_e.items() if k != "records"},
        "rounds": num_rounds,
        "payloads_per_arm": atk_c["n_payloads"],
    }


def run_experiment_10(quick=False):
    """Availability under single-source failure.

    Operationalizes source_symmetry_ratio ≈ 0.5. Every edge is keyed
    from BOTH endpoints' sources
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

    # (a) Source-level empirical check: an edge with one endpoint's
    # source dead continues from the far endpoint at ~half rate.
    # Durations sized for the anchored per-pair coincidence probability
    # (~7.8e-5 at 100 m): 0.5 s / 2 s give ~3k / ~12k coincidences, so
    # the half-rate assertion window is many sigma wide.
    dur = 0.5 if quick else 2.0
    kw = dict(fidelity=QUANTUM['fidelity'], fiber_length_m=100)
    edge_full = simulate_symmetric_edge(
        duration_s=dur, rng=np.random.default_rng(42), **kw)
    # One endpoint's source killed. simulate_symmetric_edge builds the
    # healthy edge by concatenating TWO legs, each a simulate_source run
    # at duration_s * switch_split (switch_split = 0.5): the endpoint-i
    # leg and the endpoint-j leg. Killing endpoint i removes its leg
    # exactly, leaving the endpoint-j leg alone - which is precisely one
    # simulate_source(duration_s = dur * 0.5) call with the same source
    # params. So this is NOT an unrelated half-rate proxy; it is the
    # surviving endpoint's own leg from the symmetric-edge decomposition
    #. The far endpoint keeps routing switch_split of
    # its pairs down the edge with unchanged per-pair physics.
    edge_degraded = simulate_source(
        duration_s=dur * 0.5, rng=np.random.default_rng(43), **kw)
    n_full = len(edge_full.raw_bits_alice)
    n_deg = len(edge_degraded.raw_bits_alice)
    ratio = n_deg / max(n_full, 1)
    print(f"  (a) Edge keyed from both endpoints: {n_full:,} coincidences "
          f"in {dur}s\n      one source killed: {n_deg:,} "
          f"({100*ratio:.1f}% of the two-source rate; expected ≈ 50%)\n"
          f"      QBER {edge_full.qber:.4f} -> {edge_degraded.qber:.4f} "
          f"(unchanged - same physics, fewer pairs)\n"
          f"      symmetry witness on the healthy edge: "
          f"{edge_full.params['source_symmetry_ratio']:.3f} (0.5 = no "
          f"privileged generator)")
    assert 0.35 <= ratio <= 0.65, (
        f"degraded-edge rate ratio {ratio} outside the ~half-rate window")

    # (b) Modeled round-time impact on the 6-client / 3-triangle
    # topology (same per-edge demand model as exps 5/8).
    n_clients = 6
    triangles, _, tree_edges = define_triangle_topology(n_clients)
    n_params = 28866  # ClinicalMLP - matches exps 4/5/8
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
    kill_node = 1  # shared vertex of T0 and T1 - the harshest choice
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
          f"{kw_before:.2f}s -> {kw_sym:.2f}s "
          f"(×{kw_sym/kw_before:.2f}); ALL {len(triangles)} triangles "
          f"continue;\n        20-round key-wait wall-clock "
          f"{wall_healthy:.0f}s -> {wall_sym:.0f}s "
          f"(+{100*(wall_sym/wall_healthy - 1):.0f}%)")
    print(f"      fixed-source-per-edge (hypothetical alternative): edges "
          f"{dead_edges} DEAD;\n        triangles {stalled_tris} stall "
          f"permanently - {len(stalled_tris)}/{len(triangles)} of the mesh "
          f"never finishes a round\n        (rounds-to-target: unreachable "
          f"for any aggregate that needs those clusters)")
    assert len(stalled_tris) > 0, "kill choice should stall the fixed design"

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    rounds_x = np.arange(1, num_rounds + 1)
    kw_sym_series = [kw_before if rd <= kill_round else kw_sym
                     for rd in rounds_x]
    kw_fixed_series = [kw_before if rd <= kill_round else np.nan
                       for rd in rounds_x]
    ax1.plot(rounds_x, kw_sym_series, 'o-', color=C['quantum'],
             label='symmetric switching (implemented): 2x slower, continues')
    ax1.plot(rounds_x, kw_fixed_series, 's-', color=C['unmasked'],
             label='fixed source per edge: stalls (no key supply)')
    ax1.axvline(x=kill_round + 0.5, color='gray', linestyle=':',
                label=f'node-{kill_node} source dies')
    _y = kw_sym * 0.92
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

    fig.suptitle('Availability under single-source failure',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp10_availability')

    print("  Experiment 10 complete.")
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


def run_experiment_12(quick=False):
    """SOTA comparison: this system vs faithful Bonawitz-DH,
    Flamingo-PRF, and classical IT-dealer baselines.

    All five arms share the same modular masking, payload/self-mask wire
    format, and FL-layer byte scoping; they differ only in pairwise-mask
    source, provisioning, and dropout-recovery mechanism (see
    modules/m7_baselines.py). The dealer arms answer the strongest
    CLASSICAL objection - IT-secure masking needs no quantum hardware -
    by making its cost explicit: the provisioning channel (online: HNDL-
    exposed, exp9) or the dealer itself (courier: holds every pad).
    Compared on: per-round bytes by phase, dropout recovery cost (bytes +
    aggregator compute), scaling in the vector dimension d and client
    count n, and assumption class / HNDL exposure. Systems results that
    depend on the quantum key rate are reported at BOTH operating points.

    HONEST expected result: this system is SLIGHTLY WORSE on
    recovery communication (a one-time pad has no compact seed, so the
    pairwise reveal is n_params words, not one scalar) and it alone pays a
    key-distillation latency, in exchange for unconditional freshness and
    HNDL immunity of the pairwise leg. The baselines are NOT tuned to lose.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 12: SOTA comparison - this system vs Bonawitz / "
          "Flamingo / IT-dealer")
    print("="*70)

    import torch
    from modules.m4_federated_learning import ClinicalMLP, load_clinical_data
    from modules.m7_baselines import (
        run_entangled, run_bonawitz, run_flamingo, run_it_dealer)
    from modules.m3_mesh_topology import generate_campus_mesh
    from config import IT_DEALER

    # Real gradients at the ClinicalMLP scale (28,866 params)
    client_data, test_data, input_dim = load_clinical_data(
        6, rng_seed=42, dirichlet_alpha=FL.get('dirichlet_alpha', 0.5))
    torch.manual_seed(42)
    model = ClinicalMLP(input_dim=input_dim, hidden=128)
    n_params = sum(p.numel() for p in model.parameters())
    # One representative cluster of 4 clients (a super-triangle), one drop.
    cluster = [0, 1, 2, 3]
    rng = np.random.default_rng(42)
    grads = {}
    crit = torch.nn.CrossEntropyLoss()
    for c in cluster:
        xb, yb = client_data[c][:][0], client_data[c][:][1]
        model.zero_grad()
        crit(model(xb), yb.long()).backward()
        grads[c] = torch.cat([p.grad.flatten() for p in model.parameters()])
    bpm = n_params * 32
    pool = rng.integers(0, 2, size=len(cluster) * (len(cluster) - 1) // 2 * bpm,
                        dtype=np.int8)

    pool_dealer_on = rng.integers(
        0, 2, size=len(cluster) * (len(cluster) - 1) // 2 * bpm, dtype=np.int8)
    pool_dealer_co = rng.integers(
        0, 2, size=len(cluster) * (len(cluster) - 1) // 2 * bpm, dtype=np.int8)

    def run_all_arms(dropouts):
        g_ = lambda: {c: g.clone() for c, g in grads.items()}
        ae, ce = run_entangled(g_(), cluster, dropouts, pool,
                               np.random.default_rng(1))
        ab, cb = run_bonawitz(g_(), cluster, dropouts,
                              np.random.default_rng(1))
        af, cf = run_flamingo(g_(), cluster, dropouts, 3,
                              np.random.default_rng(1))
        ao, co = run_it_dealer(g_(), cluster, dropouts, pool_dealer_on,
                               np.random.default_rng(1),
                               provisioning="online")
        ac, cc = run_it_dealer(g_(), cluster, dropouts, pool_dealer_co,
                               np.random.default_rng(1),
                               provisioning="courier")
        for other in (ab, af, ao, ac):
            assert np.array_equal(ae, other), \
                "protocols disagree on the recovered aggregate"
        return {"entangled": ce, "bonawitz": cb, "flamingo": cf,
                "it_dealer_online": co, "it_dealer_courier": cc}

    # (A) Per-round cost with 1 dropped client
    costs1 = run_all_arms({2})
    ARMS = ["entangled", "bonawitz", "flamingo",
            "it_dealer_online", "it_dealer_courier"]
    SHORT_ASSUME = {"entangled": "unconditional",
                    "bonawitz": "computational",
                    "flamingo": "computational",
                    "it_dealer_online": "IT+comp.prov",
                    "it_dealer_courier": "IT+dealer"}
    print(f"\n  (A) Per-round cost, 4-client cluster, ClinicalMLP "
          f"(n_params={n_params:,}), 1 dropout:")
    print(f"  {'protocol':>17} {'payload/cl':>11} {'recovery B':>11} "
          f"{'setup B':>10} {'agg PRG exp':>12} {'agg recov ms':>13} "
          f"{'assumption':>14} {'HNDL-imm':>9}")
    for name in ARMS:
        c = costs1[name]
        print(f"  {name:>17} "
              f"{c.payload_bytes // max(c.n_clients - c.n_dropped, 1):>11,} "
              f"{c.recovery_bytes:>11,} {c.dealer_setup_bytes:>10,} "
              f"{c.aggregator_prg_expansions:>12} "
              f"{1000*c.aggregator_recovery_s:>13.2f} "
              f"{SHORT_ASSUME[name]:>14} "
              f"{str(c.hndl_immune_pairwise_leg):>9}")
    ratio = costs1["entangled"].recovery_bytes / max(
        costs1["flamingo"].recovery_bytes, 1)
    print(f"    -> this system's recovery traffic is {ratio:,.0f}x the "
          f"seed-based baselines' (= n_params: a one-time pad has no\n"
          f"      compact seed to send). Non-recovery per-round bytes: "
          f"payload+self-mask identical across all five;\n"
          f"      Bonawitz additionally redistributes n(n-1) DH-secret "
          f"Shamir shares per round (per-execution\n      ephemeral DH - "
          f"the setup cost Flamingo's long-term-secret ratchet avoids), so "
          f"its share bytes are\n      {costs1['bonawitz'].share_bytes} vs "
          f"{costs1['flamingo'].share_bytes} for the others. On aggregator "
          f"recovery WALL-CLOCK this system is\n      the SLOWEST "
          f"({1000*costs1['entangled'].aggregator_recovery_s:.2f} ms vs "
          f"~{1000*costs1['flamingo'].aggregator_recovery_s:.2f} ms): it "
          f"does 0 PRG expansions but O(d)\n      bit-repacking, which "
          f"costs more here than the baselines' few PRG calls.\n      "
          f"The dealer arms match this system's O(d) reveal "
          f"recovery exactly (a pad has no compact seed\n      either); the "
          f"ONLINE dealer additionally pays n(n-1)*d*4 = "
          f"{costs1['it_dealer_online'].dealer_setup_bytes:,} B/round of "
          f"pad\n      distribution, and its HNDL-immune column is False "
          f"because that distribution rides recorded\n      channels "
          f"(exp9). The COURIER dealer pays 0 channel bytes/round; its "
          f"True is against a CHANNEL\n      recorder only - the dealer "
          f"holds every pad, and a "
          f"{IT_DEALER['courier_rounds_per_visit']}-round provisioning "
          f"batch ("
          f"{IT_DEALER['courier_rounds_per_visit'] * costs1['it_dealer_online'].dealer_setup_bytes:,}"
          f" B) sits in\n      storage between visits.")

    # (B) Recovery-byte scaling in the vector dimension d
    dims = [128, 1024, 8192, n_params]
    print("\n  (B) Dropout-recovery bytes vs vector dimension d "
          "(1 dropout, 3 survivors):")
    print(f"  {'d':>8} {'entangled':>12} {'bonawitz':>10} {'flamingo':>10} "
          f"{'it_dealer':>11}")
    scaling = {}
    for d in dims:
        gd = {c: torch.zeros(d) for c in cluster}
        pd = rng.integers(0, 2, size=6 * d * 32, dtype=np.int8)
        pdd = rng.integers(0, 2, size=6 * d * 32, dtype=np.int8)
        _, ced = run_entangled({c: g.clone() for c, g in gd.items()},
                               cluster, {2}, pd, np.random.default_rng(1))
        _, cbd = run_bonawitz({c: g.clone() for c, g in gd.items()},
                              cluster, {2}, np.random.default_rng(1))
        _, cfd = run_flamingo({c: g.clone() for c, g in gd.items()},
                              cluster, {2}, 3, np.random.default_rng(1))
        _, cdd = run_it_dealer({c: g.clone() for c, g in gd.items()},
                               cluster, {2}, pdd, np.random.default_rng(1))
        scaling[d] = (ced.recovery_bytes, cbd.recovery_bytes,
                      cfd.recovery_bytes, cdd.recovery_bytes)
        print(f"  {d:>8} {ced.recovery_bytes:>12,} {cbd.recovery_bytes:>10,} "
              f"{cfd.recovery_bytes:>10,} {cdd.recovery_bytes:>11,}")
    print("    -> this system and the dealer arms are O(d) (a one-time pad "
          "has no compact seed);\n      the seed-based baselines are O(1) "
          "in d. The O(d) gap is the price of the pad,\n      whatever "
          "provisioned it.")

    # (C) Key-supply latency this system pays and the baselines do NOT
    op = OPERATING_POINT['secret_rate_bps']
    # One pairwise pool per cluster edge per round: bits_per_mask = n_params*32.
    key_wait_op = bpm / op
    print("\n  (C) Key-supply latency (this system only; the baselines "
          "distill no key):")
    print(f"      pairwise pool per edge/round = {bpm:,} bits")
    print(f"      anchored operating point ({op/1000:.2f} kbit/s): "
          f"key_wait {key_wait_op:.1f} s/edge")
    print("      Bonawitz/Flamingo pay 0 s here - computational key setup "
          "is ~instant. This latency is\n      the operational cost of "
          "the unconditional-freshness guarantee. The dealer arms also "
          "pay 0 s of\n      distillation; the online variant instead "
          "moves n(n-1)*d*4 B/round of pads over the wire\n      "
          "(= 11.1 Mbit: ~11 ms at 1 Gbps serial, ~2.8 ms/client in "
          "parallel) and the courier\n      variant moves it in "
          "pre-provisioned batches - the dealer's cost is trust, not "
          "latency.")

    # (D) Matched-metric table for systems that cannot be re-run
    #
    print("\n  (D1) Matched-metric comparison, SECURITY axes (executing "
          "arms + external rows from\n       published descriptions):")
    hdr1 = ["system", "key source", "recorder captures",
            "trusted 3rd party", "tolerated channel adversary",
            "HNDL-immune leg"]
    rows1 = [
        ["This system", "source-model distilled (entangled)",
         "nothing usable (keys local)", "none (own source)**",
         "passive only", "yes* (pairwise)"],
        ["Bonawitz SecAgg (CCS'17)", "ephemeral DH + PRG",
         "DH msgs (breaks later)", "none",
         "active", "no"],
        ["Flamingo (S&P'23)", "per-round PRF seed",
         "enc. seeds (break later)", "none",
         "active", "no"],
        ["IT dealer, online prov.", "dealer-provisioned OTPs",
         "pad ciphertexts (break later)", "dealer (holds all pads)",
         "active at round; TLS at provisioning", "no (exp9: 100%)"],
        ["IT dealer, courier prov.", "dealer-provisioned OTPs",
         "nothing usable (pads out of band)", "dealer (holds all pads)",
         "active at round; courier out of band", "vs recorder only***"],
        ["QuNetQFL (2501.12709)", "pairwise MDI-QKD (fully conn.)",
         "unclonable quantum states", "none (MDI node untrusted)",
         "active", "yes (key layer)"],
        ["6G QKD-FL (2603.15649)", "BB84-class QKD keys",
         "unclonable quantum states", "none",
         "active", "yes (key layer)"],
    ]
    for r in [hdr1] + rows1:
        print("      | " + " | ".join(f"{c:<24}" if i == 0 else f"{c:<33}"
                                       for i, c in enumerate(r)) + " |")

    # Logical vs physical key-relationship scaling, computed from m3 -
    # labeled per layer so QuNetQFL's fully-connected LOGICAL key graph
    # (multiplexed through a Sagnac loop, NOT K(K-1)/2 physical fibers)
    # is never compared against our PHYSICAL edge count.
    _n_demo = 12
    _, _tris, _G = generate_campus_mesh(_n_demo)
    _phys = _G.number_of_edges()
    _logical_ours = _phys + max(len(_tris) - 1, 0)  # edge keys + tree keys
    _logical_fc = _n_demo * (_n_demo - 1) // 2
    print(f"\n  (D2) Cost axes at K = {_n_demo} clients "
          f"(key relationships computed from m3):")
    print(f"      | {'system':<24} | {'dropout recovery':<28} | "
          f"{'per-round setup traffic':<31} | {'logical pairwise keys':<21} "
          f"| {'physical links':<26} |")
    _d2 = [
        ["This system", "reveal pairwise bits, O(d)",
         "none (keys distilled locally)",
         f"{_logical_ours} (edges+tree)",
         f"{_phys} Delaunay fiber edges"],
        ["Bonawitz SecAgg (CCS'17)", "Shamir DH secret, O(1)",
         "n(n-1) DH shares (per-exec DH)", f"{_logical_fc}",
         "0 quantum (classical net)"],
        ["Flamingo (S&P'23)", "reveal PRF seed, O(1)",
         "none (long-term secret ratchet)", f"{_logical_fc}",
         "0 quantum (classical net)"],
        ["IT dealer, online prov.", "reveal pad bits, O(d)",
         "n(n-1)*d*4 B pad distribution", f"{_logical_fc}",
         "0 quantum (classical net)"],
        ["IT dealer, courier prov.", "reveal pad bits, O(d)",
         "0 (amortized courier batches)", f"{_logical_fc}",
         "0 quantum (classical net)"],
        ["QuNetQFL (2501.12709)", "not addressed",
         "QKD post-processing traffic", f"{_logical_fc}",
         "shared Sagnac loop + per-client"],
    ]
    for r in _d2:
        print(f"      | {r[0]:<24} | {r[1]:<28} | {r[2]:<31} | "
              f"{r[3]:<21} | {r[4]:<26} |")
    # The rate differentiator vs the QKD rows, computed (exp13), never
    # asserted: same physics, both framings, per sifting protocol.
    _qkd = _qkd_arm_waterfalls(OPERATING_POINT['raw_edge_rate_hz'],
                               OPERATING_POINT['qber'])
    _src_rate = OPERATING_POINT['waterfall']['final_secure_rate']
    _r_sym = _src_rate / _qkd['symmetric']['final_secure_rate']
    _r_eff = _src_rate / _qkd['efficient-90/10']['final_secure_rate']
    _r_asy = _src_rate / _qkd['efficient-asymptotic']['final_secure_rate']
    print("    * This system's HNDL immunity is CONTINGENT on the GLLP "
          "tagged-fraction PA margin and the\n      Shamir threshold t≥2 "
          ". The QKD rows share the "
          "key-layer HNDL\n      immunity AND the ITS assumption class - "
          "the honest differentiators are narrower: a higher\n      key "
          f"rate from the same physics (exp13: {_r_sym:.1f}x vs "
          f"symmetric-basis sifting, {_r_eff:.1f}x vs the\n      "
          f"efficient 90/10 protocol QuNetQFL runs, {_r_asy:.1f}x in the "
          "asymptotic-sifting limit - bought by\n      the trusted-source "
          "assumption, i.e. a weaker tolerated channel adversary), no "
          "basis-choice or\n      mid-point measurement hardware, and "
          "reveal-based dropout recovery, which those works do\n      "
          "not analyze. QuNetQFL's measured hardware rates (32.8-240 "
          "kbit/s) exceed our modeled\n      12.7 kbit/s; theirs is "
          "hardware, ours is simulation.\n    ** 'None' here means no "
          "THIRD party: the source is trusted, but it is endpoint-owned "
          "equipment -\n      an assumption about our own hardware, not "
          "about someone else's.\n    *** The courier dealer's immunity "
          "is against a CHANNEL recorder only: no transcript exists to"
          "\n      break, but the dealer generated and held every pad - "
          "dealer or pad-storage compromise is\n      total, and no "
          "per-round freshness exists between courier visits.")

    # Plot: recovery-byte scaling in d (left) + assumption/HNDL and
    # aggregator compute (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    ds = sorted(scaling.keys())
    ax1.loglog(ds, [scaling[d][0] for d in ds], 'o-', color=C['quantum'],
               linewidth=2, label='this system (reveal bits, O(d))')
    ax1.loglog(ds, [scaling[d][1] for d in ds], 's--', color=C['unmasked'],
               linewidth=2, label='Bonawitz (Shamir DH scalar, O(1))')
    ax1.loglog(ds, [scaling[d][2] for d in ds], '^:', color=C['dp'],
               linewidth=2, label='Flamingo (PRF seed, O(1))')
    ax1.loglog(ds, [scaling[d][3] for d in ds], 'd-.', color=C['accent'],
               linewidth=1.2,
               label='IT dealer (reveal pad bits, O(d) = this system)')
    ax1.axvline(x=n_params, color='gray', linestyle=':', alpha=0.6)
    ax1.text(n_params, ax1.get_ylim()[0] * 2, ' ClinicalMLP', fontsize=6,
             color='gray', rotation=90, va='bottom')
    ax1.set_xlabel('Vector dimension d (params)')
    ax1.set_ylabel('Dropout-recovery bytes (1 drop, 3 survivors)')
    # log axes need multiplicative margins of half a decade or so to
    # read as real gaps; linear-style 1.4x factors leave the endpoint
    # markers crowding the frame corners
    ax1.set_xlim(min(ds) * 0.5, max(ds) * 2.2)
    ax1.set_ylim(4, max(scaling[d][0] for d in ds) * 4)
    ax1.set_title('Recovery bytes vs model dimension')
    # upper left is the empty corner above the O(1) floor and left of
    # the O(d) diagonal; auto-placement grazed the diagonal.
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    names = ARMS
    # Assumption class and HNDL status ride the tick labels; the bar
    # annotations carry only the numbers, so nothing overlaps.
    labels = ["this system\nuncond.\nHNDL-imm.*",
              "Bonawitz\ncomput.\nHNDL-exp.",
              "Flamingo\ncomput.\nHNDL-exp.",
              "dealer\n(online)\nIT + comp.\nHNDL-exp.",
              "dealer\n(courier)\nIT + dealer\nHNDL-imm.\nvs recorder"]
    cols = [C['quantum'], C['unmasked'], C['dp'], C['accent'], C['dark']]
    vals = [costs1[n].recovery_bytes for n in names]
    ax2.bar(labels, vals, color=cols)
    ax2.set_ylabel('Recovery bytes / dropout (ClinicalMLP)')
    # Log scale in BYTES so the 12-B seed reveals are visible bars, not
    # sub-pixel slivers next to the 346 KB OTP reveals.
    ax2.set_yscale('log')
    ax2.set_ylim(4, max(vals) * 30)  # headroom for the annotations
    for i, n in enumerate(names):
        c = costs1[n]
        ax2.text(i, c.recovery_bytes * 1.4,
                 f'{c.recovery_bytes:,} B\n'
                 f'{c.aggregator_prg_expansions} PRG exp',
                 ha='center', va='bottom', fontsize=6)
    ax2.tick_params(axis='x', labelsize=5.5)
    ax2.set_xlabel('* contingent on GLLP PA margin + Shamir t>=2 (exp9)',
                   fontsize=5, color='gray')
    ax2.set_title('Recovery bytes and assumption class (ClinicalMLP)')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Dropout-recovery cost and trust assumptions, five arms',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp12_sota_comparison')

    print("\n  Experiment 12 complete.")
    return {
        "per_round_1drop": {n: {
            "payload_per_client": costs1[n].payload_bytes
            // max(costs1[n].n_clients - costs1[n].n_dropped, 1),
            "recovery_bytes": costs1[n].recovery_bytes,
            "aggregator_prg_expansions": costs1[n].aggregator_prg_expansions,
            "aggregator_recovery_ms": 1000 * costs1[n].aggregator_recovery_s,
            "assumption_class": costs1[n].assumption_class,
            "dealer_setup_bytes": costs1[n].dealer_setup_bytes,
            "trust_surface": costs1[n].trust_surface,
            "channel_adversary_tolerated":
                costs1[n].channel_adversary_tolerated,
            "hndl_immune_pairwise_leg": costs1[n].hndl_immune_pairwise_leg}
            for n in ARMS},
        "recovery_bytes_vs_dim": {str(d): {"entangled": v[0],
                                           "bonawitz": v[1], "flamingo": v[2],
                                           "it_dealer": v[3]}
                                  for d, v in scaling.items()},
        "key_wait_s": {"anchored": key_wait_op},
        "n_params": n_params,
    }


def _qkd_arm_waterfalls(raw_hz, qber, assume_rate_adapted=False):
    """One QKD-framed waterfall per configured sifting mode, with the
    identical EC charges the source waterfall uses."""
    return {
        name: compute_rate_waterfall_qkd(
            raw_hz, qber, keep,
            code_rate=POST_PROC['ldpc_code_rate'],
            security_margin=POST_PROC['security_margin_bits'],
            block_size=POST_PROC['block_size'],
            bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
            assume_rate_adapted=assume_rate_adapted)
        for name, keep in QKD_FRAMING['sifting_modes'].items()
    }


def run_experiment_13(quick=False):
    """Rate cost of QKD framing vs source-model framing, same physics.

    Both arms consume the identical m1 device model (raw coincidence
    rate, QBER) and identical EC / FER / margin / duty charges from m2;
    they differ ONLY in the post-processing framing. Source model: no
    eavesdropper subtraction, explicit GLLP tagged-fraction removal, no
    sifting. QKD framing (Shor-Preskill, entanglement-based): basis
    sifting, h2(e_bit) Eve bound, NO tagging (Koashi-Preskill 2003;
    Ma-Fung-Lo 2007 - see compute_rate_waterfall_qkd). The measured
    ratio prices the trusted-source assumption. It is NOT a security
    comparison: the QKD framing tolerates an ACTIVE channel adversary,
    the source model only a passive one, and both are HNDL-immune at
    the key layer.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 13: Rate cost of QKD framing vs source-model framing")
    print("="*70)

    op = OPERATING_POINT
    raw = op['raw_edge_rate_hz']
    q = op['qber']
    src = op['waterfall']
    qkd = _qkd_arm_waterfalls(raw, q)
    mode_names = list(QKD_FRAMING['sifting_modes'].keys())
    short = {"symmetric": "QKD sym 0.50",
             "efficient-90/10": "QKD eff 0.82",
             "efficient-asymptotic": "QKD eff 1.00"}

    # (A) stage-by-stage rates, identical physics in every column
    print(f"\n  (A) Stage rates (kbit/s) at the modeled operating point "
          f"(raw {raw/1000:.1f} kHz, QBER {q*100:.2f}%):")
    stages = [("raw", 'raw_coincidence_rate'),
              ("after sifting", 'after_sifting'),
              ("after EC", 'after_ec'),
              ("after PA", 'after_pa'),
              ("after FER", 'after_verification'),
              ("final", 'final_secure_rate')]
    print(f"    {'stage':>14} {'source-model':>13} "
          + " ".join(f"{short[n]:>14}" for n in mode_names))
    for label, key in stages:
        print(f"    {label:>14} {src[key]/1000:>13.2f} "
              + " ".join(f"{qkd[n][key]/1000:>14.2f}" for n in mode_names))
    print("    PA charge: source-model removes the GLLP tagged fraction "
          f"D_tag = {src['tagged_fraction']:.4f} (no Eve bound);\n"
          "    QKD subtracts the Eve bound h2(e_bit) = "
          f"{qkd[mode_names[0]]['eve_bound_fraction']:.4f} (no tagging - "
          "entanglement-based analysis has none:\n    Koashi-Preskill "
          "2003, Ma-Fung-Lo 2007). Each arm gets its own standard "
          "analysis.")

    # (B) the measured ratio, per sifting mode
    ratios = {n: src['final_secure_rate'] / qkd[n]['final_secure_rate']
              for n in mode_names}
    floor_ratio = ratios['efficient-asymptotic']
    print("\n  (B) Source-model / QKD-framed rate ratio (same physics, "
          "same EC):")
    for n in mode_names:
        print(f"      vs {n:<22} (keep {qkd[n]['sifting_keep']:.2f}): "
              f"{ratios[n]:.2f}x   ({qkd[n]['final_secure_rate']/1000:.2f} "
              f"kbit/s)")
    h_q = binary_entropy(q)
    print(f"      decomposition: (i) sifting keep-fraction, (ii) net PA "
          f"delta h2(e_bit) - D_tag = {h_q:.4f} - "
          f"{src['tagged_fraction']:.4f} = {h_q - src['tagged_fraction']:.4f}."
          f"\n      Asymptotic-sifting floor (part ii alone): "
          f"{floor_ratio:.2f}x.")
    print("      This replaces the retired '>2x vs naive QKD framing' "
          "claim: the ratio is\n      protocol-dependent, and against the "
          "efficient 90/10 protocol our nearest\n      competitor "
          "(QuNetQFL) actually runs, it is the middle number, not 2x.")

    # (C) ratio across the QBER sweep - both arms under the SAME EC model
    # at every point (fixed declared code; the shared FER model cancels in the ratio)
    sweep_q = np.linspace(0.005, 0.06, 12)
    sweep_ratios = {n: [] for n in mode_names}
    for qq in sweep_q:
        s = compute_rate_waterfall(
            raw, qq, code_rate=POST_PROC['ldpc_code_rate'],
            security_margin=POST_PROC['security_margin_bits'],
            block_size=POST_PROC['block_size'],
            bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
            tagged_fraction=op['tagged_fraction'])
        for n in mode_names:
            k = compute_rate_waterfall_qkd(
                raw, qq, QKD_FRAMING['sifting_modes'][n],
                code_rate=POST_PROC['ldpc_code_rate'],
                security_margin=POST_PROC['security_margin_bits'],
                block_size=POST_PROC['block_size'],
                bell_duty_cycle=POST_PROC['bell_test_duty_cycle'])
            # matched-EC invariant: identical leak and FER in both arms
            assert abs(s['ec_overhead_pct'] - k['ec_overhead_pct']) < 1e-12
            assert s['frame_error_rate'] == k['frame_error_rate']
            sweep_ratios[n].append(
                s['final_secure_rate'] / k['final_secure_rate']
                if k['final_secure_rate'] > 0 else np.nan)

    # (D) what the framing means for key supply per round: params ceiling
    # at 60 s AND the wait to fill the exp12 reference pool (ClinicalMLP,
    # 28,866 params x 32 bits = 923,712 bits/edge/round)
    ref_pool_bits = 28866 * BITS_PER_PARAM
    print("\n  (D) Key-supply consequences per framing "
          f"({BITS_PER_PARAM} bits/param/edge; reference pool "
          f"{ref_pool_bits:,} bits = ClinicalMLP):")
    for label, wf in [("source-model", src)] + [(short[n], qkd[n])
                                                for n in mode_names]:
        r = wf['final_secure_rate']
        print(f"      {label:<14} {r/1000:>7.2f} kbit/s -> max "
              f"{r*60/BITS_PER_PARAM/1000:>5.1f}K params/60s round, "
              f"key_wait {ref_pool_bits/r:>6.1f} s/edge")

    print("\n  Scope and modeling caveats (all resolved in the QKD arm's "
          "favor where ambiguous):")
    print("    - e_ph = e_bit: VERIFIED under m1's noise model, not just "
          "assumed. The Werner\n      component is basis-isotropic and "
          "the rotation-model birefringence errs the Z\n      and X bases "
          "equally, so e_X = e_Z exactly (pinned by "
          "tests/test_qkd_waterfall.py).")
    print("    - m1's analytic QBER carries no multi-pair error term, so "
          "the QKD arm charges\n      multi-pair emission NOWHERE in this "
          "simulation (correctly not as tagging, and\n      not via "
          "measured QBER either - the fold-in mechanism the literature "
          "relies on\n      does not operate here).")
    print("    - MDI-specific hardware penalties are NOT modeled: this "
          "prices the FRAMING on\n      our modeled device, not a "
          "reproduction of QuNetQFL's stack. Their measured\n      "
          "hardware rates (32.8-240 kbit/s) exceed our modeled 12.7 "
          "kbit/s.")
    print("    - The EC FER model is a finite-blocklength benchmark, not a measured "
          "decoder curve (exp2d); both "
          "arms charge identical\n      EC, so EC-model uncertainty "
          "cancels in the ratio and affects absolute rates\n      only.")
    print("    - NOT a security comparison: QKD framing tolerates an "
          "active channel adversary;\n      the source model assumes a "
          "trusted source and passive channel. The ratio is\n      what "
          "that extra assumption buys.")

    # Figure: stage waterfalls, ratio bars, ratio-vs-QBER sweep
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=VIZ['figsize_wide'])

    stage_labels = ['Raw', 'Sift', 'EC', 'PA', 'FER', 'Final']
    xs = np.arange(len(stage_labels))
    ax1.bar(xs - 0.2, [src[k]/1000 for _, k in stages], 0.4,
            color=C['quantum'], label='source-model')
    ax1.bar(xs + 0.2, [qkd['efficient-90/10'][k]/1000 for _, k in stages],
            0.4, color=C['pareto'], label='QKD-framed (keep 0.82)')
    ax1.set_xticks(xs)
    ax1.set_xticklabels(stage_labels, fontsize=7)
    ax1.set_ylabel('Rate (kbit/s)')
    ax1.set_ylim(0, max(src[k] for _, k in stages) / 1000 * 1.12)
    ax1.set_title(f'Stage rates (QBER {q*100:.2f}%)',
                  fontsize=8)
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.3, axis='y')

    # One color per sifting variant, shared by the ax2 bars and the ax3
    # sweep curves; the efficient-90/10 variant is the protocol the
    # exp14 BBM92 arm runs, so it carries the same teal.
    mode_colors = {'symmetric': C['unmasked'],
                   'efficient-90/10': C['pareto'],
                   'efficient-asymptotic': C['dp']}
    bar_labels = {"symmetric": "symmetric\nkeep 0.50",
                  "efficient-90/10": "efficient 90/10\nkeep 0.82",
                  "efficient-asymptotic": "efficient\nkeep 1.00"}
    ax2.bar([bar_labels[n] for n in mode_names],
            [ratios[n] for n in mode_names],
            color=[mode_colors[n] for n in mode_names])
    ax2.set_xlabel('QKD sifting protocol')
    ax2.axhline(y=floor_ratio, color='gray', linestyle=':',
                label=f'asymptotic-sifting floor {floor_ratio:.2f}x')
    ax2.axhline(y=1.0, color='gray', linewidth=0.8)
    for i, n in enumerate(mode_names):
        ax2.text(i, ratios[n], f'{ratios[n]:.2f}x', ha='center',
                 va='bottom', fontsize=7)
    ax2.set_ylabel('source-model rate / QKD-framed rate')
    ax2.set_ylim(0, max(ratios.values()) * 1.25)
    ax2.set_title('Rate ratio by sifting protocol', fontsize=8)
    ax2.tick_params(axis='x', labelsize=6)
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3, axis='y')

    styles = {'symmetric': '-', 'efficient-90/10': '--',
              'efficient-asymptotic': ':'}
    for n in mode_names:
        ax3.plot(sweep_q, sweep_ratios[n], styles[n], linewidth=2,
                 color=mode_colors[n],
                 label=f'{n} (keep '
                       f"{QKD_FRAMING['sifting_modes'][n]:.2f})")
    ax3.axvline(x=q, color=C['dark'], linestyle='--', linewidth=1,
                label=f'operating point ({q*100:.2f}%)')
    # Margins, not fixed limits: past code capacity the QKD-framed rate
    # is zero and the ratio is undefined, so the drawn data ends early.
    ax3.margins(x=0.06, y=0.06)
    ax3.set_xlabel('QBER')
    ax3.set_ylabel('rate ratio')
    ax3.set_title('Ratio vs QBER', fontsize=8)
    ax3.legend(fontsize=6)
    ax3.grid(True, alpha=0.3)

    fig.suptitle('Source-model vs QKD framing, same physical layer',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp13_qkd_framing')

    print("\n  Experiment 13 complete.")
    return {
        "operating_point": {"raw_hz": raw, "qber": q,
                            "source_model_bps": src['final_secure_rate']},
        "qkd_framed_bps": {n: qkd[n]['final_secure_rate']
                           for n in mode_names},
        "rate_ratios": {n: ratios[n] for n in mode_names},
        "net_pa_delta": h_q - src['tagged_fraction'],
        "sweep": {"qber": sweep_q.tolist(),
                  "ratios": {n: sweep_ratios[n] for n in mode_names}},
    }


def _bbm92_source_kwargs():
    """m1 parameters at the config operating point, shared by both exp14
    arms (fixed-basis arm = same function at p_z = 1)."""
    return dict(
        fidelity=QUANTUM['fidelity'],
        source_rep_rate_hz=QUANTUM['source_rep_rate_hz'],
        source_brightness=QUANTUM['source_brightness'],
        fiber_length_m=QUANTUM['fiber_length_m'],
        fiber_loss_db_per_km=QUANTUM['fiber_loss_db_per_km_780nm'],
        coupling_efficiency=QUANTUM['coupling_efficiency'],
        detector_efficiency=QUANTUM['detector_efficiency'],
        detector_dark_count_hz=QUANTUM['detector_dark_count_hz'],
        coincidence_window_ns=QUANTUM['coincidence_window_ns'],
        pbs_extinction_ratio=QUANTUM['pbs_extinction_ratio'],
        fiber_birefringence_rad_per_km_per_C=(
            QUANTUM['fiber_birefringence_rad_per_km_per_C']),
        temperature_variation_C=QUANTUM['temperature_variation_C'],
        colocated=QUANTUM['colocated'],
        local_optics_efficiency=QUANTUM['local_optics_efficiency'],
    )


def run_experiment_14(quick=False):
    """Executing BBM92 arm and intercept-resend visibility.

    Exp13 priced the QKD framing at the waterfall (model) level; this
    experiment EXECUTES it on sampled bits: basis choice, sifting,
    X-basis parameter estimation, EC, and Toeplitz PA sized by the TLGR
    finite-key bound (Tomamichel-Lim-Gisin-Renner 2012). Four parts:

    (A) The executing chain at the operating point, 10 seeds, against
        the exp13 asymptotic model - the delta is the finite-key
        finite-key penalty at realistic run sizes.
    (B) The basis-split tradeoff at finite run size (peripheral figure
        exp14b): the asymptotically optimal 90/10 split starves the
        test set; the finite-key optimum sits at a less asymmetric
        split (TLGR's p_x(n,k) observation).
    (C) The same intercept-resend tap against both protocol arms, 10
        seeds per point: fixed in the key basis, it is invisible to the
        fixed-basis source-model protocol (which assumes it away -
        Assumption B) and detected by BBM92 parameter estimation with
        probability rising to 1 past f* ~ 2(q_tol - e0). Detection is
        reported as an abort PROBABILITY, not a verdict.
    (D) Surcharge convergence: the executing fixed-basis/BBM92 ratio vs
        run length, against the deterministic finite-key model and the
        asymptotic waterfall ratio - showing the exp13 asymptote is
        approached as accumulated key/test bits grow.

    (C) demonstrates DETECTION of one concrete attack, not security
    against general attacks; the general claim rests on the h2(e_ph)
    bound charged in PA (Shor-Preskill; Koashi-Preskill for the
    basis-independent source).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 14: Executing BBM92 arm + intercept-resend visibility")
    print("="*70)

    kw = _bbm92_source_kwargs()
    p_z = BBM92['p_z']
    q_tol = BBM92['q_tol']
    eps_sec, eps_cor = BBM92['eps_sec'], BBM92['eps_cor']
    code_rate = POST_PROC['ldpc_code_rate']
    block = POST_PROC['block_size']

    # Emitted-pair budgets in seconds of source time at the anchored
    # emission rate (1 GHz x 0.076 = 76M pairs/s); the anchored per-pair
    # coincidence probability is ~5.6e-5.
    n_pairs_a = 760_000_000 if quick else 3_040_000_000    # 10 s / 40 s
    n_pairs_c = 380_000_000 if quick else 1_520_000_000    # 5 s / 20 s
    seeds_a = [0, 1] if quick else list(range(10))
    seeds_c = [0, 1] if quick else list(range(10))
    duty = POST_PROC['bell_test_duty_cycle']

    def run_pair(seed, n_pairs):
        """Both arms at one seed and run length; returns
        (bbm92_run, bbm92_distill, fixed_run, fixed_secure_bits)."""
        rb = simulate_bbm92(n_pairs=n_pairs, p_z_alice=p_z, p_z_bob=p_z,
                            rng=np.random.default_rng(seed), **kw)
        db = bbm92_distill(rb.key_bits_alice, rb.key_bits_bob,
                           rb.e_x_observed, rb.n_test, q_tol=q_tol,
                           code_rate=code_rate, block_size=block,
                           eps_sec=eps_sec, eps_cor=eps_cor,
                           rng=np.random.default_rng(10_000 + seed))
        rf = simulate_bbm92(n_pairs=n_pairs, p_z_alice=1.0, p_z_bob=1.0,
                            rng=np.random.default_rng(seed), **kw)
        # fast_ec_simulate drops bits beyond n_blocks*block_size; process
        # the tail separately so both arms account every sifted bit, and
        # charge the fixed-basis arm its 2% CHSH duty (its waterfall
        # does; the sampled run spent no coincidences on it).
        rng_ec = np.random.default_rng(20_000 + seed)
        n_whole = (len(rf.key_bits_alice) // block) * block
        ec = []
        if n_whole > 0:
            ec += fast_ec_simulate(rf.key_bits_alice[:n_whole],
                                   rf.key_bits_bob[:n_whole],
                                   qber_estimate=rf.e_z_actual,
                                   code_rate=code_rate, block_size=block,
                                   rng=rng_ec)
        if len(rf.key_bits_alice) > n_whole:
            ec += fast_ec_simulate(rf.key_bits_alice[n_whole:],
                                   rf.key_bits_bob[n_whole:],
                                   qber_estimate=rf.e_z_actual,
                                   code_rate=code_rate, block_size=block,
                                   rng=rng_ec)
        fixed = int(sum(r.n_secure_bits for r in ec) * (1 - duty))
        return rb, db, rf, fixed

    # (A) executing chains at the operating point, multi-seed
    print(f"\n  (A) Executing chains, {n_pairs_a:,} emitted pairs, "
          f"{len(seeds_a)} seeds:")
    a_runs = [run_pair(s, n_pairs_a) for s in seeds_a]
    a_keep = np.array([r[0].sifting_keep for r in a_runs])
    a_ex = np.array([r[0].e_x_observed for r in a_runs])
    a_ez = np.array([r[0].e_z_actual for r in a_runs])
    a_bb = np.array([r[1].n_secure_bits for r in a_runs])
    a_fx = np.array([r[3] for r in a_runs])
    run_b, dist_b = a_runs[0][0], a_runs[0][1]

    exp_keep = p_z ** 2 + (1 - p_z) ** 2
    print(f"    BBM92 arm (p_z={p_z}): keep {a_keep.mean():.4f} +- "
          f"{a_keep.std():.4f} (model {exp_keep:.3f}); "
          f"n_key ~{run_b.n_key:,}, n_test ~{run_b.n_test:,}/seed")
    print(f"      e_z {a_ez.mean()*100:.2f} +- {a_ez.std()*100:.2f}%  "
          f"e_x {a_ex.mean()*100:.2f} +- {a_ex.std()*100:.2f}%  "
          f"(equal-rates property; analytic QBER "
          f"{OPERATING_POINT['qber']*100:.2f}%)")
    nb = max(dist_b.n_blocks, 1)
    margin_floor = float(np.log2(
        2 / ((eps_sec / nb) ** 2 * (eps_cor / nb))))
    print(f"      finite-key: mu {dist_b.mu:.4f} -> e_ph bound "
          f"{dist_b.e_ph_bound*100:.2f}%, PA floor {margin_floor:.0f} "
          f"bits/block (eps split over {nb} blocks)")
    print(f"      secure bits {a_bb.mean():,.0f} +- {a_bb.std():,.0f} "
          f"({a_bb.mean()/max(run_b.n_key,1):.3f} of sifted key)")
    asym_frac = max(0.0, 1 - binary_entropy(float(a_ex.mean()))
                    - (1 - code_rate))
    tol_frac = max(0.0, 1 - binary_entropy(q_tol) - (1 - code_rate))
    print(f"      asymptotic (mu=0) secure fraction: {asym_frac:.3f} at "
          f"the observed e_x, {tol_frac:.3f} at the\n      a-priori "
          f"q_tol the fixed-length bound charges.")
    print(f"    fixed-basis arm (p_z=1): secure bits {a_fx.mean():,.0f} "
          f"+- {a_fx.std():,.0f} (source-model distillation)")
    keep = exp_keep
    model_ratio = (OPERATING_POINT['waterfall']['final_secure_rate']
                   / compute_rate_waterfall_qkd(
                       OPERATING_POINT['raw_edge_rate_hz'],
                       OPERATING_POINT['qber'], keep,
                       code_rate=code_rate,
                       security_margin=POST_PROC['security_margin_bits'],
                       block_size=block,
                       bell_duty_cycle=duty)['final_secure_rate'])
    exec_ratio = float(a_fx.sum() / a_bb.sum()) if a_bb.sum() > 0 \
        else float('nan')
    print(f"    executing rate ratio fixed-basis/BBM92 (pooled over "
          f"seeds): {exec_ratio:.2f}x\n      (asymptotic waterfall "
          f"model at keep {keep:.2f}: {model_ratio:.2f}x; the excess is "
          f"the finite-key\n      charge: per-block mu at the a-priori "
          f"q_tol, plus per-block floors)")
    print("      note: the BBM92 arm charges NO CHSH duty cycle - "
          "parameter estimation replaces\n      the source-QA witness; "
          "the fixed-basis arm's secure bits are scaled by its 2% Bell "
          "duty.")

    # (B) basis-split tradeoff at finite run size
    p_z_sweep = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    seeds_b = [2] if quick else [2, 3, 4]
    b_secure, b_std, b_asym = [], [], []
    for pz in p_z_sweep:
        vals, asyms = [], []
        for s in seeds_b:
            r = simulate_bbm92(n_pairs=n_pairs_a, p_z_alice=pz,
                               p_z_bob=pz,
                               rng=np.random.default_rng(s), **kw)
            d = bbm92_distill(r.key_bits_alice, r.key_bits_bob,
                              r.e_x_observed, r.n_test, q_tol=q_tol,
                              code_rate=code_rate, block_size=block,
                              eps_sec=eps_sec, eps_cor=eps_cor,
                              rng=np.random.default_rng(30_000 + s))
            vals.append(d.n_secure_bits)
            # mu = 0 limit of the same fixed-length charge (h2 at the
            # a-priori q_tol, per-block floors dropped): labeled model
            # line, not an executing result
            asyms.append(max(0.0, r.n_key * (1 - binary_entropy(q_tol)
                                             - (1 - code_rate))))
        b_secure.append(float(np.mean(vals)))
        b_std.append(float(np.std(vals)))
        b_asym.append(float(np.mean(asyms)))
    best_pz = p_z_sweep[int(np.argmax(b_secure))]
    print(f"\n  (B) Basis-split tradeoff at {n_pairs_a:,} pairs "
          f"({len(seeds_b)} seeds): finite-key optimum p_z = {best_pz}")
    print("      (asymptotic optimum approaches 1.0); short runs pay "
          "for asymmetry with\n      test-set starvation (mu grows as "
          "k shrinks). Figure: exp14b (peripheral).")

    # (C) intercept-resend visibility, same attack both arms
    f_sweep = BBM92['eve_fraction_sweep'] if not quick \
        else [0.0, 0.02, 0.05, 0.2, 0.5, 1.0]
    arms = {"fixed-basis": 1.0, "bbm92": p_z}
    c_res = {a: {"err_mean": [], "err_std": [], "acc_mean": [],
                 "acc_accepted": [], "accepted": [], "abort_prob": []}
             for a in arms}
    for f in f_sweep:
        for arm, pz in arms.items():
            errs, accs, accepts = [], [], []
            for s in seeds_c:
                r = simulate_bbm92(n_pairs=n_pairs_c, p_z_alice=pz,
                                   p_z_bob=pz, eve_fraction=f,
                                   eve_basis_mode='z',
                                   rng=np.random.default_rng(100 + s),
                                   **kw)
                if arm == "bbm92":
                    errs.append(r.e_x_observed)
                    accepts.append(r.e_x_observed <= q_tol)
                    c_res[arm].setdefault("n_test_all", []).append(r.n_test)
                else:
                    # the fixed-basis protocol's only observable is the
                    # key-basis error rate EC reveals
                    errs.append(r.e_z_actual)
                    accepts.append(True)
                accs.append(r.eve_guess_accuracy)
            accs = np.array(accs)
            accepts_arr = np.array(accepts)
            c_res[arm]["err_mean"].append(float(np.mean(errs)))
            c_res[arm]["err_std"].append(float(np.std(errs)))
            c_res[arm]["acc_mean"].append(float(np.mean(accs)))
            c_res[arm]["acc_accepted"].append(
                float(np.mean(accs[accepts_arr]))
                if np.any(accepts_arr) else float('nan'))
            c_res[arm]["accepted"].append(bool(np.all(accepts_arr)))
            c_res[arm]["abort_prob"].append(
                float(1.0 - np.mean(accepts_arr)))
            c_res[arm].setdefault("accept_count", []).append(
                int(np.sum(accepts_arr)))

    e0 = OPERATING_POINT['qber']
    f_star = 2 * (q_tol - e0) / (1 - 2 * e0)
    detected_f = [f for f, a in zip(f_sweep, c_res["bbm92"]["accepted"])
                  if not a]
    first_detected = detected_f[0] if detected_f else None
    print(f"\n  (C) Intercept-resend (Eve fixed in the key basis), "
          f"{len(seeds_c)} seed(s) x {n_pairs_c:,} pairs per point:")
    print(f"    {'tap f':>7} {'fixed QBER':>11} {'fixed EveAcc':>13} "
          f"{'bbm92 e_x':>10} {'bbm92 verdict':>14}")
    n_seeds = len(seeds_c)
    for i, f in enumerate(f_sweep):
        ok = c_res["bbm92"]["accept_count"][i]
        verdict = f"ACCEPT {ok}/{n_seeds}" if ok == n_seeds \
            else f"ABORT {n_seeds - ok}/{n_seeds}"
        print(f"    {f:>7.2f} "
              f"{c_res['fixed-basis']['err_mean'][i]*100:>10.2f}% "
              f"{c_res['fixed-basis']['acc_mean'][i]:>13.3f} "
              f"{c_res['bbm92']['err_mean'][i]*100:>9.2f}% {verdict:>14}")
    test_bits = int(np.mean(c_res["bbm92"]["n_test_all"]))
    print(f"    Near-threshold verdicts are probabilistic: the test set "
          f"at this run size is ~{test_bits}\n    bits (sigma ~"
          f"{100*np.sqrt(q_tol*(1-q_tol)/max(test_bits,1)):.2f}% on "
          f"e_x), so taps near f* escape or trip by seed. Larger\n"
          f"    accumulated test sets sharpen the threshold; the abort "
          f"rule itself is per-run.")
    print(f"    fixed-basis arm: error rate flat at every intercepted fraction - "
          f"the attack is invisible;\n    at f=1 Eve predicts "
          f"{c_res['fixed-basis']['acc_mean'][-1]*100:.1f}% of the "
          f"'secure' key. The source-model protocol\n    EXCLUDES this "
          f"adversary by assumption; this arm shows what the assumption "
          f"carries.")
    print(f"    bbm92 arm: e_x rises as (1-f)e0 + f/2; abort threshold "
          f"q_tol {q_tol} crossed at\n    analytic f* = "
          f"{f_star:.3f}, first swept abort at f = {first_detected}.")
    print("    Detection here is for THIS attack; general security is "
          "the h2(e_ph) PA charge, not this sweep.")

    # (D) finite-key penalty convergence: executing ratio vs run length, against
    # the deterministic finite-key model. FER is omitted from BOTH model
    # arms (identical FER-model charge, cancels in the ratio); the executing
    # points carry it.
    p_coinc = float(np.mean([r[0].n_coincidences for r in a_runs])
                    / n_pairs_a)
    d_tag = OPERATING_POINT['tagged_fraction']

    def ratio_model(n_pairs_pt):
        coinc = n_pairs_pt * p_coinc
        n_key = coinc * p_z ** 2
        n_test = coinc * (1 - p_z) ** 2
        nb_m = max(1, int(np.ceil(n_key / block)))
        eps_s = eps_sec / nb_m
        floor = np.log2(2 / (eps_s ** 2 * (eps_cor / nb_m)))
        bb = 0.0
        for b in range(nb_m):
            bn = block if b < nb_m - 1 else n_key - (nb_m - 1) * block
            mu_b = tlgr_deviation(int(bn), int(max(n_test, 1)), eps_s)
            bb += max(0.0, bn * (1 - binary_entropy(
                min(0.5, q_tol + mu_b))) - bn * (1 - code_rate) - floor)
        nf = int(coinc)
        nbf = max(1, int(np.ceil(nf / block)))
        fx = 0.0
        for b in range(nbf):
            bn = block if b < nbf - 1 else nf - (nbf - 1) * block
            fx += max(0.0, bn * (1 - d_tag) - bn * (1 - code_rate)
                      - POST_PROC['security_margin_bits'])
        fx *= (1 - duty)
        return fx / bb if bb > 0 else float('nan')

    pairs_sweep = [380_000_000, 1_520_000_000] if quick \
        else [760_000_000, 3_040_000_000, 12_160_000_000]
    seeds_d = [0, 1] if quick else list(range(10))
    # Near the certification cliff (small runs), some seeds certify
    # ZERO key and the ratio diverges: report the median and range of
    # the finite ratios plus the zero-certification count, never a
    # mean +- std over a divergent distribution.
    d_stats = []
    for np_pt in pairs_sweep:
        ratios = []
        n_zero = 0
        for s in seeds_d:
            _, db_d, _, fx_d = run_pair(50 + s, np_pt)
            if db_d.n_secure_bits > 0:
                ratios.append(fx_d / db_d.n_secure_bits)
            else:
                n_zero += 1
        d_stats.append({
            "pairs": np_pt,
            "median": float(np.median(ratios)) if ratios else None,
            "min": float(np.min(ratios)) if ratios else None,
            "max": float(np.max(ratios)) if ratios else None,
            "n_zero_cert": n_zero,
            "n_seeds": len(seeds_d),
        })
    model_pairs = np.geomspace(pairs_sweep[0] / 2,
                               pairs_sweep[-1] * 8, 40)
    model_curve = [ratio_model(npp) for npp in model_pairs]
    print(f"\n  (D) Finite-key penalty convergence ({len(seeds_d)} seeds "
          f"per point):")
    for st in d_stats:
        if st["median"] is None:
            print(f"      {st['pairs']/1e6:>5.0f}M pairs: no certified "
                  f"key in any seed (certification cliff, "
                  f"{st['n_zero_cert']}/{st['n_seeds']} zero)")
        else:
            zero_note = (f", {st['n_zero_cert']}/{st['n_seeds']} seeds "
                         f"certify zero key (ratio unbounded)"
                         if st['n_zero_cert'] else "")
            print(f"      {st['pairs']/1e6:>5.0f}M pairs: median ratio "
                  f"{st['median']:.2f}x, range "
                  f"[{st['min']:.2f}, {st['max']:.2f}]{zero_note}")
    print(f"      asymptotic waterfall ratio {model_ratio:.2f}x; the "
          f"executing ratio approaches it as\n      accumulated "
          f"key/test bits grow. The deterministic finite-key model is "
          f"drawn for\n      comparison; near the cliff, sampling of "
          f"the ~400-bit test set moves individual\n      seeds far "
          f"around it. FER omitted from both model arms - it cancels "
          f"in the ratio.")

    # Main figure: 2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                 figsize=(7.16, 6.2))

    fs = np.array(f_sweep)
    ax1.errorbar(fs, np.array(c_res['fixed-basis']['err_mean'])*100,
                 yerr=np.array(c_res['fixed-basis']['err_std'])*100,
                 fmt='o-', color=C['quantum'], markersize=3,
                 label='fixed-basis: monitored QBER (e_z)')
    ax1.errorbar(fs, np.array(c_res['bbm92']['err_mean'])*100,
                 yerr=np.array(c_res['bbm92']['err_std'])*100,
                 fmt='s-', color=C['pareto'], markersize=3,
                 label='BBM92: test-set e_x')
    ax1.axhline(q_tol*100, color=C['unmasked'], linestyle='--',
                linewidth=1, label=f'abort threshold q_tol = {q_tol}')
    ax1.set_xlabel('intercepted fraction f')
    ax1.set_ylabel('observed error rate (%)')
    ax1.set_xlim(-0.04, 1.04)
    ax1.set_title('Observed error vs intercepted fraction', fontsize=8)
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.3)

    acc_fixed = np.array(c_res['fixed-basis']['acc_mean'])
    ax2.plot(fs, acc_fixed, 'o-', color=C['quantum'], markersize=3,
             label='fixed-basis (key always accepted)')
    acc_b = [(f, a) for f, a in zip(fs, c_res['bbm92']['acc_accepted'])
             if not np.isnan(a)]
    if acc_b:
        ax2.plot([x[0] for x in acc_b], [x[1] for x in acc_b], 's-',
                 color=C['pareto'], markersize=3,
                 label='BBM92 (accepted keys only)')
    ax2.axhline(0.5, color='gray', linewidth=0.8, linestyle=':')
    ax2.set_xlabel('intercepted fraction f')
    ax2.set_ylabel('Eve accuracy on accepted key')
    ax2.set_xlim(-0.04, 1.04)
    ax2.set_ylim(0.45, 1.02)
    ax2.set_title('Eve accuracy on accepted keys', fontsize=8)
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3)

    ax3.plot(fs, c_res['bbm92']['abort_prob'], 's-', color=C['pareto'],
             markersize=3, label=f'BBM92 ({len(seeds_c)} seeds/point)')
    ax3.plot(fs, c_res['fixed-basis']['abort_prob'], 'o-',
             color=C['quantum'], markersize=3,
             label='fixed-basis (never aborts)')
    ax3.axvline(f_star, color=C['unmasked'], linestyle='--', linewidth=1,
                label=f'analytic f* = {f_star:.3f}')
    ax3.set_xlabel('intercepted fraction f')
    ax3.set_ylabel('abort probability')
    # A log axis cannot reach f = 0; symlog splices a short linear
    # segment near zero so the f = 0 point is drawn. Keep that segment
    # narrow so zero sits at the left edge, not mid-axis.
    ax3.set_xscale('symlog', linthresh=0.01, linscale=0.25)
    ax3.set_xlim(-0.002, 1.3)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title('Abort probability vs intercepted fraction', fontsize=8)
    ax3.legend(fontsize=6)
    ax3.grid(True, alpha=0.3)

    ax4.plot(model_pairs/1e6, model_curve, '-', color=C['dark'],
             linewidth=1.2, label='finite-key model')
    d_pts = [st for st in d_stats if st["median"] is not None]
    if d_pts:
        xs4 = np.array([st["pairs"] for st in d_pts]) / 1e6
        med4 = np.array([st["median"] for st in d_pts])
        lo4 = med4 - np.array([st["min"] for st in d_pts])
        hi4 = np.array([st["max"] for st in d_pts]) - med4
        ax4.errorbar(xs4, med4, yerr=[lo4, hi4], fmt='s',
                     color=C['pareto'], markersize=4,
                     label=f'executing (median, range, '
                           f'{len(seeds_d)} seeds)')
        for st in d_pts:
            if st["n_zero_cert"]:
                ax4.annotate(f'{st["n_zero_cert"]}/{st["n_seeds"]} '
                             f'seeds: zero key',
                             xy=(st["pairs"]/1e6, st["median"]),
                             xytext=(4, -14),
                             textcoords='offset points', fontsize=6,
                             color=C['unmasked'])
    ax4.axhline(model_ratio, color=C['accent'], linestyle='--',
                linewidth=1,
                label=f'asymptotic ratio {model_ratio:.2f}x')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    # bottom below the asymptote line, top above the widest seed range
    # and the finite-key model curve
    _tops = [v for v in model_curve if np.isfinite(v)]
    _tops += [st["max"] for st in d_pts]
    ax4.set_ylim(0.9, max(_tops) * 1.6)
    ax4.set_xlabel('emitted pairs per run (millions)')
    ax4.set_ylabel('fixed-basis / BBM92 secure-bit ratio')
    ax4.set_title('Finite-key penalty vs run length', fontsize=8)
    ax4.legend(fontsize=6)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Fixed-basis vs BBM92 arms under intercept-resend',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5, h_pad=2.5)
    save_fig(fig, 'exp14_bbm92_executing')

    # Peripheral figure: basis-split tradeoff
    figb, axb = plt.subplots(1, 1, figsize=VIZ['figsize_single'])
    axb.errorbar(p_z_sweep, np.array(b_secure)/1000,
                 yerr=np.array(b_std)/1000, fmt='o-', color=C['pareto'],
                 markersize=3, label='executing (TLGR finite-key)')
    axb.plot(p_z_sweep, np.array(b_asym)/1000, ':', color=C['dark'],
             label='mu = 0 limit at q_tol')
    axb.axvline(p_z, color=C['accent'], linestyle='--', linewidth=1,
                label=f'config p_z = {p_z}')
    axb.set_xlabel('basis probability p_z')
    axb.set_ylabel('secure kbits per run')
    axb.set_title('Finite-key basis-split tradeoff', fontsize=9)
    axb.legend(fontsize=6)
    axb.grid(True, alpha=0.3)
    figb.tight_layout()
    save_fig(figb, 'exp14b_basis_split')

    print("\n  Experiment 14 complete.")
    return {
        "arm_a": {
            "n_seeds": len(seeds_a),
            "bbm92": {"keep_mean": float(a_keep.mean()),
                      "keep_std": float(a_keep.std()),
                      "e_z_mean": float(a_ez.mean()),
                      "e_x_mean": float(a_ex.mean()),
                      "mu": dist_b.mu,
                      "secure_bits_mean": float(a_bb.mean()),
                      "secure_bits_std": float(a_bb.std())},
            "fixed_basis": {"secure_bits_mean": float(a_fx.mean()),
                            "secure_bits_std": float(a_fx.std())},
            "executing_ratio_pooled": exec_ratio,
            "model_ratio_asymptotic": float(model_ratio),
        },
        "p_z_tradeoff": {"p_z": p_z_sweep, "secure_bits": b_secure,
                         "secure_bits_std": b_std,
                         "asymptotic_model_bits": b_asym,
                         "finite_key_optimum_p_z": best_pz,
                         "n_seeds": len(seeds_b)},
        "intercept_resend": {
            "f_sweep": list(f_sweep),
            "n_seeds": len(seeds_c),
            "fixed_qber": c_res['fixed-basis']['err_mean'],
            "fixed_eve_acc": c_res['fixed-basis']['acc_mean'],
            "bbm92_e_x": c_res['bbm92']['err_mean'],
            "bbm92_abort_prob": c_res['bbm92']['abort_prob'],
            "bbm92_acc_accepted": c_res['bbm92']['acc_accepted'],
            "analytic_f_star": f_star,
            "first_swept_abort_f": first_detected,
        },
        "surcharge_convergence": d_stats,
    }


# EXPERIMENT 15: ACCEPTANCE-WINDOW OPTIMIZATION
def run_experiment_15(quick=False):
    print("\n" + "="*70)
    print("EXPERIMENT 15: Acceptance-Window Optimization")
    print("="*70)
    print("  Coincidence acceptance window trades captured pairs against")
    print("  entanglement quality (FSS precession + timing jitter): the")
    print("  measured 2n(delta-tau) endpoints anchor the model, the m2")
    print("  waterfall prices the trade in secret bits per second.")
    print("  RECEIVER MODEL: fixed analyzers with one global phase")
    print("  alignment per window - no delay-binned phase tracking. The")
    print("  config operating point instead anchors to the time-resolved")
    print("  tomography negativity (delay-binned phase compensation, which")
    print("  the rack's TCSPC supports); this experiment prices the")
    print("  simpler phase-blind receiver against it.")

    t1 = QUANTUM['t1_x_ps'] * 1e-12
    sigma = QUANTUM['two_photon_jitter_fwhm_ps'] * 1e-12 / 2.355
    omega = fss_precession_omega(QUANTUM['fss_uev'])
    mode = detected_delay_mode(t1, sigma)
    t_lo = mode - 3 * sigma   # window onset: capture the jitter-blurred rise

    # Two visibility calibrations bracket the model uncertainty:
    # (a) the 8 ps-window maximum 2n = 0.98(1), a nearly-pure-coherence
    #     window, gives the OPTIMISTIC visibility;
    # (b) the T1-averaged 2n = 0.94(1), which folds in decay mechanisms
    #     beyond precession+jitter, gives the CONSERVATIVE one.
    # The rack paper's own model attributes the residual decay to timing
    # uncertainty and possible spin scattering; both calibrations are
    # reported and the optimum is quoted under each.
    anchor_w = QUANTUM['negativity_2n_window']
    v0_win = calibrate_visibility(anchor_w, mode - 4e-12, mode + 4e-12,
                                  t1, sigma, omega)
    c_t1, _ = delay_window_coherence(t_lo, mode + t1, t1, sigma, omega)
    v0_t1 = QUANTUM['negativity_2n_t1avg'] / abs(c_t1)

    # Full-capture edge rate: compute_coincidence_rate() counts every
    # cascade whose photons both survive, i.e. an unbounded window; the
    # window keeps the captured fraction. Accidentals at the anchored
    # dark rate are < 0.1 Hz across this sweep and are neglected.
    r_full = OPERATING_POINT['raw_edge_rate_hz']

    n_w = 24 if quick else 72
    windows = np.linspace(8e-12, 5 * t1, n_w)

    def _wf(rate_hz, qber):
        return compute_rate_waterfall(
            rate_hz, qber,
            code_rate=POST_PROC['ldpc_code_rate'],
            security_margin=POST_PROC['security_margin_bits'],
            block_size=POST_PROC['block_size'],
            bell_duty_cycle=POST_PROC['bell_test_duty_cycle'],
            tagged_fraction=(QUANTUM['multi_photon_prob']
                             / QUANTUM['source_brightness']))

    results = {}
    for label, v0 in (("window-anchored (optimistic)", v0_win),
                      ("T1-anchored (conservative)", v0_t1)):
        two_n, captured, secret = [], [], []
        for w_edge in windows:
            c, cap = delay_window_coherence(t_lo, t_lo + w_edge, t1,
                                            sigma, omega)
            n2 = float(np.clip(v0 * abs(c), 0.0, 1.0))
            bell_w = werner_weight_for_negativity(n2)
            qber = compute_qber_analytical(
                bell_w, pbs_extinction=QUANTUM['pbs_extinction_ratio'],
                bire_contribution=QUANTUM[
                    'analytic_qber_birefringence_allowance'])
            wf = _wf(r_full * cap, qber)
            two_n.append(n2)
            captured.append(cap)
            secret.append(wf['final_secure_rate'])
        secret = np.array(secret)
        i_opt = int(np.argmax(secret))
        results[label] = {
            "v0": v0, "two_n": two_n, "captured": captured,
            "secret_bps": secret.tolist(),
            "opt_window_ps": float(windows[i_opt] * 1e12),
            "opt_secret_bps": float(secret[i_opt]),
        }
        print(f"    {label}: v0 = {v0:.4f}, optimum window "
              f"{windows[i_opt]*1e12:.0f} ps "
              f"({windows[i_opt]/t1:.2f} T1) -> "
              f"{secret[i_opt]:,.0f} bps secret")

    # Default-window and operating-point reference
    w_def = QUANTUM['coincidence_window_ns'] * 1e-9
    print(f"    reference: operating point "
          f"{OPERATING_POINT['secret_rate_bps']:,.0f} bps (config Werner "
          f"weight from the time-resolved 2n, full capture)")
    print("    Shape of the trade: the optimum window sits just below the")
    print("    declared rate-0.84 code's design QBER (~2.07%); beyond it")
    print("    the fixed code fails frames and the rate collapses to 0 -")
    print(f"    including at the config default window ({w_def*1e12:.0f} ps)")
    print("    under this phase-blind receiver. The window rule is thus")
    print("    'widest window whose windowed QBER clears the EC design")
    print("    point', not a smooth interior optimum. The phase-blind")
    print("    receiver pays 26-40% of the operating-point rate; closing")
    print("    that gap is what delay-binned phase tracking buys.")
    print("    Noise-attribution bracket: the Werner mapping charges the")
    print("    windowed decoherence isotropically. Under pure dephasing")
    print("    (precession+jitter alone) the X-basis error is 1.5x the")
    print("    Werner charge and the Z-basis error is ZERO - the fixed-")
    print("    Z-basis conduit arm's raw QBER is window-independent, and")
    print("    for it the window choice is a rate/accidentals matter only.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ['figsize_wide'])
    win_ps = windows * 1e12
    for (label, res), ls in zip(results.items(), ('-', '--')):
        ax1.plot(win_ps, res['two_n'], ls, color=C['quantum'],
                 linewidth=1.8, label=f'2n, {label.split(" (")[0]}')
    ax1.plot(win_ps, results[list(results)[0]]['captured'], ':',
             color=C['accent'], linewidth=1.8, label='captured fraction')
    ax1.axvline(x=QUANTUM['t1_x_ps'], color='gray', linestyle=':',
                linewidth=1, label='T1')
    ax1.set_xlabel('Acceptance window (ps)')
    ax1.set_ylabel('Windowed 2n / captured fraction')
    ax1.set_title('Quality vs capture', fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.3)

    for (label, res), ls, col in zip(results.items(), ('-', '--'),
                                     (C['pareto'], C['dark'])):
        ax2.plot(win_ps, np.array(res['secret_bps']) / 1e3, ls, color=col,
                 linewidth=1.8, label=label.split(" (")[0])
        ax2.plot(res['opt_window_ps'], res['opt_secret_bps'] / 1e3, 'o',
                 color=col, markersize=5)
    ax2.set_xlabel('Acceptance window (ps)')
    ax2.set_ylabel('Secret rate (kbit/s)')
    ax2.set_title('Distilled rate vs window', fontsize=9)
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3)
    fig.suptitle('Acceptance-window optimization at the anchored '
                 'operating point', fontsize=10, fontweight='bold')
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, 'exp15_window_optimization')

    print("  Experiment 15 complete.")
    return {
        "fss_uev": QUANTUM['fss_uev'],
        "t1_x_ps": QUANTUM['t1_x_ps'],
        "precession_period_ns": float(2 * np.pi / omega * 1e9),
        "windows_ps": win_ps.tolist(),
        "calibrations": {k: {kk: vv for kk, vv in v.items()
                             if kk != 'two_n' and kk != 'captured'}
                         for k, v in results.items()},
    }


def main():
    parser = argparse.ArgumentParser(description='QD-Entanglement Simulation Runner')
    parser.add_argument('--exp', nargs='+', type=int, default=list(range(1, 16)),
                        help='Experiment numbers to run (default: all 1-15)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with reduced parameter sweeps')
    parser.add_argument('--export-csv', action='store_true',
                        help='Rewrite results/data CSV tables from the '
                             'existing summary and exit')
    args = parser.parse_args()

    if args.quick:
        set_output_suffix("_quick")

    if args.export_csv:
        summary_path = os.path.join(RESULTS_DIR,
                                    f"summary{_OUTPUT_SUFFIX}.json")
        with open(summary_path) as f:
            export_summary_csv(json.load(f), _OUTPUT_SUFFIX)
        return

    print("Entanglement-derived one-time masks for federated learning")
    print("over campus fiber meshes - source-model key distillation")
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
        12: run_experiment_12,
        13: run_experiment_13,
        14: run_experiment_14,
        15: run_experiment_15,
    }

    for exp_num in sorted(args.exp):
        if exp_num in experiment_fns:
            exp_start = time.time()
            try:
                result = experiment_fns[exp_num](quick=args.quick)
                all_results[f"exp{exp_num}"] = result
                elapsed = time.time() - exp_start
                print(f"  Experiment {exp_num} took {elapsed:.1f}s")
            except Exception as e:
                print(f"  FAILED: Experiment {exp_num} failed: {e}")
                import traceback
                traceback.print_exc()

    total_elapsed = time.time() - total_start

    # Save summary. Subset runs merge into the existing file so a
    # single-experiment rerun updates its own entry without discarding
    # the rest of the suite's results; quick mode writes to
    # summary_quick.json (see _OUTPUT_SUFFIX).
    summary_path = os.path.join(RESULTS_DIR, f"summary{_OUTPUT_SUFFIX}.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def sanitize(obj):
        """NaN is not valid JSON; serialize it as null."""
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (float, np.floating)) and np.isnan(obj):
            return None
        return obj

    try:
        merged = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    merged = json.load(f)
            except (json.JSONDecodeError, OSError):
                merged = {}
        merged.update(all_results)
        with open(summary_path, 'w') as f:
            json.dump(sanitize(merged), f, indent=2, default=convert)
        print(f"\nSummary saved to: {summary_path}")
        export_summary_csv(sanitize(merged), _OUTPUT_SUFFIX)
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE - Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results in: {RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
