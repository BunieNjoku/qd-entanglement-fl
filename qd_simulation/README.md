# Entanglement-Based Shared Randomness for Federated Learning

Simulation package for an entanglement-keyed secure-aggregation system:
quantum-dot Bell pairs over a campus-scale fiber mesh provision the
pairwise one-time masks of two-level Bonawitz secure aggregation in
federated learning.

## Hardware anchoring

Source and detection parameters are anchored to one real deployed
device: the rack-based GaAs quantum dot entangled photon pair source
of the QD-CamNetz collaboration (Zena et al., arXiv:2604.02024) - 1 GHz
clocked two-photon excitation, in-situ fiber-coupled microlens QD,
SSPD detection (85%, 20 ps RMS), native 780 nm emission, measured
entanglement negativity 2n = 0.960(4) averaged over six hours of
hands-off operation. Quantities the rack paper does not report come
from sister-device measurements by the same group with the same
excitation protocol: per-pulse excitation efficiency 0.076(9)
(Hopfmann et al., PRB 103, 075413) and g2_XX(0) = 8.0(6)e-4 (Langer
et al., APL Photonics 10, 066117). The per-arm transmission is
calibrated to the measured 697(8) kHz singles rate and cross-checks
against the sister device's independently measured 1.1(1)% within 20%.
The two-node fiber link, the protocol layer, and key distillation
remain modeled: derived rates are simulation-grade, not hardware
measurements. Provenance (measured, derived, assumption) is noted per
value in `config.py`.

## Framing

The simulation executes the same physical layer under two protocol
framings; BBM92 is the default:

- BBM92 arm (system default): efficient asymmetric sifting (90/10
  basis split, Lo-Chau-Ardehali), X-basis parameter estimation, TLGR
  fixed-length finite-key privacy amplification. Tolerates an active
  fiber adversary; source trust is discharged by the
  basis-independent-source argument. This is entanglement-based QKD.
- Trusted-conduit mode (fixed basis): source-model secret-key
  agreement (Maurer 1993; Ahlswede-Csiszar 1993) with a trusted
  co-located source (CHSH verifies the state and does not bound an
  eavesdropper), passive noisy fiber, and fixed-basis measurement. It
  yields 1.41x the BBM92 rate at the operating point and is blind to an
  in-fiber intercept-resend adversary (exp14), so it is an option for
  physically secured fiber rather than the default.

Both framings assume trusted endpoints (clients trust each other and
their local apparatus) and draw fresh per-round shared bits from a
pooled key buffer filled by batched distillation.

## Core construction

- Two endpoints share Bell pairs from a QD source over native 780 nm
  fiber, without frequency conversion.
- Coincidence detection, LDPC reconciliation and Toeplitz privacy
  amplification yield shared bit strings.
- Triangles (3-node clusters with 3 pairwise links) are the smallest
  unit covering all pairs needed for 3-client Bonawitz masking.
- Two-mask design per Bonawitz 2017: pairwise mask from shared bits,
  self-mask from local randomness with Shamir shares across the cluster.
- Modular one-time-pad masking on int32-quantized gradients.
- Hierarchical aggregation: intra-cluster Bonawitz, then inter-cluster
  Bonawitz over a tree derived from the Delaunay mesh dual.
- Dropout recovery: survivors reveal pairwise bits with dropped clients
  to the aggregator, safe because bits are single-use per round.

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py             # Full run, all experiments
python run_all.py --quick     # Reduced sweeps, ~1.5 min
python run_all.py --exp 1 2 3 # Specific experiments
```

## Experiments

| # | Name | What It Demonstrates |
|---|------|---------------------|
| 1 | Source characterization | QBER, CHSH S, coincidence rates vs source/fiber parameters |
| 2 | Post-processing | Shared-bit rate budget closes; reconciliation efficiency |
| 3 | Mesh scaling | Delaunay topology, triangle coverage, aggregation depth |
| 4 | FL training correctness | Modular masking does not degrade convergence vs unmasked baseline |
| 5 | Dropout recovery | entangled-key two-mask aggregation tolerates client dropout at per-round freshness; recovery bytes drawn against the Bonawitz and Flamingo seed-reveal models on the same dropout events (O(d) vs O(1)) |
| 6 | Throughput | Round-rate ceiling under shared-bit-rate constraints |
| 7 | QBER to key-rate / capacity envelope | QBER sets the key supply (rate, max maskable model size) while the PA mask entropy per bit stays near 1 across the sweep |
| 8 | Multi-triangle FL | Two-level hierarchical Bonawitz at 3/6/9/12 client scales |
| 9 | HNDL transcript attack | Record everything, break all computational secrets later: classical arm 100% retro-recovered, IT-dealer online provisioning 100%, IT-dealer courier provisioning 0%, entangled arm 0%; immunity tracks provisioning |
| 10 | Availability under source failure | Kill one node's source: edges continue at ~half rate from far endpoints (availability/fairness, not security) |
| 11 | Gradient inversion (real construction) | Tabular DLG/iDLG: unmasked leaks (100% attr-infer), modular mask blocks (theorem), plaintext aggregate partially leaks (motivates DP-on-top) |
| 12 | Baseline comparison | Five executing arms: Bonawitz-DH, Flamingo-PRF, and a classical IT-dealer (online and courier provisioning). Every OTP arm pays O(d) recovery; the arms differ in what must be trusted: DH/PRF hardness, a dealer, or one's own source |
| 13 | QKD-framing rate cost | Same physics, two post-processings: source-model vs Shor-Preskill QKD framing. Ratio 2.32x/1.41x/1.16x by sifting protocol, pricing the trusted-source assumption; not a security ranking. The ratio is the asymptote in exp14(d) |
| 14 | Executing BBM92 arm | The QKD framing executed on sampled bits (basis choice, sifting, TLGR finite-key PA): the same key-basis intercept-resend tap is invisible to the fixed-basis protocol (flat QBER, Eve predicts 99.9% of accepted key at full tap) and aborts BBM92 near the analytic threshold f* = 0.010. The anchored QBER (1.53%) sits close under q_tol = 0.02, so a nonzero false-abort baseline appears at small test sets. Finite-key penalty: executing ratio 4.0x at 40 s runs, flooring near 1.8x under the implemented accounting (per-block mu at the fixed block size); exp13's 1.41x infinite-key waterfall ratio is a different-accounting reference the protocol does not reach |
| 15 | Acceptance-window optimization | Coincidence window trades captured pairs against FSS-precession decoherence for a phase-blind receiver; optimum (177-279 ps, 1.1-1.7 T1, by calibration) tracks the EC design-QBER boundary and pays 22-41% vs the time-resolved operating point |

## Assumptions

- A, bufferless FPGA RTL: treated as a design target, not simulated.
- B, trusted source: the QD is at an owned node, not adversary
  controlled. Binds the trusted-conduit mode only; the BBM92 default
  discharges it (basis-independent-source argument, Koashi-Preskill
  2003).
- C, FPGA latency: published LDPC benchmarks with a 2x safety margin.
- D, fiber QBER: anchored operating QBER 1.53% (Werner mapping of the
  measured 2n = 0.960 plus extinction and birefringence allowances);
  parametric sweeps over 1-6% (exp2) and 0.5-6% (exp7).
- E, sister-device values: excitation efficiency and g2 are measured
  on the same group's predecessor devices, not on the rack unit; the
  derived coincidence rate scales as 1/eta_ex at the fixed measured
  singles rate.

## Output

Results land in `results/`:

- `figures/`: one PNG per figure. Quick mode (`--quick`) writes
  `*_quick`-suffixed files so reduced-sweep runs do not overwrite
  full-mode figures.
- `summary.json`: numeric results. Subset runs (`--exp N`) merge into
  the existing summary.

## Tests

Plain-python asserts, no test framework; each file runs standalone:

```bash
python tests/test_noise_models.py
```

- `test_anchored_source.py`: anchored source parameters and the FSS
  precession window model (m1)
- `test_noise_models.py`: exp1a noise models (m1) and the
  finite-blocklength FER model (m2)
- `test_qkd_waterfall.py`: exp13 QKD-framed rate waterfall (m2)
- `test_bbm92.py`: exp14 executing BBM92 arm (m1, m2)
- `test_cell_sum_privacy.py`: inter-cell masking in
  run_multitriangle_fl (m4)
- `test_tree_mask_keying.py`: inter-triangle tree-mask key
  assignment (m4)
- `test_mesh_edge_load.py`: per-edge load bounds of the mesh
  scaling model (m3, m4)
- `test_it_dealer.py`: IT-dealer baseline arm (m7, m6)

## Hardware Requirements

CPU, 16 GB RAM, ~2 GB disk. No GPU required.

## Project Structure

```
qd_simulation/
|-- config.py                  # System parameters
|-- run_all.py                 # Runner
|-- modules/
|   |-- m1_quantum_source.py   # Bell-pair source + detection
|   |-- m2_post_processing.py  # Reconciliation + privacy amplification
|   |-- m3_mesh_topology.py    # Delaunay mesh + aggregation tree
|   |-- m4_federated_learning.py # Two-mask Bonawitz + dropout recovery
|   |-- m6_hndl.py             # HNDL transcript-replay arms (exp9)
|   `-- m7_baselines.py        # Classical baseline arms (exp12)
`-- results/
```

## Terminology

The quantum layer runs one of two framings over the same physics. The
default is BBM92 (Bennett-Brassard-Mermin 1992) with efficient
asymmetric sifting (Lo-Chau-Ardehali 2005), X-basis parameter
estimation under squashing (Beaudry-Moroder-Lutkenhaus 2008), and
TLGR fixed-length finite-key privacy amplification (Tomamichel-Lim-
Gisin-Renner 2012); its PA charge supports the harvest-now-decrypt-later
immunity claim in that mode. The trusted-conduit mode is secret-key
agreement in the source model (Maurer 1993; Ahlswede & Csiszar 1993):
sifting-free fixed-basis measurement whose PA removes a GLLP tagged
fraction (multi-pair emissions; Gottesman-Lo-Lutkenhaus-Preskill 2004)
instead. Both use LDPC information reconciliation and Toeplitz privacy
amplification; the Toeplitz step alone is randomness extraction under
the Leftover Hash Lemma (Renner & Konig 2005). The CHSH check is an
entanglement witness (Horodecki 1996; Terhal 2000) used as source QA in
the conduit mode, not device independence. Masks are one-time pads over
Z_2^32 in the Bonawitz two-mask construction (CCS 2017), consumed per
round.

Mode names: `entangled`, `gaussian-legacy` (comparison only, no privacy
proof), `prng`, `unmasked`.

## Related work (positioning)

- Bonawitz et al., CCS 2017: the two-mask secure-aggregation core used
  here (per-execution ephemeral DH seeds; Shamir recovery on both legs).
- Flamingo (Ma et al., IEEE S&P 2023): classical reveal-based per-round
  dropout recovery. Same recovery mechanism as here; the difference is
  that per-round freshness here is unconditional (no PRF/DH assumption,
  no reliance on key erasure or forward secrecy).
- QuNetQFL (arXiv:2501.12709): QKD-derived one-time-pad masks for FL,
  validated on a real quantum network with fully connected pairwise
  MDI-QKD among clients and an aggregation-only server holding no key
  material. Measured 32.8-240 kbit/s pairwise key rates (above our
  BBM92-framed 2.5 kbit/s and conduit-mode 3.5 kbit/s; theirs is
  hardware, ours a simulation anchored to the QD-CamNetz source),
  supporting ~1.4K maskable parameters over 160-200 rounds. Same
  assumption class (ITS keys, HNDL-immune key layer). Differences
  measured in exp13: the trusted-conduit mode's higher key rate from the
  same physics (2.3x vs symmetric-basis sifting, 1.4x vs the efficient
  90/10 protocol they run, 1.2x in the asymptotic-sifting limit, bought
  by the trusted-source assumption), no basis-choice or mid-point
  measurement hardware, and an analyzed dropout-recovery path, which
  they do not address.
- Gold & Chitambar (arXiv:2505.10385): entanglement-derived correlated
  randomness replacing the MPC dealer (multiplication triples, with a
  Referee); this work targets FL aggregation with no Referee.
- SecAgg+ (Bell et al., CCS 2020): communication-efficient secure
  aggregation over sparse k-regular graphs; related to but distinct
  from the two-level tree over triangles used here.
- Classical information-theoretic secure aggregation (Zhao & Sun, IEEE
  Trans. IT 2022; arXiv:2606.07277, 2607.20890): IT masking with no
  quantum hardware, given dealer- or setup-provisioned correlated
  randomness. exp12's IT-dealer arm prices the provisioning: online
  distribution is HNDL-exposed (exp9) and courier distribution
  concentrates trust in the dealer. The quantum layer removes that
  provisioning step.
