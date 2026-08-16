# Entanglement-Based Shared Randomness for Federated Learning

Simulation package for an entanglement-based randomness extraction (source-model key distillation)
system feeding two-level Bonawitz secure aggregation in federated learning
over a campus-scale fiber mesh.

## Framing

This is **not** a QKD system. The protocol assumes:

- **Trusted endpoints.** Cooperating clients trust each other and their local
  measurement apparatus.
- **Trusted source.** The QD source is co-located at a known node; the CHSH
  inequality is used to verify the entangled state, not to bound an external
  eavesdropper's information.
- **Noisy channel.** The fiber is untrusted but not actively adversarial. QBER
  arises from birefringence, multi-photon emission, dark counts.

Under this model the protocol extracts **per-round shared random bits** from
pairwise Bell pairs between cluster members. Each round consumes fresh bits.
This is source-model secret-key agreement (Maurer 1993; Ahlswede-Csiszár 1993), not QKD.

## Core construction

- Two endpoints share Bell pair |Φ+⟩ via QD source -> QFC -> fiber.
- Coincidence detection + LDPC reconciliation + Toeplitz privacy amplification
  yields shared bit strings.
- **Triangles** (3-node clusters with 3 pairwise links) are the minimum
  geometric unit covering all pairs needed for 3-client Bonawitz masking.
- **Two-mask design** per Bonawitz 2017: pairwise mask from shared bits,
  self-mask from local quantum-randomness, Shamir shares of self-mask
  distributed across cluster.
- **Modular OTP masking** on int32-quantized gradients (not Gaussian floats).
- **Hierarchical aggregation:** intra-cluster Bonawitz -> inter-cluster
  Bonawitz over a tree derived from the Delaunay mesh dual.
- **Dropout recovery:** survivors reveal pairwise bits with dropped clients
  to the aggregator. Safe because bits are single-use per round.

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
| 5 | **Dropout recovery** | entangled-key two-mask aggregation tolerates client dropout at per-round freshness; classical Bonawitz needs Shamir reconstruction for pairwise leg |
| 6 | Throughput | Round-rate ceiling under shared-bit-rate constraints |
| 7 | **QBER -> key-rate / capacity envelope** | QBER prices key SUPPLY (rate, max maskable model size); the PA mask-entropy floor stays flat |
| 8 | Multi-triangle FL | Two-level hierarchical Bonawitz at 3/6/9/12 client scales |
| 9 | **HNDL transcript attack** | Record everything, break all computational secrets later: classical arm 100% retro-recovered, IT-dealer (online provisioning) arm 100% - IT masking with recorded provisioning still falls - entangled arm 0% |
| 10 | Availability under source failure | Kill one node's source: edges continue at ~half rate from far endpoints (availability/fairness, not security) |
| 11 | **Gradient inversion (real construction)** | Tabular DLG/iDLG: unmasked leaks (100% attr-infer), modular mask blocks (theorem), plaintext aggregate partially leaks (motivates DP-on-top) |
| 12 | **SOTA comparison** | Five executing arms: Bonawitz-DH, Flamingo-PRF, and a classical IT-dealer (online + courier provisioning). Every OTP arm pays O(d) recovery; the arms differ in what must be trusted: DH/PRF hardness, a dealer, or one's own source |
| 13 | **QKD-framing rate cost** | Same physics, two post-processings: source-model vs Shor-Preskill QKD framing. Ratio 2.2×/1.3×/1.1× by sifting protocol - prices the trusted-source assumption; not a security ranking |
| 14 | **Executing BBM92 arm** | The QKD framing executed on sampled bits (basis choice, sifting, TLGR finite-key PA): the same key-basis intercept-resend tap is invisible to the fixed-basis protocol (flat QBER, Eve predicts 99.9% of accepted key at full tap) and aborts BBM92 near the analytic threshold f* = 0.023. Finite-key surcharge priced honestly (executing ratio 4.1x at 8M-pair runs vs 1.32x asymptotic; fixed-length TLGR charge at the a-priori q_tol) |

## Assumptions

- **A - Bufferless FPGA RTL:** Treated as design target; not simulated.
- **B - Trusted source:** QD is at owned node, not adversary-controlled.
- **C - FPGA latency:** Published LDPC benchmarks × 2× safety margin.
- **D - Fiber QBER:** Parametric sweep over 1.5-4.0%.

## Output

Results in `results/`, tiered for the publication figure budget
(IEEE Network: six figures/tables total):

- `main/` - publication candidates: exp5 (dropout cost), exp7 (QBER
  envelope), exp9 (HNDL), exp12 (five-arm comparison), exp13
  (QKD-framing cost), exp14 (executing BBM92 arm)
- `peripheral/` - validation, sensitivity, and scaling figures kept
  for the technical-report version and reviewers (exp1-4, 6, 8, 10,
  11, exp14b basis split, quick-mode variants)
- `figure_captions.md` - paper-ready captions; figure titles are kept
  short and technical, the configuration and scope context lives here
- `replaced_by_text.md` - prose blocks replacing peripheral figures
  in the manuscript
- `summary.json` - numeric results (source of record)

The tier assignment is `MAIN_FIGURES` in run_all.py. Quick mode
(`--quick`) writes `*_quick`-suffixed files into `peripheral/` so
reduced-sweep runs never overwrite full-mode artifacts. Subset runs
(`--exp N`) merge into the existing summary rather than replacing it.

## Hardware Requirements

CPU, 16 GB RAM, ~2 GB disk. No GPU required.

## Project Structure

```
qd_simulation/
├── config.py                  # System parameters
├── run_all.py                 # Runner
├── modules/
│   ├── m1_quantum_source.py   # Bell-pair source + detection
│   ├── m2_post_processing.py  # Reconciliation + privacy amplification
│   ├── m3_mesh_topology.py    # Delaunay mesh + aggregation tree
│   └── m4_federated_learning.py # Two-mask Bonawitz + dropout recovery
└── results/
```

## Terminology (v2 - established concepts only)

The quantum layer is
**secret-key agreement in the source model** (Maurer 1993; Ahlswede & Csiszár
1993): sifting-free, fixed-basis BBM92-class entanglement key generation, LDPC
**information reconciliation**, and Toeplitz **privacy amplification** - the
Toeplitz step alone is "randomness extraction" under the Leftover Hash Lemma
(Renner & König 2005). The CHSH check is an **entanglement witness** (Horodecki
1996; Terhal 2000) used as source QA, not device independence. Masks are
**one-time pads** over Z_2^32 in the Bonawitz two-mask construction (CCS 2017),
consumed per round. PA additionally removes a **GLLP tagged fraction**
(multi-pair emissions; Gottesman-Lo-Lütkenhaus-Preskill 2004), which is what
supports the **harvest-now-decrypt-later** immunity claim.

Mode names: `entangled`, `gaussian-legacy` (comparison only - carries **no**
privacy proof), `prng`, `dp`, `unmasked`.

## Related work (positioning)

- **Bonawitz et al., CCS 2017** - the two-mask secure-aggregation core used here
  (per-execution ephemeral DH seeds; Shamir recovery on both legs).
- **Flamingo (Ma et al., IEEE S&P 2023)** - classical reveal-based per-round
  dropout recovery. Same recovery *mechanism* as ours; our differentiator is
  that per-round freshness is *unconditional* (no PRF/DH assumption, no
  key-erasure or forward-secrecy reliance).
- **QuNetQFL (arXiv:2501.12709)** - QKD-derived one-time-pad masks for FL,
  experimentally validated on a real quantum network: fully connected pairwise
  MDI-QKD among clients, classical aggregation-only server holding no key
  material, measured 32.8-240 kbit/s pairwise key rates (above our modeled
  12.6 kbit/s - theirs is hardware, ours simulation), supporting ~1.4K
  maskable parameters over 160-200 rounds. Same assumption class as ours (ITS
  keys, HNDL-immune key layer). Our differentiators are narrower and measured
  (exp13): a higher key rate from the same physics (2.2× vs symmetric-basis
  sifting, 1.3× vs the efficient 90/10 protocol they run, 1.1× in the
  asymptotic-sifting limit - bought by the trusted-source assumption, i.e. a
  weaker tolerated channel adversary), no basis-choice or mid-point
  measurement hardware, and an analyzed dropout-recovery story, which they do
  not address.
- **Gold & Chitambar (arXiv:2505.10385)** - entanglement-derived correlated
  randomness replacing the MPC dealer (multiplication triples, with a Referee);
  we target FL aggregation with no Referee.
- **SecAgg+ (Bell et al., CCS 2020)** - communication-efficient secure
  aggregation over sparse k-regular graphs; related to but distinct from our
  two-level tree over triangles.
- **Classical information-theoretic secure aggregation** (Zhao & Sun, IEEE
  Trans. IT 2022; recent constructions arXiv:2606.07277, 2607.20890) - IT
  masking with no quantum hardware, given dealer- or setup-provisioned
  correlated randomness. exp12's executing IT-dealer arm makes the
  provisioning cost explicit: online distribution is HNDL-exposed (exp9)
  and courier distribution concentrates total trust in the dealer. The
  quantum layer's role is precisely to remove that provisioning hole.
