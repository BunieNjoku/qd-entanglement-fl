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
This is closer to the randomness-expansion framework than to QKD.

## Core construction

- Two endpoints share Bell pair |Φ+⟩ via QD source → QFC → fiber.
- Coincidence detection + LDPC reconciliation + Toeplitz privacy amplification
  yields shared bit strings.
- **Triangles** (3-node clusters with 3 pairwise links) are the minimum
  geometric unit covering all pairs needed for 3-client Bonawitz masking.
- **Two-mask design** per Bonawitz 2017: pairwise mask from shared bits,
  self-mask from local quantum-randomness, Shamir shares of self-mask
  distributed across cluster.
- **Modular OTP masking** on int32-quantized gradients (not Gaussian floats).
- **Hierarchical aggregation:** intra-cluster Bonawitz → inter-cluster
  Bonawitz over a tree derived from the Delaunay mesh dual.
- **Dropout recovery:** survivors reveal pairwise bits with dropped clients
  to the aggregator. Safe because bits are single-use per round.

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py             # Full run, all experiments
python run_all.py --quick     # Reduced sweeps, ~15-20 min
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
| 7 | **QBER → key-rate / capacity envelope** | QBER prices key SUPPLY (rate, max maskable model size); the PA mask-entropy floor stays flat |
| 8 | Multi-triangle FL | Two-level hierarchical Bonawitz at 3/6/9/12 client scales |
| 9 | **HNDL transcript attack** | Record everything, break all computational secrets later: classical arm 100% retro-recovered, entangled arm 0% (pairwise bits never transit a channel) |

## Assumptions

- **A — Bufferless FPGA RTL:** Treated as design target; not simulated.
- **B — Trusted source:** QD is at owned node, not adversary-controlled.
- **C — FPGA latency:** Published LDPC benchmarks × 2× safety margin.
- **D — Fiber QBER:** Parametric sweep over 1.5–4.0%.

## Output

Results in `results/`:
- `exp1*` — Source characterization
- `exp2*` — Rate budget
- `exp3*` — Mesh scaling
- `exp4*` — FL training convergence
- `exp5_dropout_recovery.png` — Dropout recovery validation
- `exp6_throughput.png` — Rate-constrained throughput
- `exp7_qber_key_rate_envelope.png` — QBER → key-rate / capacity envelope
- `exp8*` — Multi-triangle scaling
- `exp9_hndl_transcript.png` — HNDL retroactive-recovery comparison
- `summary.json` — Numeric results

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
│   ├── m4_federated_learning.py # Two-mask Bonawitz + dropout recovery
│   └── m5_cost_model.py       # CAPEX/OPEX (illustrative)
└── results/
```

## Terminology (v2 — established concepts only)

The quantum layer is
**secret-key agreement in the source model** (Maurer 1993; Ahlswede & Csiszár
1993): sifting-free, fixed-basis BBM92-class entanglement key generation, LDPC
**information reconciliation**, and Toeplitz **privacy amplification** — the
Toeplitz step alone is "randomness extraction" under the Leftover Hash Lemma
(Renner & König 2005). The CHSH check is an **entanglement witness** (Horodecki
1996; Terhal 2000) used as source QA, not device independence. Masks are
**one-time pads** over Z_2^32 in the Bonawitz two-mask construction (CCS 2017),
consumed per round. PA additionally removes a **GLLP tagged fraction**
(multi-pair emissions; Gottesman–Lo–Lütkenhaus–Preskill 2004), which is what
supports the **harvest-now-decrypt-later** immunity claim.

Mode names: `entangled`, `gaussian-legacy` (comparison only — carries **no**
privacy proof), `prng`, `dp`, `unmasked`.

## Related work (positioning)

- **Bonawitz et al., CCS 2017** — the two-mask secure-aggregation core used here
  (per-execution ephemeral DH seeds; Shamir recovery on both legs).
- **Flamingo (Ma et al., IEEE S&P 2023)** — classical reveal-based per-round
  dropout recovery. Same recovery *mechanism* as ours; our differentiator is
  that per-round freshness is *unconditional* (no PRF/DH assumption, no
  key-erasure or forward-secrecy reliance).
- **QuNetQFL (arXiv:2501.12709)** — QKD-derived masks for FL over a real quantum
  network, with a central server and transmitted (MDI-QKD) keys; here keys are
  source-model distilled, never transit, and we quantify the ~2× rate dividend
  of dropping the QKD framing.
- **Gold & Chitambar (arXiv:2505.10385)** — entanglement-derived correlated
  randomness replacing the MPC dealer (multiplication triples, with a Referee);
  we target FL aggregation with no Referee.
- **SecAgg+ (Bell et al., CCS 2020)** — hierarchical secure aggregation; our
  two-level triangle mesh is a small-n instance of that design space.
