# Entanglement-Based Shared Randomness for Federated Learning

Simulation of a federated learning system whose secure-aggregation
masks are provisioned by entangled photons: GaAs quantum-dot Bell
pairs, distributed over a campus-scale fiber mesh and distilled into
shared secret bits, supply the pairwise one-time masks of two-level
Bonawitz secure aggregation.

## What is modeled

- Physical layer: Bell-pair generation and detection anchored to a
  published rack-integrated GaAs quantum-dot source (Zena et al.,
  Adv. Quantum Technol. 9(10), 2026; arXiv:2604.02024), with
  sister-device measurements from the same group where the rack
  system does not report a quantity.
- Key distillation: LDPC reconciliation under a finite-blocklength
  frame-error model and Toeplitz privacy amplification, executed in
  two framings over the same physics: BBM92 with efficient sifting
  and TLGR finite-key privacy amplification (the default), and a
  fixed-basis trusted-conduit mode (source-model secret-key
  agreement).
- Learning layer: two-mask Bonawitz secure aggregation with modular
  one-time-pad masking of int32-quantized gradients, hierarchical
  aggregation over a Delaunay triangle mesh, dropout recovery, and
  per-round key freshness from pooled per-edge buffers.
- Adversaries: harvest-now-decrypt-later transcript replay against
  five provisioning arms, gradient-inversion attacks on the real
  construction, and an intercept-resend attack on the executing
  BBM92 arm.
- Baselines: Bonawitz-DH, Flamingo-PRF, and a classical
  information-theoretic dealer, all charged under the same byte
  accounting.

Fifteen experiments cover the rate budget, mesh scaling, convergence
parity, dropout recovery cost, throughput, HNDL exposure, availability
under source failure, and the rate cost of the QKD framing.

## Quick start

```bash
pip install -r qd_simulation/requirements.txt
cd qd_simulation
python run_all.py --quick       # reduced sweeps
python run_all.py               # full balanced mode
python run_all.py --exp 4 5     # selected experiments
```

Figures land in `qd_simulation/results/figures/`, numeric results in
`qd_simulation/results/summary.json`.

## Layout

```
qd_simulation/
  config.py           system parameters, provenance noted per value
  run_all.py          experiment runner
  modules/            m1 source and detection, m2 post-processing,
                      m3 mesh topology, m4 federated learning,
                      m6 HNDL arms, m7 classical baselines
  tests/              standalone assert suites
  requirements.txt
```

[qd_simulation/README.md](qd_simulation/README.md) documents the
hardware anchoring, the two protocol framings, the experiment roster,
the stated assumptions, and the terminology.

## Tests

Plain-python asserts, no test framework; each file runs standalone:

```bash
cd qd_simulation
python tests/test_noise_models.py
```
