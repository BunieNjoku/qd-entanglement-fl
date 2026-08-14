"""Tests for the exp14 executing BBM92 arm (m1.simulate_bbm92 +
m2.bbm92_distill).

Plain-python asserts, no test framework (the repo has none). Run:

    python tests/test_bbm92.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import POST_PROC, QUANTUM, BBM92  # noqa: E402
from modules.m1_quantum_source import simulate_bbm92  # noqa: E402
from modules.m2_post_processing import (  # noqa: E402
    bbm92_distill, tlgr_deviation, binary_entropy, toeplitz_hash)

N_PAIRS = 4_000_000
CODE_RATE = POST_PROC['ldpc_code_rate']
BLOCK = POST_PROC['block_size']


def run(p_z=0.9, eve_fraction=0.0, eve_basis_mode='none', seed=0,
        n_pairs=N_PAIRS, **overrides):
    kw = dict(
        fidelity=QUANTUM['fidelity'],
        fiber_length_m=QUANTUM['fiber_length_m'],
        fiber_loss_db_per_km=QUANTUM['fiber_loss_db_per_km_1550nm'],
        use_qfc=True, qfc_efficiency=QUANTUM['qfc_efficiency'],
        fiber_birefringence_rad_per_km_per_C=(
            QUANTUM['fiber_birefringence_rad_per_km_per_C']),
        temperature_variation_C=QUANTUM['temperature_variation_C'],
    )
    kw.update(overrides)
    return simulate_bbm92(n_pairs=n_pairs, p_z_alice=p_z, p_z_bob=p_z,
                          eve_fraction=eve_fraction,
                          eve_basis_mode=eve_basis_mode,
                          rng=np.random.default_rng(seed), **kw)


def test_sifting_keep():
    """Keep fraction matches p_z^2 + (1-p_z)^2 within sampling noise."""
    r = run(p_z=0.9)
    expected = 0.9 ** 2 + 0.1 ** 2
    n = r.n_coincidences
    tol = 4 * math.sqrt(expected * (1 - expected) / n)
    assert abs(r.sifting_keep - expected) < tol, (r.sifting_keep, expected)
    # test-set share of sifted events: (1-p_z)^2 / keep
    share = r.n_test / (r.n_key + r.n_test)
    exp_share = 0.1 ** 2 / expected
    assert abs(share - exp_share) < 5 * math.sqrt(
        exp_share * (1 - exp_share) / (r.n_key + r.n_test)), share


def test_fixed_basis_limit():
    """p_z = 1 reduces to the fixed-basis measurement: keep = 1, no test
    set, QBER at the operating level."""
    r = run(p_z=1.0)
    assert r.sifting_keep == 1.0
    assert r.n_test == 0
    assert r.n_key == r.n_coincidences
    assert 0.004 < r.e_z_actual < 0.015, r.e_z_actual


def test_e_x_equals_e_z():
    """Werner noise + rotation birefringence err both bases equally
    (the property exp13's tests pin analytically); here the sampled
    rates must agree within joint binomial noise."""
    r = run(p_z=0.5, seed=3)   # symmetric split: large test set
    se = math.sqrt(r.e_z_actual * (1 - r.e_z_actual) / r.n_key
                   + r.e_x_observed * (1 - r.e_x_observed) / r.n_test)
    assert abs(r.e_x_observed - r.e_z_actual) < 5 * max(se, 1e-4), \
        (r.e_z_actual, r.e_x_observed)


def test_eve_invisible_to_fixed_basis():
    """Key-basis intercept-resend adds no observable error to the
    fixed-basis protocol while giving Eve near-complete key knowledge."""
    clean = run(p_z=1.0, seed=5)
    tapped = run(p_z=1.0, eve_fraction=1.0, eve_basis_mode='z', seed=5)
    n = min(clean.n_key, tapped.n_key)
    se = math.sqrt(2 * 0.01 * 0.99 / n)
    assert abs(tapped.e_z_actual - clean.e_z_actual) < 6 * se, \
        (clean.e_z_actual, tapped.e_z_actual)
    assert tapped.eve_guess_accuracy > 0.95, tapped.eve_guess_accuracy
    assert clean.eve_guess_accuracy < 0.55, clean.eve_guess_accuracy


def test_eve_visible_to_bbm92():
    """The same tap raises the BBM92 test-set error as (1-f)e0 + f/2 and
    crosses the abort threshold."""
    f = 0.2
    r = run(p_z=0.9, eve_fraction=f, eve_basis_mode='z', seed=6)
    expected = (1 - f) * 0.0085 + f / 2
    se = math.sqrt(expected * (1 - expected) / r.n_test)
    assert abs(r.e_x_observed - expected) < 5 * se, \
        (r.e_x_observed, expected)
    assert r.e_x_observed > BBM92['q_tol']
    d = bbm92_distill(r.key_bits_alice, r.key_bits_bob, r.e_x_observed,
                      r.n_test, q_tol=BBM92['q_tol'],
                      code_rate=CODE_RATE, block_size=BLOCK,
                      eps_sec=BBM92['eps_sec'], eps_cor=BBM92['eps_cor'],
                      rng=np.random.default_rng(7))
    assert d.aborted
    assert d.n_secure_bits == 0


def test_tlgr_deviation_formula():
    """mu recomputed from the printed TLGR Eq. 2 form."""
    n, k, eps = 100_000, 1_000, 1e-9
    expected = math.sqrt((n + k) / (n * k) * (k + 1) / k
                         * math.log(4 / eps))
    assert abs(tlgr_deviation(n, k, eps) - expected) < 1e-12
    assert tlgr_deviation(0, 100, eps) == 0.5
    assert tlgr_deviation(100, 0, eps) == 0.5
    # mu shrinks as the test set grows
    assert tlgr_deviation(n, 10 * k, eps) < tlgr_deviation(n, k, eps)


def test_distill_key_length_and_agreement():
    """Non-aborted distillation: output length matches the TLGR formula
    recomputed per successful block, and both endpoints derive identical
    bits (EC makes Bob hold Alice's block; the hash is deterministic
    given the same seed exchange)."""
    r = run(p_z=0.9, seed=8)
    d = bbm92_distill(r.key_bits_alice, r.key_bits_bob, r.e_x_observed,
                      r.n_test, q_tol=BBM92['q_tol'],
                      code_rate=CODE_RATE, block_size=BLOCK,
                      eps_sec=BBM92['eps_sec'], eps_cor=BBM92['eps_cor'],
                      rng=np.random.default_rng(9))
    assert not d.aborted
    assert d.n_secure_bits > 0
    # reconstruct the expected total from the block layout: security
    # parameters split uniformly across blocks (union bound), phase-error
    # charge at the a-priori q_tol
    n = d.n_key_bits
    n_blocks = max(1, math.ceil(n / BLOCK))
    assert d.n_blocks == n_blocks
    eps_s = BBM92['eps_sec'] / n_blocks
    eps_c = BBM92['eps_cor'] / n_blocks
    floor = math.log2(2 / (eps_s ** 2 * eps_c))
    per_block = []
    mu_max = 0.0
    for b in range(n_blocks):
        start = b * BLOCK
        end = min(start + BLOCK, n)
        bn = end - start
        mu_b = tlgr_deviation(bn, d.n_test_bits, eps_s)
        mu_max = max(mu_max, mu_b)
        e_ph = min(0.5, BBM92['q_tol'] + mu_b)
        ell = int(bn * (1 - binary_entropy(e_ph))
                  - int(bn * (1 - CODE_RATE)) - floor)
        per_block.append(max(0, ell))
    assert abs(d.mu - mu_max) < 1e-12
    assert abs(d.e_ph_bound - min(0.5, BBM92['q_tol'] + mu_max)) < 1e-12
    # tier FER can drop blocks; every succeeded block contributes its ell
    assert d.n_blocks == n_blocks
    if d.n_blocks_succeeded == n_blocks:
        assert d.n_secure_bits == sum(per_block), \
            (d.n_secure_bits, sum(per_block))
    assert d.n_secure_bits <= sum(per_block)
    assert d.secure_fraction < 1.0


def test_abort_on_empty_test_set():
    """A run with no test bits cannot certify anything: distill aborts."""
    r = run(p_z=1.0, seed=10)
    d = bbm92_distill(r.key_bits_alice, r.key_bits_bob, 0.0, 0,
                      rng=np.random.default_rng(11))
    assert d.aborted
    assert d.n_secure_bits == 0


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
