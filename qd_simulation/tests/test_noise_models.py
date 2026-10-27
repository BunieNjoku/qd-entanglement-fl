"""Tests for the m1 exp1a noise models and the m2 finite-blocklength
FER model.

Plain-python asserts, no test framework (the repo has none). Run:

    python tests/test_noise_models.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.m1_quantum_source import (  # noqa: E402
    werner_state, compute_chsh_s, amplitude_damping_state,
    amplitude_damping_gamma_for_fidelity, qber_in_basis, sample_chsh_s)
from modules.m2_post_processing import (  # noqa: E402
    tier_frame_error_rate, binary_entropy, estimate_capacity_qber)


def test_ad_fidelity_inversion():
    """gamma_for_fidelity inverts the Kraus-form fidelity exactly."""
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    for f in [0.9, 0.95, 0.987, 0.999]:
        g = amplitude_damping_gamma_for_fidelity(f)
        rho = amplitude_damping_state(g)
        f_out = float(np.real(phi_plus @ rho @ phi_plus))
        assert abs(f_out - f) < 1e-12, (f, f_out)


def test_ad_qber_z_closed_form():
    """One-sided amplitude damping on |phi+|: the only Z-basis error
    branch is K1 mapping |11> to |10>, with weight gamma/2."""
    for g in [0.01, 0.05, 0.2]:
        rho = amplitude_damping_state(g)
        assert abs(qber_in_basis(rho, 0.0) - g / 2) < 1e-12, g


def test_ad_breaks_basis_symmetry_werner_keeps_it():
    """Werner noise errs Z and X equally; amplitude damping does not.
    The e_ph = e_bit shortcut is a Werner property."""
    f = 0.987
    rho_w = werner_state(f)
    assert abs(qber_in_basis(rho_w, 0.0)
               - qber_in_basis(rho_w, np.pi / 4)) < 1e-12
    rho_ad = amplitude_damping_state(
        amplitude_damping_gamma_for_fidelity(f))
    e_z = qber_in_basis(rho_ad, 0.0)
    e_x = qber_in_basis(rho_ad, np.pi / 4)
    # closed forms: e_z = g/2 exactly; e_x = (1-s)^2/(4(1+s^2)/2)
    # branch + (g/2)(1/2) from the damped |10> branch ~ g/4, so the
    # X-basis rate is about HALF the Z-basis rate.
    assert e_x < e_z, (e_z, e_x)
    assert abs(e_x / e_z - 0.5) < 0.02, e_x / e_z


def test_sampled_chsh_converges():
    """The finite-statistics estimator is unbiased toward the exact S
    and its spread shrinks with sample size."""
    rho = werner_state(0.987)
    s_exact = compute_chsh_s(rho)
    m_small, sd_small = sample_chsh_s(rho, 2_000,
                                      rng=np.random.default_rng(1),
                                      n_repeats=50)
    m_large, sd_large = sample_chsh_s(rho, 200_000,
                                      rng=np.random.default_rng(2),
                                      n_repeats=50)
    assert abs(m_large - s_exact) < 0.02, (m_large, s_exact)
    assert sd_large < sd_small / 5, (sd_small, sd_large)


def test_fer_normal_approximation():
    """Finite-blocklength FER recomputed from the printed formula, plus
    the qualitative shape: ~0 deep in the design band, ~1 above the
    effective design point, monotone in QBER."""
    n, r, beta = 50000, 0.84, 1.10
    for q in [0.0085, 0.02, 0.03]:
        h = binary_entropy(q)
        v = q * (1 - q) * (math.log2((1 - q) / q)) ** 2
        z = (n * ((1 - r) / beta - h) + 0.5 * math.log2(n)) \
            / math.sqrt(n * v)
        expected = 0.5 * math.erfc(z / math.sqrt(2))
        got = tier_frame_error_rate(q, r, block_size=n, beta_dec=beta)
        assert abs(got - expected) < 1e-12, (q, got, expected)
    assert tier_frame_error_rate(0.0085, r) < 1e-50
    assert 0.05 < tier_frame_error_rate(0.02, r) < 0.25
    assert tier_frame_error_rate(0.03, r) > 0.999
    qs = np.linspace(0.005, 0.05, 20)
    fers = [tier_frame_error_rate(float(q), r) for q in qs]
    assert all(a <= b + 1e-12 for a, b in zip(fers, fers[1:]))
    # effective design point h2_inv((1-R)/beta) sits below capacity
    assert tier_frame_error_rate(estimate_capacity_qber(r), r) > 0.9


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
