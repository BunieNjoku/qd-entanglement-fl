"""Tests for the anchored source parameterization and the FSS
precession window model (m1).

Plain-python asserts, no test framework (the repo has none). Run:

    python tests/test_anchored_source.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import QUANTUM, OPERATING_POINT  # noqa: E402
from modules.m1_quantum_source import (  # noqa: E402
    werner_state, negativity_2n, werner_weight_for_negativity,
    fss_precession_omega, delay_window_coherence, calibrate_visibility,
    detected_delay_mode)

T1 = QUANTUM['t1_x_ps'] * 1e-12
SIGMA = QUANTUM['two_photon_jitter_fwhm_ps'] * 1e-12 / 2.355
OMEGA = fss_precession_omega(QUANTUM['fss_uev'])


def test_werner_negativity_inversion():
    """werner_weight_for_negativity inverts the partial-transpose
    negativity of m1's Werner family exactly."""
    for two_n in (0.5, 0.9, 0.94, 0.96, 0.98, 1.0):
        w = werner_weight_for_negativity(two_n)
        assert abs(negativity_2n(werner_state(w)) - two_n) < 1e-12, two_n


def test_config_bell_weight_matches_measured_negativity():
    """config's Bell weight is the Werner inversion of the measured
    six-hour-average negativity."""
    w = werner_weight_for_negativity(QUANTUM['negativity_2n'])
    assert abs(QUANTUM['fidelity'] - w) < 1e-12


def test_singles_calibration_closes():
    """The calibrated arm transmission reproduces the measured singles
    rate: clock x eta_ex x coupling x detector = 697 kHz."""
    model_singles = (QUANTUM['source_rep_rate_hz']
                     * QUANTUM['source_brightness']
                     * QUANTUM['coupling_efficiency']
                     * QUANTUM['detector_efficiency'])
    assert abs(model_singles - QUANTUM['singles_rate_v_basis_hz']) < 1.0


def test_operating_point_ranges():
    """Derived operating point sits in the ranges implied by the
    anchored inputs (order-of-magnitude guard against silent drift)."""
    assert 3000 < OPERATING_POINT['raw_edge_rate_hz'] < 6000
    assert 0.014 < OPERATING_POINT['qber'] < 0.017
    assert OPERATING_POINT['secret_rate_bps'] > 1000
    # Waks-2002 tagged fraction: g2 * eta_ex / 2
    expected_tag = 0.5 * QUANTUM['g2_xx_zero'] * QUANTUM['source_brightness']
    assert abs(OPERATING_POINT['tagged_fraction'] - expected_tag) < 1e-12


def test_coherence_limits():
    """Window-coherence limits: no precession gives |c| = 1; the full
    window approaches the analytic phase-compensated value
    |1/(1 - i Omega T1)| x exp(-(Omega sigma)^2 / 2)."""
    c0, cap0 = delay_window_coherence(-4 * SIGMA, 10 * T1, T1, SIGMA, 0.0)
    assert abs(abs(c0) - 1.0) < 1e-9
    assert cap0 > 0.999
    c_full, _ = delay_window_coherence(-5 * SIGMA, 10 * T1, T1, SIGMA, OMEGA)
    analytic = (1.0 / math.sqrt(1.0 + (OMEGA * T1) ** 2)
                * math.exp(-(OMEGA * SIGMA) ** 2 / 2.0))
    assert abs(abs(c_full) - analytic) < 0.01 * analytic, \
        (abs(c_full), analytic)


def test_captured_fraction_monotone():
    """Captured fraction grows with the window upper edge and tends
    to 1."""
    edges = np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * T1
    caps = [delay_window_coherence(-4 * SIGMA, e, T1, SIGMA, OMEGA)[1]
            for e in edges]
    assert all(b > a for a, b in zip(caps, caps[1:])), caps
    assert caps[-1] > 0.98


def test_visibility_calibration_reproduces_anchor():
    """v0 calibrated at a window reproduces the anchor negativity."""
    mode = detected_delay_mode(T1, SIGMA)
    lo, hi = mode - 4e-12, mode + 4e-12
    v0 = calibrate_visibility(QUANTUM['negativity_2n_window'],
                              lo, hi, T1, SIGMA, OMEGA)
    c, _ = delay_window_coherence(lo, hi, T1, SIGMA, OMEGA)
    assert abs(v0 * abs(c) - QUANTUM['negativity_2n_window']) < 1e-12
    # narrow-window coherence is nearly pure: v0 close to the anchor
    assert 0.9 < v0 <= 1.05, v0


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
