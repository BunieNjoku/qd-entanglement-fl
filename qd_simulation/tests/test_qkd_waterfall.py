"""Tests for the exp13 QKD-framed rate waterfall (COMPARISON_DESIGN 6).

Plain-python asserts, no test framework (the repo has none). Run:

    python tests/test_qkd_waterfall.py
"""
import inspect
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import POST_PROC, QUANTUM, QKD_FRAMING, OPERATING_POINT  # noqa: E402
from modules.m2_post_processing import (  # noqa: E402
    compute_rate_waterfall, compute_rate_waterfall_qkd, binary_entropy,
    tier_frame_error_rate, default_tagged_fraction)

RAW = OPERATING_POINT['raw_edge_rate_hz']
Q = OPERATING_POINT['qber']
CODE_RATE = POST_PROC['ldpc_code_rate']
MARGIN = POST_PROC['security_margin_bits']
BLOCK = POST_PROC['block_size']
DUTY = POST_PROC['bell_test_duty_cycle']
BETA = 1.15


def h2(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def src_wf(raw, q, **kw):
    return compute_rate_waterfall(raw, q, code_rate=CODE_RATE,
                                  security_margin=MARGIN, block_size=BLOCK,
                                  bell_duty_cycle=DUTY, **kw)


def qkd_wf(raw, q, keep, **kw):
    return compute_rate_waterfall_qkd(raw, q, keep, code_rate=CODE_RATE,
                                      security_margin=MARGIN,
                                      block_size=BLOCK,
                                      bell_duty_cycle=DUTY, **kw)


def test_closed_forms():
    """Design test 1: both arms against independently computed closed
    forms, at two sifting keeps (a keep=1-only check is blind to
    misplaced sifting factors)."""
    leak = max(BETA * h2(Q), 1 - CODE_RATE)
    fer = tier_frame_error_rate(Q, CODE_RATE)
    d_tag = default_tagged_fraction()

    src_expect = (1 - leak - d_tag - MARGIN / BLOCK) * (1 - fer) \
        * (1 - DUTY) * RAW
    got = src_wf(RAW, Q)['final_secure_rate']
    assert abs(got - src_expect) < 1e-6, (got, src_expect)

    for keep in (0.5, 0.82):
        qkd_expect = keep * (1 - h2(Q) - leak - MARGIN / BLOCK) \
            * (1 - fer) * (1 - DUTY) * RAW
        got = qkd_wf(RAW, Q, keep)['final_secure_rate']
        assert abs(got - qkd_expect) < 1e-6, (keep, got, qkd_expect)


def test_reduction_identity():
    """At keep=1, the per-block fraction gap between the arms is exactly
    D_tag - h2(q): the source arm charges tagging, the QKD arm the Eve
    bound, and everything else is identical."""
    s = src_wf(RAW, Q)
    k = qkd_wf(RAW, Q, 1.0)
    frac_s = s['after_pa'] / RAW
    frac_k = k['after_pa'] / RAW
    gap = frac_s - frac_k
    expect = h2(Q) - default_tagged_fraction()
    assert abs(gap - expect) < 1e-12, (gap, expect)


def test_monotonicity_and_ratio_floor():
    """Design test 2: QKD rate strictly increasing in sifting keep,
    strictly decreasing in QBER; ratio >= 1 across the configured
    sweep range."""
    rates = [qkd_wf(RAW, Q, keep)['final_secure_rate']
             for keep in (0.3, 0.5, 0.82, 1.0)]
    assert all(b > a for a, b in zip(rates, rates[1:])), rates

    qs = np.linspace(0.005, 0.06, 12)
    prev = None
    for qq in qs:
        r = qkd_wf(RAW, qq, 0.82)['final_secure_rate']
        if prev is not None:
            assert r < prev, (qq, r, prev)  # strictly decreasing in QBER
        prev = r
        s = src_wf(RAW, qq)['final_secure_rate']
        if r > 0:
            assert s / r >= 1.0, (qq, s, r)


def test_no_tagging_path():
    """Design test 3: the QKD waterfall has no tagged-fraction input at
    all (entanglement-based analysis has none), reports tagged_fraction
    0.0, and the SOURCE waterfall does respond to tagging - the
    deliberate per-arm asymmetry, pinned."""
    params = inspect.signature(compute_rate_waterfall_qkd).parameters
    assert 'tagged_fraction' not in params, list(params)
    assert qkd_wf(RAW, Q, 0.82)['tagged_fraction'] == 0.0

    s0 = src_wf(RAW, Q, tagged_fraction=0.008)['final_secure_rate']
    s1 = src_wf(RAW, Q, tagged_fraction=0.020)['final_secure_rate']
    assert s0 != s1


def test_matched_ec_model():
    """Design test 3b: at every sweep point, both arms charge identical
    leak_EC and FER - in the fixed-code model and in the rate-adapted
    envelope alike. Mixing EC models across arms is the round-2 MAJOR
    this pins out."""
    for adapted in (False, True):
        for qq in np.linspace(0.005, 0.06, 12):
            s = src_wf(RAW, qq, assume_rate_adapted=adapted)
            k = qkd_wf(RAW, qq, 0.82, assume_rate_adapted=adapted)
            assert abs(s['ec_overhead_pct'] - k['ec_overhead_pct']) < 1e-12
            assert s['frame_error_rate'] == k['frame_error_rate']


def test_multi_photon_guard():
    """Design test 3, config-plumbing form: mutating the multi-pair
    probability moves the SOURCE arm's default-tagged waterfall and
    cannot move the QKD arm (which has no tagging path at all)."""
    orig = QUANTUM['multi_photon_prob']
    try:
        base_src = src_wf(RAW, Q)['final_secure_rate']
        base_qkd = qkd_wf(RAW, Q, 0.82)['final_secure_rate']
        QUANTUM['multi_photon_prob'] = orig * 3
        assert src_wf(RAW, Q)['final_secure_rate'] != base_src
        assert qkd_wf(RAW, Q, 0.82)['final_secure_rate'] == base_qkd
    finally:
        QUANTUM['multi_photon_prob'] = orig


def test_phase_error_symmetry():
    """e_ph = e_bit is VERIFIED, not assumed, under m1's noise model:
    the Werner component is basis-isotropic and a birefringence rotation
    errs the Z and X measurement bases equally, so e_X = e_Z exactly.
    Pins the exp13 caveat's 'verified' claim."""
    from modules.m1_quantum_source import werner_state

    def qber_in_basis(rho, angle):
        # Projectors for both endpoints measuring at `angle` and
        # `angle`+90deg; QBER = P(outcomes differ).
        def proj(a):
            v = np.array([math.cos(a), math.sin(a)])
            return np.outer(v, v)
        p0, p1 = proj(angle), proj(angle + math.pi / 2)
        err = np.kron(p0, p1) + np.kron(p1, p0)
        return float(np.real(np.trace(err @ rho)))

    for theta in (0.0, 0.05, 0.3):  # birefringence rotation on photon 2
        c, s = math.cos(theta), math.sin(theta)
        r2 = np.array([[c, -s], [s, c]])
        u = np.kron(np.eye(2), r2)
        rho = u @ werner_state(QUANTUM['fidelity']) @ u.T
        e_z = qber_in_basis(rho, 0.0)
        e_x = qber_in_basis(rho, math.pi / 4)
        assert abs(e_z - e_x) < 1e-12, (theta, e_z, e_x)


def test_operating_point_ratios():
    """The design-review estimates, now pinned as regression values:
    ratios ~2.2x / ~1.3x / ~1.1x at the modeled operating point."""
    src_rate = src_wf(RAW, Q)['final_secure_rate']
    modes = QKD_FRAMING['sifting_modes']
    ratios = {n: src_rate / qkd_wf(RAW, Q, keep)['final_secure_rate']
              for n, keep in modes.items()}
    assert 2.0 < ratios['symmetric'] < 2.4, ratios
    assert 1.2 < ratios['efficient-90/10'] < 1.45, ratios
    assert 1.0 < ratios['efficient-asymptotic'] < 1.2, ratios


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
