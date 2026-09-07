"""Tests for the IT-dealer baseline arm.

Plain-python asserts, no test framework. Run:

    python tests/test_it_dealer.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.m7_baselines import (  # noqa: E402
    run_entangled, run_bonawitz, run_flamingo, run_it_dealer)
from modules.m6_hndl import run_transcript_round, retroactive_attack  # noqa: E402
from modules.m4_federated_learning import PoolAuditor, _SHAMIR_PRIME  # noqa: E402

CLUSTER = [0, 1, 2, 3]
D = 512
BPM = D * 32


def _grads(rng):
    return {c: torch.tensor(rng.normal(0, 0.1, D), dtype=torch.float32)
            for c in CLUSTER}


def _pool(rng, n_pairs=6, d=D):
    return rng.integers(0, 2, size=n_pairs * d * 32, dtype=np.int8)


def test_aggregate_equality_all_arms():
    """Design test 4a: all five arms recover the identical aggregate, with
    and without dropout."""
    rng = np.random.default_rng(7)
    grads = _grads(rng)
    for drops in (set(), {2}):
        g = lambda: {c: t.clone() for c, t in grads.items()}
        ae, _ = run_entangled(g(), CLUSTER, drops, _pool(rng),
                              np.random.default_rng(1))
        ab, _ = run_bonawitz(g(), CLUSTER, drops, np.random.default_rng(1))
        af, _ = run_flamingo(g(), CLUSTER, drops, 3,
                             np.random.default_rng(1))
        ao, _ = run_it_dealer(g(), CLUSTER, drops, _pool(rng),
                              np.random.default_rng(1), provisioning="online")
        ac, _ = run_it_dealer(g(), CLUSTER, drops, _pool(rng),
                              np.random.default_rng(1), provisioning="courier")
        for other in (ab, af, ao, ac):
            assert np.array_equal(ae, other), drops


def test_dealer_bytes_closed_forms():
    """Design test 4b: online setup bytes = n(n-1)*d*4 (each pad reaches
    BOTH endpoints), courier setup = 0 channel bytes/round; recovery
    bytes identical to the entangled arm's (same reveal mechanism)."""
    rng = np.random.default_rng(7)
    grads = _grads(rng)
    n = len(CLUSTER)
    _, ce = run_entangled({c: t.clone() for c, t in grads.items()},
                          CLUSTER, {2}, _pool(rng), np.random.default_rng(1))
    _, co = run_it_dealer({c: t.clone() for c, t in grads.items()},
                          CLUSTER, {2}, _pool(rng),
                          np.random.default_rng(1), provisioning="online")
    _, cc = run_it_dealer({c: t.clone() for c, t in grads.items()},
                          CLUSTER, {2}, _pool(rng),
                          np.random.default_rng(1), provisioning="courier")
    assert co.dealer_setup_bytes == n * (n - 1) * D * 4, co.dealer_setup_bytes
    assert cc.dealer_setup_bytes == 0
    assert co.recovery_bytes == ce.recovery_bytes == cc.recovery_bytes
    assert co.hndl_immune_pairwise_leg is False
    assert cc.hndl_immune_pairwise_leg is True  # vs a CHANNEL recorder only
    assert ce.dealer_setup_bytes == 0


def test_hndl_three_way():
    """Design test 5: mini transcript run - classical and online-dealer
    arms 100% retro-recovered, entangled arm 0%."""
    rng = np.random.default_rng(11)
    n = 3
    cluster = list(range(n))
    grads = {c: torch.tensor(rng.normal(0, 0.1, D), dtype=torch.float32)
             for c in cluster}
    self_seeds = {c: int(rng.integers(0, _SHAMIR_PRIME)) for c in cluster}
    pair_seeds = {(i, j): int(rng.integers(0, 2**31))
                  for i in range(n) for j in range(i + 1, n)}
    tr = {"entangled": [], "classical": [], "it_dealer_online": []}
    for rd, drops in enumerate([set(), {1}]):
        pool_e = _pool(rng, n_pairs=3)
        pool_d = _pool(rng, n_pairs=3)
        t_e, a_e = run_transcript_round(rd, grads, cluster, drops,
                                        "entangled", pairwise_pool=pool_e,
                                        self_seeds=self_seeds)
        t_c, a_c = run_transcript_round(rd, grads, cluster, drops,
                                        "classical",
                                        pair_mask_seeds=pair_seeds,
                                        self_seeds=self_seeds)
        t_d, a_d = run_transcript_round(rd, grads, cluster, drops,
                                        "it_dealer_online",
                                        pairwise_pool=pool_d,
                                        self_seeds=self_seeds)
        assert np.array_equal(a_e, a_c) and np.array_equal(a_e, a_d)
        tr["entangled"].append(t_e)
        tr["classical"].append(t_c)
        tr["it_dealer_online"].append(t_d)
    assert retroactive_attack(tr["classical"],
                              "classical")["recovered_fraction"] == 1.0
    assert retroactive_attack(tr["it_dealer_online"],
                              "it_dealer_online")["recovered_fraction"] == 1.0
    assert retroactive_attack(tr["entangled"],
                              "entangled")["recovered_fraction"] == 0.0


def test_provisioner_compromise():
    """The second adversary (recorder + dealer archive) opens BOTH
    dealer arms - courier included - and still nothing in the
    entangled arm, which has no provisioning third party."""
    rng = np.random.default_rng(23)
    n = 3
    cluster = list(range(n))
    grads = {c: torch.tensor(rng.normal(0, 0.1, D), dtype=torch.float32)
             for c in cluster}
    self_seeds = {c: int(rng.integers(0, _SHAMIR_PRIME)) for c in cluster}
    tr = {"it_dealer_courier": [], "entangled": []}
    for rd, drops in enumerate([set(), {2}]):
        for arm in tr:
            t, _ = run_transcript_round(rd, grads, cluster, drops, arm,
                                        pairwise_pool=_pool(rng, n_pairs=3),
                                        self_seeds=self_seeds)
            tr[arm].append(t)
    # recorder alone: courier survives
    assert retroactive_attack(
        tr["it_dealer_courier"],
        "it_dealer_courier")["recovered_fraction"] == 0.0
    # recorder + dealer archive: courier falls, entangled does not
    assert retroactive_attack(
        tr["it_dealer_courier"], "it_dealer_courier",
        provisioner_compromise=True)["recovered_fraction"] == 1.0
    assert retroactive_attack(
        tr["entangled"], "entangled",
        provisioner_compromise=True)["recovered_fraction"] == 0.0


def test_determinism():
    """Design test 6: a fixed-seed five-arm run is identical across
    repeats - aggregates and byte counts (wall-clock fields excluded:
    host-dependent)."""
    def once():
        rng = np.random.default_rng(7)
        grads = _grads(rng)
        g = lambda: {cl: t.clone() for cl, t in grads.items()}
        out = {}
        out['entangled'] = run_entangled(g(), CLUSTER, {2}, _pool(rng),
                                         np.random.default_rng(1))
        out['bonawitz'] = run_bonawitz(g(), CLUSTER, {2},
                                       np.random.default_rng(1))
        out['flamingo'] = run_flamingo(g(), CLUSTER, {2}, 3,
                                       np.random.default_rng(1))
        out['dealer_on'] = run_it_dealer(g(), CLUSTER, {2}, _pool(rng),
                                         np.random.default_rng(1),
                                         provisioning="online")
        out['dealer_co'] = run_it_dealer(g(), CLUSTER, {2}, _pool(rng),
                                         np.random.default_rng(1),
                                         provisioning="courier")
        return {k: (a, (c.recovery_bytes, c.dealer_setup_bytes,
                        c.payload_bytes, c.share_bytes))
                for k, (a, c) in out.items()}
    r1, r2 = once(), once()
    for k in r1:
        assert np.array_equal(r1[k][0], r2[k][0]), k
        assert r1[k][1] == r2[k][1], k


def test_pool_auditor_on_dealer_pool():
    """Design test 7: the dealer pool obeys the same single-use OTP audit
    as the distilled pool - per-pair per-round slices are disjoint, and
    reusing any slice raises."""
    n_pairs, rounds = 6, 3
    auditor = PoolAuditor("dealer", n_pairs * BPM * rounds)
    off = 0
    for _ in range(rounds):
        for _ in range(n_pairs):
            auditor.consume(off, BPM)
            off += BPM
    assert auditor.bits_consumed == n_pairs * BPM * rounds
    try:
        auditor.consume(0, BPM)  # reuse round 0, pair 0
        raise AssertionError("PoolAuditor accepted an OTP reuse")
    except ValueError:
        pass


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
