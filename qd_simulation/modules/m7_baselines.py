"""
Module 7: Classical secure-aggregation baselines.

Bonawitz SecAgg (DH-seeded PRG masks, Shamir recovery of a dropped
client's DH secret), Flamingo (per-round PRF seeds, seed reveal) and an
IT dealer (provisioned one-time pads, online or courier delivery), run
beside this system's entangled arm. All arms share m4's modular masking,
wire format and byte scoping and differ only in pairwise-mask source and
dropout recovery. Each round is executed and the recovered aggregate is
asserted equal to the plaintext survivor sum.
"""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set

try:
    from modules.m4_federated_learning import (
        quantize_gradient, derive_modular_mask, apply_modular_mask,
        shamir_split, shamir_reconstruct, _SHAMIR_PRIME, shamir_field_element,
        BYTES_PER_MASKED_WORD, BYTES_PER_SHAMIR_SHARE,
        BYTES_PER_SELFMASK_SEED,
    )
except ImportError:  # pragma: no cover
    from .m4_federated_learning import (
        quantize_gradient, derive_modular_mask, apply_modular_mask,
        shamir_split, shamir_reconstruct, _SHAMIR_PRIME,
        shamir_field_element,
        BYTES_PER_MASKED_WORD, BYTES_PER_SHAMIR_SHARE,
        BYTES_PER_SELFMASK_SEED,
    )

# Wire sizes of the recovered objects that differ across protocols.
# Bonawitz: one share of a dropped client's DH secret scalar.
BYTES_PER_DH_SHARE = BYTES_PER_SHAMIR_SHARE
# Flamingo: the per-round PRF pairwise seed revealed by a survivor.
BYTES_PER_PRF_SEED = 4
# This system: one revealed pairwise mask word, n_params of them.
BYTES_PER_REVEAL_WORD = BYTES_PER_MASKED_WORD


def _prg(seed, n_params):
    """Shared PRG expansion (the same primitive m4 uses for self-masks)."""
    return (np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
            .integers(0, 1 << 32, size=n_params, dtype=np.uint64)
            .astype(np.uint32))


@dataclass
class BaselineCost:
    """Per-round cost of one protocol under a given (n, d, dropouts)."""
    protocol: str
    n_clients: int
    n_params: int
    n_dropped: int
    payload_bytes: int = 0          # survivors' masked gradients
    selfmask_bytes: int = 0         # survivors' self-mask seed handling
    share_bytes: int = 0            # Shamir shares distributed (self-mask leg)
    recovery_bytes: int = 0         # pairwise-leg dropout recovery traffic
    downlink_bytes: int = 0         # global-model broadcast
    aggregator_prg_expansions: int = 0   # PRG mask (re)builds at the server
    aggregator_recovery_s: float = 0.0   # measured recovery wall-clock
    assumption_class: str = ""      # e.g. "unconditional" | "computational"
    hndl_immune_pairwise_leg: bool = False
    # Per-round pairwise-key setup traffic beyond the shared self-mask
    # machinery (the dealer arm's pad distribution; Bonawitz's DH share
    # redistribution is already in share_bytes). 0 when amortized or local.
    dealer_setup_bytes: int = 0
    # Who must be trusted for the pairwise-mask leg to hold.
    trust_surface: str = ""
    # Strongest channel adversary the key layer tolerates.
    channel_adversary_tolerated: str = ""

    def uplink_per_client(self):
        up = (self.payload_bytes + self.selfmask_bytes + self.share_bytes
              + self.recovery_bytes)
        return up / max(self.n_clients - self.n_dropped, 1)


def _common_setup(gradients, cluster, rng):
    cluster = list(cluster)
    n = len(cluster)
    n_params = gradients[cluster[0]].numel()
    quant = {c: quantize_gradient(gradients[c]) for c in cluster}
    self_seeds = {c: shamir_field_element(rng) for c in cluster}
    self_masks = {c: _prg(self_seeds[c], n_params) for c in cluster}
    # Self-mask seeds are Shamir-shared (threshold 2) in all arms: the
    # double mask, charged identically so it does not bias the comparison.
    shares = {c: shamir_split(self_seeds[c], n, 2,
                              np.random.default_rng(self_seeds[c] + 1))
              for c in cluster}
    return cluster, n, n_params, quant, self_seeds, self_masks, shares


def _finish(cluster, dropouts, quant, self_masks, agg):
    """Remove survivors' self-masks, verify against the plaintext survivor
    sum, and return the recovered aggregate."""
    survivors = [c for c in cluster if c not in dropouts]
    for s in survivors:
        agg = (agg - self_masks[s]).astype(np.uint32)
    truth = np.zeros_like(agg)
    for s in survivors:
        truth = (truth + quant[s]).astype(np.uint32)
    assert np.array_equal(agg, truth), "baseline recovered wrong aggregate"
    return agg


def run_entangled(gradients, cluster, dropouts, distilled_pool, rng):
    """This system: one-time-pad pairwise masks, reveal-based recovery."""
    cluster, n, n_params, quant, self_seeds, self_masks, shares = \
        _common_setup(gradients, cluster, rng)
    dropouts = set(dropouts)
    survivors = [c for c in cluster if c not in dropouts]

    # Pairwise masks from distilled bits (one-time pad).
    pair_bits, pair_masks, off = {}, {}, 0
    bpm = n_params * 32
    for i, ci in enumerate(cluster):
        for cj in cluster[i + 1:]:
            pair_bits[(ci, cj)] = distilled_pool[off:off + bpm]
            pair_masks[(ci, cj)] = derive_modular_mask(
                n_params, pair_bits[(ci, cj)])
            off += bpm

    # Survivors' payloads.
    agg = np.zeros(n_params, dtype=np.uint32)
    for c in survivors:
        p = quant[c].copy()
        for c2 in cluster:
            if c2 == c:
                continue
            lo, hi = (c, c2) if c < c2 else (c2, c)
            p = apply_modular_mask(p, pair_masks[(lo, hi)],
                                   sign=+1 if c < c2 else -1)
        p = apply_modular_mask(p, self_masks[c], sign=+1)
        agg = (agg + p).astype(np.uint32)

    # Recovery: each survivor reveals its full pairwise bits with each
    # dropped partner; the aggregator subtracts directly, no PRG expansion.
    t0 = time.perf_counter()
    recovery_bytes = 0
    for d in dropouts:
        for s in survivors:
            lo, hi = (s, d) if s < d else (d, s)
            mask = derive_modular_mask(n_params, pair_bits[(lo, hi)])
            agg = ((agg - mask) if s < d else (agg + mask)).astype(np.uint32)
            recovery_bytes += n_params * BYTES_PER_REVEAL_WORD
    rec_s = time.perf_counter() - t0

    agg = _finish(cluster, dropouts, quant, self_masks, agg)
    cost = BaselineCost(
        protocol="entangled", n_clients=n, n_params=n_params,
        n_dropped=len(dropouts),
        payload_bytes=len(survivors) * n_params * BYTES_PER_MASKED_WORD,
        selfmask_bytes=len(survivors) * BYTES_PER_SELFMASK_SEED,
        share_bytes=n * (n - 1) * BYTES_PER_SHAMIR_SHARE,
        recovery_bytes=recovery_bytes,
        aggregator_prg_expansions=0,
        aggregator_recovery_s=rec_s,
        assumption_class="unconditional",
        hndl_immune_pairwise_leg=True,
        trust_surface="own source (endpoint-owned hardware)",
        channel_adversary_tolerated="passive only")
    return agg, cost


def run_bonawitz(gradients, cluster, dropouts, rng):
    """Bonawitz SecAgg: DH-seeded pairwise PRG masks; dropout recovery by
    Shamir-reconstructing the dropped client's DH secret and recomputing."""
    cluster, n, n_params, quant, self_seeds, self_masks, shares = \
        _common_setup(gradients, cluster, rng)
    dropouts = set(dropouts)
    survivors = [c for c in cluster if c not in dropouts]

    # Models the result of DH agreement (a symmetric per-pair scalar),
    # not the group operation.
    dh_secret = {c: max(1, shamir_field_element(rng)) for c in cluster}
    dh_shares = {c: shamir_split(dh_secret[c], n, 2,
                                 np.random.default_rng(dh_secret[c] + 7))
                 for c in cluster}

    def pair_seed(a, b):
        # Symmetric in (a,b): both endpoints derive the same scalar.
        return (dh_secret[a] * dh_secret[b]) % _SHAMIR_PRIME

    pair_masks = {}
    for i, ci in enumerate(cluster):
        for cj in cluster[i + 1:]:
            pair_masks[(ci, cj)] = _prg(pair_seed(ci, cj), n_params)

    agg = np.zeros(n_params, dtype=np.uint32)
    for c in survivors:
        p = quant[c].copy()
        for c2 in cluster:
            if c2 == c:
                continue
            lo, hi = (c, c2) if c < c2 else (c2, c)
            p = apply_modular_mask(p, pair_masks[(lo, hi)],
                                   sign=+1 if c < c2 else -1)
        p = apply_modular_mask(p, self_masks[c], sign=+1)
        agg = (agg + p).astype(np.uint32)

    # Recovery: reconstruct each dropped client's DH secret from a
    # threshold of survivors' shares, then recompute and cancel its
    # pairwise masks with each survivor. The PRG expansions are
    # aggregator compute.
    t0 = time.perf_counter()
    recovery_bytes = 0
    prg_expansions = 0
    for d in dropouts:
        avail = [dh_shares[d][cluster.index(s)] for s in survivors][:2]
        rec = shamir_reconstruct(avail)
        assert rec == dh_secret[d], "Bonawitz DH reconstruction failed"
        recovery_bytes += 2 * BYTES_PER_DH_SHARE  # threshold-many scalar shares
        for s in survivors:
            seed = (rec * dh_secret[s]) % _SHAMIR_PRIME
            mask = _prg(seed, n_params)
            prg_expansions += 1
            lo, hi = (s, d) if s < d else (d, s)
            # survivor s carried sign(s,d)*mask; cancel it
            agg = ((agg - mask) if s < d else (agg + mask)).astype(np.uint32)
    rec_s = time.perf_counter() - t0

    agg = _finish(cluster, dropouts, quant, self_masks, agg)
    # Per-execution ephemeral DH: each round redistributes n(n-1) Shamir
    # shares of the fresh DH secrets in addition to the self-mask shares.
    cost = BaselineCost(
        protocol="bonawitz", n_clients=n, n_params=n_params,
        n_dropped=len(dropouts),
        payload_bytes=len(survivors) * n_params * BYTES_PER_MASKED_WORD,
        selfmask_bytes=len(survivors) * BYTES_PER_SELFMASK_SEED,
        share_bytes=(n * (n - 1) * BYTES_PER_SHAMIR_SHARE       # self-mask
                     + n * (n - 1) * BYTES_PER_DH_SHARE),       # DH secret
        recovery_bytes=recovery_bytes,
        aggregator_prg_expansions=prg_expansions,
        aggregator_recovery_s=rec_s,
        assumption_class="computational",
        hndl_immune_pairwise_leg=False,
        trust_surface="none (pairwise DH/PRF)",
        channel_adversary_tolerated="active")
    return agg, cost


def run_flamingo(gradients, cluster, dropouts, round_idx, rng):
    """Flamingo: per-round PRF pairwise seeds; dropout recovery by
    revealing the per-round PRF seed (scalar) for the dropped partner."""
    cluster, n, n_params, quant, self_seeds, self_masks, shares = \
        _common_setup(gradients, cluster, rng)
    dropouts = set(dropouts)
    survivors = [c for c in cluster if c not in dropouts]

    # Long-term pairwise secret r_{i,j}; per-round seed h = PRF(r, t).
    # The PRF is modeled as a keyed hash into the scalar field; the
    # comparison depends on the reveal being a per-round seed, not on
    # the PRF family.
    longterm = {}
    for i, ci in enumerate(cluster):
        for cj in cluster[i + 1:]:
            longterm[(ci, cj)] = max(1, shamir_field_element(rng))

    def prf(rk, t):
        return (rk * 1103515245 + 12345 + t * 2654435761) % _SHAMIR_PRIME

    round_seed, pair_masks = {}, {}
    for key, rk in longterm.items():
        round_seed[key] = prf(rk, round_idx)
        pair_masks[key] = _prg(round_seed[key], n_params)

    agg = np.zeros(n_params, dtype=np.uint32)
    for c in survivors:
        p = quant[c].copy()
        for c2 in cluster:
            if c2 == c:
                continue
            lo, hi = (c, c2) if c < c2 else (c2, c)
            p = apply_modular_mask(p, pair_masks[(lo, hi)],
                                   sign=+1 if c < c2 else -1)
        p = apply_modular_mask(p, self_masks[c], sign=+1)
        agg = (agg + p).astype(np.uint32)

    # Recovery: each survivor reveals the per-round PRF seed for the
    # dropped partner; the aggregator PRG-expands and cancels.
    t0 = time.perf_counter()
    recovery_bytes = 0
    prg_expansions = 0
    for d in dropouts:
        for s in survivors:
            lo, hi = (s, d) if s < d else (d, s)
            mask = _prg(round_seed[(lo, hi)], n_params)
            prg_expansions += 1
            recovery_bytes += BYTES_PER_PRF_SEED  # one scalar seed revealed
            agg = ((agg - mask) if s < d else (agg + mask)).astype(np.uint32)
    rec_s = time.perf_counter() - t0

    agg = _finish(cluster, dropouts, quant, self_masks, agg)
    cost = BaselineCost(
        protocol="flamingo", n_clients=n, n_params=n_params,
        n_dropped=len(dropouts),
        payload_bytes=len(survivors) * n_params * BYTES_PER_MASKED_WORD,
        selfmask_bytes=len(survivors) * BYTES_PER_SELFMASK_SEED,
        share_bytes=n * (n - 1) * BYTES_PER_SHAMIR_SHARE,
        recovery_bytes=recovery_bytes,
        aggregator_prg_expansions=prg_expansions,
        aggregator_recovery_s=rec_s,
        assumption_class="computational",
        hndl_immune_pairwise_leg=False,
        trust_surface="none (pairwise DH/PRF)",
        channel_adversary_tolerated="active")
    return agg, cost


def run_it_dealer(gradients, cluster, dropouts, dealer_pool, rng,
                  provisioning="online"):
    """IT-dealer baseline: an offline trusted dealer provisions each pair
    with one-time pads; round mechanics match run_entangled. "online"
    charges n(n-1)*d*4 B of pad distribution per round over recorded
    channels; "courier" charges 0 channel bytes (pads delivered out of
    band). Separate from run_entangled because the cost labels differ.
    """
    assert provisioning in ("online", "courier"), provisioning
    cluster, n, n_params, quant, self_seeds, self_masks, shares = \
        _common_setup(gradients, cluster, rng)
    dropouts = set(dropouts)
    survivors = [c for c in cluster if c not in dropouts]

    # Pairwise pads sliced from the dealer pool: same OTP mechanics and
    # canonical (min, max) keying as the entangled arm.
    pair_bits, pair_masks, off = {}, {}, 0
    bpm = n_params * 32
    for i, ci in enumerate(cluster):
        for cj in cluster[i + 1:]:
            pair_bits[(ci, cj)] = dealer_pool[off:off + bpm]
            pair_masks[(ci, cj)] = derive_modular_mask(
                n_params, pair_bits[(ci, cj)])
            off += bpm

    agg = np.zeros(n_params, dtype=np.uint32)
    for c in survivors:
        p = quant[c].copy()
        for c2 in cluster:
            if c2 == c:
                continue
            lo, hi = (c, c2) if c < c2 else (c2, c)
            p = apply_modular_mask(p, pair_masks[(lo, hi)],
                                   sign=+1 if c < c2 else -1)
        p = apply_modular_mask(p, self_masks[c], sign=+1)
        agg = (agg + p).astype(np.uint32)

    # Recovery: the entangled arm's reveal mechanism; a pad has no
    # compact seed either.
    t0 = time.perf_counter()
    recovery_bytes = 0
    for d in dropouts:
        for s in survivors:
            lo, hi = (s, d) if s < d else (d, s)
            mask = derive_modular_mask(n_params, pair_bits[(lo, hi)])
            agg = ((agg - mask) if s < d else (agg + mask)).astype(np.uint32)
            recovery_bytes += n_params * BYTES_PER_REVEAL_WORD
    rec_s = time.perf_counter() - t0

    agg = _finish(cluster, dropouts, quant, self_masks, agg)

    # Each of the n(n-1)/2 pads is sent to both endpoints: n(n-1)*d*4 B.
    setup = n * (n - 1) * n_params * 4 if provisioning == "online" else 0
    cost = BaselineCost(
        protocol=f"it_dealer_{provisioning}", n_clients=n, n_params=n_params,
        n_dropped=len(dropouts),
        payload_bytes=len(survivors) * n_params * BYTES_PER_MASKED_WORD,
        selfmask_bytes=len(survivors) * BYTES_PER_SELFMASK_SEED,
        share_bytes=n * (n - 1) * BYTES_PER_SHAMIR_SHARE,
        recovery_bytes=recovery_bytes,
        aggregator_prg_expansions=0,
        aggregator_recovery_s=rec_s,
        assumption_class=("IT mask; computational provisioning (TLS)"
                          if provisioning == "online"
                          else "IT mask; trusted dealer"),
        hndl_immune_pairwise_leg=(provisioning == "courier"),
        dealer_setup_bytes=setup,
        trust_surface=("dealer holds all pads; TLS transit"
                       if provisioning == "online"
                       else "dealer holds all pads; pad storage between visits"),
        channel_adversary_tolerated=("active at round time; provisioning "
                                     "rests on TLS"
                                     if provisioning == "online"
                                     else "active at round time; provisioning "
                                          "out of band"))
    return agg, cost
