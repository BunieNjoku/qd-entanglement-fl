"""
Module 7: Faithful classical secure-aggregation baselines (HANDOFF 4.1).

Implements the two classical protocols the secure-aggregation literature
compares against, at the level of fidelity that determines the honest
comparison axes (per-client bytes, aggregator recovery compute, dropout
recovery cost, assumption class, HNDL exposure). All three protocols —
this system and the two baselines — share the SAME Bonawitz-style
modular masking over Z_2^32, the SAME payload/self-mask wire format, and
the SAME FL-layer byte scoping (m4's BYTES_PER_* constants), so the
columns are matched. They differ ONLY in the pairwise-mask SOURCE and
the DROPOUT-RECOVERY mechanism:

  entangled (this system)
      pairwise mask = single-use source-model distilled bits (a true
      one-time pad; no compact seed). Dropout recovery on the pairwise
      leg: each survivor REVEALS its full pairwise bit-string with the
      dropped client (n_params x 4 B) — there is no seed to send, which
      is exactly the cost of unconditional freshness. Assumption class:
      unconditional. HNDL: the pairwise leg is immune (bits never transit
      a channel; reveals are single-use).

  bonawitz (SecAgg, Bonawitz et al. CCS 2017)
      pairwise mask = PRG expansion of a per-execution ephemeral
      Diffie-Hellman shared secret s_{i,j} = DH(sk_i, pk_j). Dropout
      recovery: threshold survivors Shamir-reconstruct the DROPPED
      client's DH secret key (one scalar), and the aggregator RECOMPUTES
      that client's pairwise masks with each survivor (PRG expansion).
      Recovery bytes are tiny (shares of one scalar) but the aggregator
      pays compute, and freshness/HNDL rest on DH + PRG being unbroken.
      Assumption class: computational.

  flamingo (Ma et al., IEEE S&P 2023)
      pairwise mask = PRG expansion of a per-round PRF seed
      h_{i,j,t} = PRF(r_{i,j}, t) from a long-term pairwise secret.
      Dropout recovery: survivors REVEAL the per-round PRF seed (one
      scalar) for the dropped partner; the aggregator PRG-expands and
      cancels. Same reveal-based logic as this system, but the revealed
      object is a computational SEED (4 B) not the mask itself, so
      recovery bytes are tiny — at the price of computational freshness
      (a future break of the PRF/long-term secret reopens the transcript,
      cf. exp9). Assumption class: computational.

DROP MODEL (identical across all three, the standard Bonawitz model):
a dropped client's payload does NOT reach the aggregator; survivors'
payloads do. Pairwise masks between two survivors cancel; the pairwise
mask between a survivor and a dropped client is uncancelled and must be
removed by the protocol's recovery mechanism. Each survivor's self-mask
is removed via a Shamir-reconstructed seed (the Bonawitz double-mask,
reconstructed from a threshold of OTHER survivors' shares — charged
identically in all three arms so it does not bias the comparison).

Each round is really executed and the recovered aggregate is asserted
equal to the plaintext sum of the survivors' quantized gradients, so a
byte/compute record can never silently diverge from a protocol that does
not actually reconstruct the correct value.
"""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set

try:
    from modules.m4_federated_learning import (
        quantize_gradient, derive_modular_mask, apply_modular_mask,
        shamir_split, shamir_reconstruct, _SHAMIR_PRIME,
        BYTES_PER_MASKED_WORD, BYTES_PER_SHAMIR_SHARE,
        BYTES_PER_SELFMASK_SEED,
    )
except ImportError:  # pragma: no cover
    from .m4_federated_learning import (
        quantize_gradient, derive_modular_mask, apply_modular_mask,
        shamir_split, shamir_reconstruct, _SHAMIR_PRIME,
        BYTES_PER_MASKED_WORD, BYTES_PER_SHAMIR_SHARE,
        BYTES_PER_SELFMASK_SEED,
    )

# Wire sizes for the recovered objects that differ across protocols.
BYTES_PER_DH_SHARE = BYTES_PER_SHAMIR_SHARE  # share of a dropped client's
#                                              DH secret scalar (Bonawitz)
BYTES_PER_PRF_SEED = 4   # per-round PRF pairwise seed revealed (Flamingo)
BYTES_PER_REVEAL_WORD = BYTES_PER_MASKED_WORD  # one revealed pairwise word
#                                                (this system): n_params of them


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
    assumption_class: str = ""      # "unconditional" | "computational"
    hndl_immune_pairwise_leg: bool = False

    def uplink_per_client(self):
        up = (self.payload_bytes + self.selfmask_bytes + self.share_bytes
              + self.recovery_bytes)
        return up / max(self.n_clients - self.n_dropped, 1)


def _common_setup(gradients, cluster, rng):
    cluster = list(cluster)
    n = len(cluster)
    n_params = gradients[cluster[0]].numel()
    quant = {c: quantize_gradient(gradients[c]) for c in cluster}
    self_seeds = {c: int(rng.integers(0, _SHAMIR_PRIME)) for c in cluster}
    self_masks = {c: _prg(self_seeds[c], n_params) for c in cluster}
    # Self-mask seeds are Shamir-shared (threshold 2) in ALL arms — the
    # double-mask, charged identically so it does not bias the comparison.
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

    # Recovery: each survivor REVEALS full pairwise bits with each dropped
    # partner; aggregator subtracts directly (no PRG expansion).
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
        hndl_immune_pairwise_leg=True)
    return agg, cost


def run_bonawitz(gradients, cluster, dropouts, rng):
    """Bonawitz SecAgg: DH-seeded pairwise PRG masks; dropout recovery by
    Shamir-reconstructing the dropped client's DH secret and recomputing."""
    cluster, n, n_params, quant, self_seeds, self_masks, shares = \
        _common_setup(gradients, cluster, rng)
    dropouts = set(dropouts)
    survivors = [c for c in cluster if c not in dropouts]

    # Per-execution ephemeral DH: model each client's secret scalar and
    # the pairwise shared secret as a deterministic function of the two
    # (symmetric), then PRG-expand. (We model DH agreement's RESULT — a
    # shared per-pair scalar — not the group operation; the comparison
    # axes depend on the recovery mechanism, not the DH group.)
    dh_secret = {c: int(rng.integers(1, _SHAMIR_PRIME)) for c in cluster}
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
    # threshold of survivors' shares (scalar), then RECOMPUTE and cancel
    # its pairwise masks with each survivor (PRG expansion = aggregator
    # compute).
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
    # Bonawitz uses PER-EXECUTION ephemeral DH (HANDOFF 4.1a); in a
    # multi-round FL run each round is an execution, so each round
    # redistributes n(n-1) Shamir shares of the fresh DH secrets IN
    # ADDITION to the self-mask seed shares. Charging both is faithful —
    # this per-round DH-setup share traffic is exactly the cost Flamingo
    # was designed to avoid with its long-term-secret + PRF ratchet, and
    # NOT counting it would undercount (flatter) Bonawitz (Gate-4 F-min1).
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
        hndl_immune_pairwise_leg=False)
    return agg, cost


def run_flamingo(gradients, cluster, dropouts, round_idx, rng):
    """Flamingo: per-round PRF pairwise seeds; dropout recovery by
    revealing the per-round PRF seed (scalar) for the dropped partner."""
    cluster, n, n_params, quant, self_seeds, self_masks, shares = \
        _common_setup(gradients, cluster, rng)
    dropouts = set(dropouts)
    survivors = [c for c in cluster if c not in dropouts]

    # Long-term pairwise secret r_{i,j}; per-round seed h = PRF(r, t).
    # PRF modeled as a keyed hash into the scalar field (the comparison
    # depends on the reveal being a per-round SEED, not on the PRF family).
    longterm = {}
    for i, ci in enumerate(cluster):
        for cj in cluster[i + 1:]:
            longterm[(ci, cj)] = int(rng.integers(1, _SHAMIR_PRIME))

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

    # Recovery: each survivor REVEALS the per-round PRF seed (scalar) for
    # the dropped partner; aggregator PRG-expands and cancels.
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
        hndl_immune_pairwise_leg=False)
    return agg, cost
