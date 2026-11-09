"""
Module 6: Harvest-now / decrypt-later (HNDL) transcript experiment.

Records the full classical transcript of two-mask aggregation rounds for
several pairwise-mask provisioning arms (classical DH/PRF seeds, pqc KEM
seeds, it_dealer_online, it_dealer_courier, entangled), then replays it
with every computational secret broken. Wire objects come from the same
primitives as the live protocol and the aggregate is asserted equal to
secure_aggregation_round's.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

try:
    from modules.m4_federated_learning import (
        quantize_gradient, dequantize_gradient, derive_modular_mask,
        apply_modular_mask, shamir_split, shamir_reconstruct,
        _SHAMIR_PRIME, QUANT_SCALE,
    )
    from modules.m2_post_processing import nist_monobit
except ImportError:  # pragma: no cover - package-relative fallback
    from .m4_federated_learning import (
        quantize_gradient, dequantize_gradient, derive_modular_mask,
        apply_modular_mask, shamir_split, shamir_reconstruct,
        _SHAMIR_PRIME, QUANT_SCALE,
    )
    from .m2_post_processing import nist_monobit


def _prg_mask(seed, n_params):
    """The protocol's PRG expansion (identical to m4's self-mask path)."""
    return (np.random.default_rng(int(seed))
            .integers(0, 1 << 32, size=n_params, dtype=np.uint64)
            .astype(np.uint32))


@dataclass
class TranscriptRound:
    """Everything a recording adversary captured for one cluster round,
    plus ground truth held separately for scoring."""
    round_idx: int
    cluster: List[int]
    dropouts: Set[int]
    # Recorded wire objects.
    masked_payloads: Dict[int, np.ndarray]          # survivors only
    seed_shares: Dict[int, list]                     # Shamir shares in flight
    disclosed_seeds: Dict[int, int]                  # survivors' self-mask seeds
    revealed_pair_bits: Dict[Tuple[int, int], np.ndarray]  # dropout reveals
    # classical and pqc arms: the per-pair mask seeds as recorded (DH or
    # KEM exchange). None in the pool arms.
    pair_mask_seeds: Dict[Tuple[int, int], int] = None
    # it_dealer_online only: the dealer's per-round pad distribution,
    # recorded as TLS ciphertexts. None elsewhere; the courier arm's pads
    # never touch a recorded channel.
    dealer_pad_bits: Dict[Tuple[int, int], np.ndarray] = None
    # Both dealer arms: the dealer's own archive of every pad. Not part
    # of the transcript; read only under provisioner_compromise=True.
    provisioner_pad_bits: Dict[Tuple[int, int], np.ndarray] = None
    # Ground truth for scoring, not in the transcript.
    truth_quant: Dict[int, np.ndarray] = field(default_factory=dict)


# Arms whose pairwise masks are one-time pads sliced from a pool (the
# classical arm PRG-expands seeds instead).
_POOL_ARMS = ("entangled", "it_dealer_online", "it_dealer_courier")


def run_transcript_round(round_idx, gradients, cluster, dropouts, arm,
                         pairwise_pool=None, pair_mask_seeds=None,
                         self_seeds=None):
    """One two-mask round, wire-faithful, recording the full transcript.

    Pool arms ('entangled', 'it_dealer_online', 'it_dealer_courier')
    slice pairwise masks from `pairwise_pool`; seed arms ('classical',
    'pqc') PRG-expand them from `pair_mask_seeds`. Only the online-dealer
    arm records its pad distribution in the transcript.

    Returns (TranscriptRound, aggregate_quantized) where the aggregate is
    the unmasked modular sum of survivor payloads (for the coherence
    assertion against secure_aggregation_round).
    """
    cluster = list(cluster)
    dropouts = set(dropouts)
    survivors = [c for c in cluster if c not in dropouts]
    n_params = gradients[cluster[0]].numel()
    bits_per_mask = n_params * 32

    quant = {c: quantize_gradient(gradients[c]) for c in cluster}

    # Pairwise masks: same iteration order and canonical (min, max) keying
    # as secure_aggregation_round step 2.
    pair_masks = {}
    pair_bits = {}
    off = 0
    for i_idx, ci in enumerate(cluster):
        for cj in cluster[i_idx + 1:]:
            key = (ci, cj)
            if arm in _POOL_ARMS:
                bits = pairwise_pool[off:off + bits_per_mask]
                off += bits_per_mask
                pair_masks[key] = derive_modular_mask(n_params, bits)
                pair_bits[key] = bits
            else:
                pair_masks[key] = _prg_mask(pair_mask_seeds[key], n_params)
    # Self-masks and Shamir shares, same expansion as m4 step 3.
    self_masks, shares = {}, {}
    for c in cluster:
        seed = int(self_seeds[c]) % _SHAMIR_PRIME
        self_masks[c] = _prg_mask(seed, n_params)
        shares[c] = shamir_split(seed, len(cluster), 2,
                                 np.random.default_rng(seed + 1))

    # Payloads, m4 step 4; survivors only (pre-payload drop timing).
    payloads = {}
    for c in survivors:
        p = quant[c].copy()
        for c2 in cluster:
            if c2 == c:
                continue
            lo, hi = (c, c2) if c < c2 else (c2, c)
            p = apply_modular_mask(p, pair_masks[(lo, hi)],
                                   sign=+1 if c < c2 else -1)
        p = apply_modular_mask(p, self_masks[c], sign=+1)
        payloads[c] = p

    # Dropout reveals: each survivor reveals its pairwise bits with each
    # dropped partner (m4 step 6). In the seed arms the analogous reveal
    # is the pair seed; either suffices for the adversary, so the mask
    # source is recorded in both cases.
    revealed = {}
    for c in dropouts:
        for s in survivors:
            lo, hi = (s, c) if s < c else (c, s)
            if arm in _POOL_ARMS:
                revealed[(lo, hi)] = pair_bits[(lo, hi)]
            else:
                revealed[(lo, hi)] = np.int64(pair_mask_seeds[(lo, hi)])

    # Aggregator unmask (for the coherence check): sum payloads, remove
    # surviving self-masks, neutralize dropped clients' pairwise residues.
    agg = np.zeros(n_params, dtype=np.uint32)
    for c, p in payloads.items():
        agg = (agg + p).astype(np.uint32)
    for c in survivors:
        agg = (agg - self_masks[c]).astype(np.uint32)
    for c in dropouts:
        for s in survivors:
            lo, hi = (s, c) if s < c else (c, s)
            resid = pair_masks[(lo, hi)]
            agg = ((agg - resid) if s < c else (agg + resid)).astype(np.uint32)

    tr = TranscriptRound(
        round_idx=round_idx, cluster=cluster, dropouts=dropouts,
        masked_payloads=payloads, seed_shares=shares,
        disclosed_seeds={c: int(self_seeds[c]) % _SHAMIR_PRIME
                         for c in survivors},
        revealed_pair_bits=revealed,
        pair_mask_seeds=(dict(pair_mask_seeds)
                         if arm in ("classical", "pqc") else None),
        # The dealer's per-round pad distribution is on the wire under
        # computational encryption; the recorder keeps the ciphertexts.
        dealer_pad_bits=(dict(pair_bits) if arm == "it_dealer_online"
                         else None),
        # Whatever the delivery channel, the dealer generated these pads
        # and can retain them, so its archive exists in both dealer arms.
        provisioner_pad_bits=(dict(pair_bits)
                              if arm in ("it_dealer_online",
                                         "it_dealer_courier") else None),
        truth_quant=quant,
    )
    return tr, agg


def retroactive_attack(transcript: List[TranscriptRound], arm,
                       provisioner_compromise=False):
    """Replay the recorded transcript with every computational secret
    broken (Shamir shares, self-mask seeds, dropout reveals in all arms;
    DH/PRF seeds in the classical arm; pad ciphertexts in the
    online-dealer arm) and attempt to unmask every recorded payload.
    provisioner_compromise=True also hands over the dealer's own pad
    archive where one exists. Returns per-payload records and summary
    fractions.
    """
    records = []
    for tr in transcript:
        n_params = next(iter(tr.truth_quant.values())).shape[0]
        # Self-masks: reconstruct each owner's seed from any 2 recorded
        # shares (threshold 2) and PRG-expand with the protocol's primitive.
        selfmask = {}
        for owner, sh in tr.seed_shares.items():
            seed = shamir_reconstruct(sh[:2])
            selfmask[owner] = _prg_mask(seed, n_params)
            if owner in tr.disclosed_seeds:
                assert seed == tr.disclosed_seeds[owner], \
                    "share-derived seed != disclosed seed (transcript bug)"
        for c, payload in tr.masked_payloads.items():
            residual = (payload - selfmask[c]).astype(np.uint32)
            stripped = 0
            unstripped_pairs = []
            for c2 in tr.cluster:
                if c2 == c:
                    continue
                lo, hi = (c, c2) if c < c2 else (c2, c)
                sign = +1 if c < c2 else -1
                mask = None
                if arm == "classical":
                    mask = _prg_mask(tr.pair_mask_seeds[(lo, hi)], n_params)
                elif arm == "pqc":
                    # KEM ciphertexts stay closed under this adversary.
                    mask = None
                elif arm == "it_dealer_online":
                    # The recorded pad ciphertexts decrypt: every
                    # pairwise pad is in the transcript.
                    mask = derive_modular_mask(
                        n_params, tr.dealer_pad_bits[(lo, hi)])
                elif (provisioner_compromise
                      and tr.provisioner_pad_bits is not None
                      and (lo, hi) in tr.provisioner_pad_bits):
                    # Second adversary: the dealer's own archive.
                    mask = derive_modular_mask(
                        n_params, tr.provisioner_pad_bits[(lo, hi)])
                elif (lo, hi) in tr.revealed_pair_bits:
                    mask = derive_modular_mask(
                        n_params, tr.revealed_pair_bits[(lo, hi)])
                if mask is not None:
                    residual = apply_modular_mask(residual, mask, sign=-sign)
                    stripped += 1
                else:
                    unstripped_pairs.append((lo, hi))
            truth = tr.truth_quant[c]
            exact = bool(np.array_equal(residual, truth))
            # Residual uniformity: on arms with an unstripped pad this is
            # an implementation check only; the result is the recovered
            # fraction.
            res_bits = np.unpackbits(
                residual[:4096].view(np.uint8)).astype(np.int8)
            err = np.abs(dequantize_gradient(residual).numpy()
                         - dequantize_gradient(truth).numpy())
            records.append({
                "round": tr.round_idx, "client": c,
                "masks_stripped": stripped,
                "masks_remaining": len(unstripped_pairs),
                "exact_recovery": exact,
                "residual_monobit_p": nist_monobit(res_bits),
                "median_abs_error": float(np.median(err)),
            })
    n = len(records)
    return {
        "records": records,
        "n_payloads": n,
        "recovered_fraction": sum(r["exact_recovery"] for r in records) / n,
        "min_masks_remaining": min(r["masks_remaining"] for r in records),
        "mean_residual_monobit_p": float(np.mean(
            [r["residual_monobit_p"] for r in records])),
        "median_abs_error": float(np.median(
            [r["median_abs_error"] for r in records])),
    }
