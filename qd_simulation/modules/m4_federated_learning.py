"""
Module 4: Federated learning with entanglement-derived masked aggregation
(pairwise one-time masks from source-model key distillation).

Implements FedAvg with two-mask Bonawitz pairwise masking (Bonawitz et al.
2017). Pairwise masks are derived from per-round entanglement-based shared
random bits (Module 1 source -> Module 2 reconciliation + PA). Self-masks
are drawn from local high-quality randomness and Shamir-shared across the
cluster for dropout recovery.

Modes:
  - entangled:       pairwise masks from shared Bell-pair bits (source-model
                     distilled keys); self-mask Shamir-shared.
  - gaussian-legacy: float Gaussian masks - NO Bonawitz proof; retained for
                     comparison only.
  - prng:     identical structure but classical PRNG seeds (baseline)
  - unmasked: no privacy protection (accuracy ceiling)

Masking law:
  Modular OTP on int32-quantized gradients. Each pairwise mask is uniform
  over Z_{2^32} derived directly from shared bits; aggregation is modular
  addition. This is the construction under which Bonawitz's privacy proof
  applies. Float Gaussian masks (legacy) are retained only for comparison
  with prior literature and DO NOT inherit the Bonawitz privacy guarantee.

Dropout recovery:
  Entanglement-derived pairwise bits are single-use per round. If client j
  drops, surviving clients may safely reveal their pairwise bits with j to
  the aggregator (no future rounds compromised). Self-masks are
  reconstructed via Shamir threshold over the surviving set; the pairwise
  leg needs no Shamir machinery. The same reveal-based per-round recovery
  exists classically with computational per-round seeds (Flamingo; Ma,
  Woods, Angel, Polychroniadou & Rabin, IEEE S&P 2023). The differentiator
  here is that freshness is UNCONDITIONAL: revealed bits are
  information-theoretically independent of every other round - no PRF/DH
  assumption, no key-erasure or forward-secrecy reliance.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


# MODELS

class ClinicalMLP(nn.Module):
    """Small MLP for tabular clinical data. ~50K parameters."""
    def __init__(self, input_dim=30, hidden=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# DATASETS

def load_clinical_data(num_clients=3, rng_seed=42, dirichlet_alpha=0.5):
    """Load sklearn breast cancer dataset, split non-IID across clients.

    Uses Dirichlet allocation (standard in FL literature) for heterogeneous
    label distributions. Lower alpha = more heterogeneity.
    """
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rng_seed, stratify=y
    )

    # Dirichlet non-IID split: each client gets a different label proportion
    rng = np.random.default_rng(rng_seed)
    n_classes = len(np.unique(y_train))
    client_indices = [[] for _ in range(num_clients)]

    for k in range(n_classes):
        class_indices = np.where(y_train == k)[0]
        rng.shuffle(class_indices)
        # Draw proportions from Dirichlet distribution
        proportions = rng.dirichlet(np.repeat(dirichlet_alpha, num_clients))
        # Ensure at least 1 sample per client for this class if possible
        proportions = np.maximum(proportions, 1.0 / len(class_indices))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for c, chunk in enumerate(np.split(class_indices, splits)):
            client_indices[c].extend(chunk.tolist())

    # Guard (found by exp5's 10-seed sweep, ): at some seeds the
    # cumsum split boundaries round so that a client ends up EMPTY - the
    # empty list became a float64 array (IndexError) and an empty client is
    # a degenerate FL participant. Move one sample from the largest client;
    # no-op at seeds where every client is non-empty.
    for c in range(num_clients):
        if len(client_indices[c]) == 0:
            donor = max(range(num_clients),
                        key=lambda j: len(client_indices[j]))
            client_indices[c].append(client_indices[donor].pop())

    client_datasets = []
    for indices in client_indices:
        indices = np.array(indices, dtype=np.int64)
        rng.shuffle(indices)
        client_datasets.append(TensorDataset(
            torch.tensor(X_train[indices]),
            torch.tensor(y_train[indices])
        ))

    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    return client_datasets, test_dataset, X_train.shape[1]


# MODULAR OTP MASKING (Bonawitz-compatible)
#
# Bonawitz 2017's privacy proof requires masks uniform over the gradient
# group. We quantize gradients to int32 (Q-format, see QUANT_SCALE) and
# perform masking in Z_{2^32} via uint32 modular addition. Shared bits
# from the distilled pool are sliced directly into uint32 mask words.
#
# Modular wraparound is intentional and correct: the masked aggregate is
# computed in the same ring and dequantized after summation. Per-pair masks
# cancel exactly because the same bits are derived deterministically at
# both endpoints from the shared bit string.

QUANT_SCALE = 1 << 20  # 20-bit fractional precision; gradients clipped to +-2^11

def quantize_gradient(grad_tensor, scale=QUANT_SCALE):
    """Map a float tensor to int32 (modular ring Z_{2^32}) via fixed-point."""
    g = grad_tensor.detach().cpu().numpy().astype(np.float64)
    g_clipped = np.clip(g * scale, -(2**31 - 1), 2**31 - 1)
    return g_clipped.astype(np.int64).astype(np.uint32)

def dequantize_gradient(quant_arr, scale=QUANT_SCALE):
    """Invert quantize_gradient. Re-interprets uint32 as signed int32."""
    signed = quant_arr.view(np.int32).astype(np.float64)
    return torch.tensor(signed / scale, dtype=torch.float32)

def bits_to_uint32(bits):
    """Convert a flat array of {0,1} bits into uint32 words (little-endian)."""
    n_words = len(bits) // 32
    if n_words == 0:
        return np.zeros(0, dtype=np.uint32)
    b = bits[:n_words * 32].astype(np.uint32).reshape(n_words, 32)
    powers = (1 << np.arange(32, dtype=np.uint32))
    return (b * powers).sum(axis=1).astype(np.uint32)

def derive_modular_mask(n_params, shared_bits):
    """Derive a uint32 mask of length n_params from shared distilled bits.

    Each parameter consumes 32 bits -> one uint32 mask word. The mask is
    uniform over Z_{2^32} when the shared bits are uniform (Leftover Hash
    Lemma after Toeplitz PA). Bonawitz's privacy proof applies directly.

    Raises ValueError if insufficient bits - caller must size the pool.
    """
    bits_needed = n_params * 32
    if len(shared_bits) < bits_needed:
        raise ValueError(
            f"derive_modular_mask: have {len(shared_bits)} bits, need {bits_needed}. "
            "Pool exhausted - increase source duration or reduce model size."
        )
    return bits_to_uint32(shared_bits[:bits_needed])

def apply_modular_mask(quant_grad, mask, sign=+1):
    """Apply mask to a quantized gradient under modular addition."""
    if sign > 0:
        return (quant_grad + mask).astype(np.uint32)
    return (quant_grad - mask).astype(np.uint32)


# SHAMIR SECRET SHARING (for self-mask recovery)
#
# (t, n) threshold sharing over GF(2^127 - 1) (Mersenne prime). Used to
# protect each client's self-mask seed so survivors can reconstruct it
# if the client drops; honest-but-curious aggregator with < t shares
# learns nothing about the secret.

# Mersenne prime. 2^31 - 1 leaves a self-mask seed brute-forceable in
# 2^31 PRG expansions, which is not a security level - the leg would be
# protected in name only. 2^127 - 1 puts it beyond exhaustive search.
# All Shamir arithmetic here is Python integer modular arithmetic, so
# the field size costs nothing but share width.
_SHAMIR_PRIME = (1 << 127) - 1

def shamir_field_element(rng):
    """Uniform draw from the Shamir field.

    numpy's integers() is bounded by int64, so a 127-bit field has to be
    sampled from raw bytes. Rejection-free: the modulus loses at most
    2^-1 bits of uniformity at this width, far below any term in the
    security accounting.
    """
    nbytes = (_SHAMIR_PRIME.bit_length() + 7) // 8
    return int.from_bytes(bytes(rng.bytes(nbytes)), 'big') % _SHAMIR_PRIME


def _shamir_eval(coeffs, x):
    y = 0
    for c in reversed(coeffs):
        y = (y * x + c) % _SHAMIR_PRIME
    return y

def shamir_split(secret_int, n_shares, threshold, rng):
    """Split secret_int into n_shares; any `threshold` shares can recover."""
    if secret_int >= _SHAMIR_PRIME:
        raise ValueError("secret must be < _SHAMIR_PRIME")
    coeffs = [int(secret_int)] + [shamir_field_element(rng)
                                  for _ in range(threshold - 1)]
    return [(x, _shamir_eval(coeffs, x)) for x in range(1, n_shares + 1)]

def shamir_reconstruct(shares):
    """Reconstruct secret from any subset of shares (Lagrange at x=0)."""
    secret = 0
    p = _SHAMIR_PRIME
    for j, (xj, yj) in enumerate(shares):
        num, den = 1, 1
        for m, (xm, _) in enumerate(shares):
            if m == j:
                continue
            num = (num * (-xm)) % p
            den = (den * (xj - xm)) % p
        secret = (secret + yj * num * pow(den, -1, p)) % p
    return secret


def generate_gaussian_mask_legacy(n_params, shared_key_bits):
    """LEGACY - float Gaussian masks; NO Bonawitz privacy proof attaches.

    Retained only for historical comparison (Experiment 5b appendix).
    Use derive_modular_mask() for the proof-carrying modular construction.

    Generates a mask directly from quantum-derived key bits.

    Converts shared random bits to Gaussian-distributed mask values using the
    Box-Muller transform on uniform variates derived from the key material.
    This uses the quantum bits directly - not as a PRNG seed - preserving
    the high-entropy guarantee from the key-distillation (PA) layer.

    If insufficient bits are available, the remaining mask entries are zero
    (in a real deployment, the round would block until enough key material
    accumulates).

    Args:
        n_params: Number of gradient parameters to mask.
        shared_key_bits: np.ndarray of {0,1} bits from post-processed quantum keys.

    Returns:
        mask: torch.Tensor of shape (n_params,) with Gaussian-distributed values.
    """
    # Each Gaussian sample needs 2 uniform variates (Box-Muller).
    # Each uniform variate is constructed from 16 bits -> resolution 1/65536.
    bits_per_uniform = 16
    bits_per_sample = 2 * bits_per_uniform  # 32 bits per Gaussian sample
    bits_needed = n_params * bits_per_sample

    key_bits = shared_key_bits
    if len(key_bits) < bits_needed:
        # Pad - in practice this means the key rate limits round speed
        key_bits = np.concatenate([
            key_bits,
            np.zeros(bits_needed - len(key_bits), dtype=np.int8)
        ])

    # Reshape to (n_params, 2, 16): two 16-bit uniform variates per sample
    bits_block = key_bits[:bits_needed].reshape(n_params, 2, bits_per_uniform)

    # Convert 16-bit groups to uniform floats in (0, 1)
    powers = 2.0 ** np.arange(bits_per_uniform - 1, -1, -1)
    u_int = (bits_block.astype(np.float64) * powers).sum(axis=2)  # (n_params, 2)
    u = (u_int + 0.5) / 65536.0  # Avoid exact 0 and 1

    # Box-Muller transform: (u1, u2) -> Gaussian
    r = np.sqrt(-2.0 * np.log(u[:, 0]))
    theta = 2.0 * np.pi * u[:, 1]
    gaussian = r * np.cos(theta)

    return torch.tensor(gaussian, dtype=torch.float32)


def generate_prng_mask(n_params, seed):
    """Generate PRNG mask (classical baseline)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(n_params, generator=g)


# FEDERATED LEARNING ENGINE

# Communication byte model
# Documented wire-format assumptions behind every bytes/client/round
# figure. Reviews audit these against the protocol steps in
# secure_aggregation_round; change them only together with the protocol.
BYTES_PER_MASKED_WORD = 4    # one uint32 word of masked payload
BYTES_PER_MODEL_WORD = 4     # one float32 word of broadcast global model
BYTES_PER_SHAMIR_SHARE = 18  # 16 B value in the 127-bit Mersenne prime
                             # field (_SHAMIR_PRIME = 2^127 - 1) + 2 B
                             # share index
BYTES_ABORT_NOTICE = 4       # live-set/abort announcement to each surviving
                             # member of a threshold-failed cluster. Drops
                             # are detected by non-arrival, so this notice
                             # follows payload collection: the survivor has
                             # already sent, and its payload IS charged.
BYTES_PER_SELFMASK_SEED = 16  # survivor's self-mask disclosure: the seed
                             # (< _SHAMIR_PRIME). The self-mask is
                             # PRG-expanded from this seed - which is
                             # exactly why the self-mask leg is only
                             # computationally protected (see the channel
                             # assumption in RoundComm's docstring);
                             # the pairwise leg carries the HNDL claim.


@dataclass
class RoundComm:
    """Per-round communication and latency accounting.

    Uplink phases (client -> aggregator), bytes summed over clients:
      payload           masked int32 gradients from every client that
                        sent. Dropped clients never send. Survivors of a
                        threshold-failed cluster DO send and are charged:
                        a client masks against all its partners before
                        anyone knows who dropped (that is why the reveal
                        leg exists), so it cannot withhold a payload on
                        the strength of an outcome not yet known.
      shares            per-round Shamir distribution - happens BEFORE
                        dropout, so all n members send n-1 shares of their
                        fresh self-mask seed
      selfmask          survivors' own seed disclosures
      recovery_shares   identically 0 under this simulation's pre-payload
                        drop timing: a dropped self-mask never enters the
                        sum, so Shamir reconstruction runs as VALIDATION
                        only (asserted, never charged as wire traffic -
                        protocol step 6). The field exists for the
                        m7 baselines' post-payload drop timing,
                        which does transmit recovery shares.
      reveals           each survivor reveals its full pairwise bit-string
                        with each dropped partner: n_paramsx4 B per
                        (survivor, dropped) pair - THE byte cost of
                        reveal-based recovery
      inter_triangle    masked triangle aggregates on tree edges
    Downlink: global-model broadcast to every client, plus a
    BYTES_ABORT_NOTICE live-set/abort notice to each surviving member of
    a threshold-failed cluster. The notice is inherent to the Bonawitz
    round structure and follows payload collection, since dropouts are
    detected by non-arrival.

    Latency decomposition: key_wait_s is MODELED from the config operating
    point - the SLOWEST physical edge's pool-fill time,
    max(key_bits_per_edge) / secret_rate (edges distill concurrently on
    dedicated per-edge optics, but a physical edge serving multiple
    logical demands - two triangles, or a triangle plus a tree mask -
    fills correspondingly slower; conservative single-distillation-plane
    model, even though m5's per-membership QD provisioning could
    parallelize shared edges). comm_s is MODELED from
    config.CLASSICAL_LINK_BPS; the others are measured wall-clock.

    Network model, stated once: FL bytes serialize through
    the aggregator's single CLASSICAL_LINK_BPS access port; quantum key
    distillation runs on per-physical-edge dedicated optics in parallel -
    two different resources, hence the different concurrency treatments.

    Channel assumption: Shamir shares and seed disclosures
    are assumed to transit CONFIDENTIAL pairwise channels (Bonawitz's
    setting); channel-encryption overhead is not charged. Without that
    confidentiality, any threshold-many recorded shares reconstruct a
    seed - degrading the self-mask leg from computationally-protected to
    broken. The pairwise (HNDL-carrying) leg is unaffected either way.

    Scope: key-DISTILLATION classical traffic is excluded
    from this FL-layer ledger by design.
    Magnitude at the operating point: reconciliation syndromes cost
    ~ 0.20 bits ~ 0.025 B per secret bit ((1-R_code)/yield = 0.16/0.809
    bits of syndrome per secret bit), ~ 23 KB per edge per round at the ~29K-param
    clinical model; PA (Toeplitz) seed traffic depends on the seed-reuse
    policy and is covered by the m6 HNDL transcript analysis. The
    exp12 comparison charges the classical baselines' key-agreement
    bytes under the same FL-layer scoping so the columns stay symmetric.
    """
    round_idx: int = 0
    n_clients: int = 0
    payload_bytes: int = 0
    share_bytes: int = 0
    selfmask_bytes: int = 0
    recovery_share_bytes: int = 0
    reveal_bytes: int = 0
    inter_triangle_bytes: int = 0
    downlink_bytes: int = 0
    key_bits_intra: int = 0
    key_bits_inter: int = 0
    key_bits_per_edge: Dict = field(default_factory=dict)
    threshold_failures: int = 0
    # Heaviest client's payload multiplicity (triangle memberships): the
    # per-client figures below are MEANS over heterogeneously loaded
    # clients; the heaviest client sends this many payloads per round.
    max_payload_multiplicity: int = 1
    aggregator_time_s: float = 0.0
    key_wait_s: float = 0.0
    local_compute_s: float = 0.0
    masking_s: float = 0.0
    comm_s: float = 0.0

    def uplink_bytes(self):
        """Total uplink bytes this round, all phases, all clients."""
        return (self.payload_bytes + self.share_bytes + self.selfmask_bytes
                + self.recovery_share_bytes + self.reveal_bytes
                + self.inter_triangle_bytes)

    def uplink_per_client(self):
        return self.uplink_bytes() / max(self.n_clients, 1)

    def recovery_bytes(self):
        """Bytes attributable to dropout recovery alone."""
        return self.recovery_share_bytes + self.reveal_bytes


@dataclass
class FLResult:
    """Results from one FL training run."""
    mode: str
    accuracies: List[float]
    losses: List[float]
    final_accuracy: float
    rounds_to_target: int
    wall_clock_s: float
    mask_gen_times_s: List[float]
    n_params: int
    quantum_bits_consumed: List[int] = field(default_factory=list)
    comm_rounds: List = field(default_factory=list)  # RoundComm per round
    # Aggregate-fidelity instrumentation (entangled mode; ):
    # per round, (a) count of uint32 words where the masked pipeline's
    # integer aggregate differs from the direct quantized sum - the
    # masking-exactness theorem says every entry must be 0 - and (b) the
    # float64 sup-norm error of the dequantized mean aggregate vs the
    # exact float mean, bounded by the 2^-20 truncation bound.
    agg_exact_mismatches: List[int] = field(default_factory=list)
    agg_error_inf: List[float] = field(default_factory=list)


def train_local(model, dataset, epochs, batch_size, lr, device='cpu'):
    """Train model on one client's local data for a few epochs."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()


def get_model_params(model):
    """Extract model parameters as a flat tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_model_params(model, flat_params):
    """Set model parameters from a flat tensor."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_params[offset:offset + n].view(p.shape))
        offset += n


def get_gradient_update(model_before, model_after):
    """Compute the gradient update (difference in parameters)."""
    params_before = get_model_params(model_before)
    params_after = get_model_params(model_after)
    return params_after - params_before


def evaluate(model, dataset, device='cpu'):
    """Evaluate model accuracy on a dataset."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    correct, total = 0, 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            total_loss += criterion(output, y_batch).item() * len(y_batch)
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += len(y_batch)

    return correct / max(total, 1), total_loss / max(total, 1)


_VALID_MODES = ("entangled", "gaussian-legacy", "prng", "unmasked")


def _validate_mode(mode):
    """Fail loudly on unknown mode names - no silent fallthrough, no aliasing."""
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. Valid modes: {', '.join(_VALID_MODES)}.")
    return mode


# Safety factor on the emitted-pair request: the coincidence count is
# a binomial draw and reconciliation drops whole frames, so ask for
# more pairs than the mean yield needs.
_SOURCE_HEADROOM = 1.5

# Runtime budget on one distilled pool. This is an engineering limit,
# not a physical one: the source-plus-distillation chain runs at about
# 3.2 Mbit/s here (measured over 5 Mbit and 20 Mbit draws, linear in
# bits), so 200 Mbit is roughly a minute of generation. Demands above
# it fall back to PRNG bits, and every caller is told which it got -
# a pool that is not distilled cannot carry an entangled-key claim, so
# the substitution is reported rather than hidden. Single-cell runs
# (exp4) sit under the budget and use real distilled key; the
# multi-cell sweeps do not, and say so.
DISTILLED_POOL_MAX_BITS = 200_000_000


def distilled_pool(n_bits, rng, label="pool", force_prng=False):
    """Draw `n_bits` of distilled key from the source and post-processing
    chain, rather than from a PRNG.

    Only a coincidence yields a raw bit and the waterfall then spends
    part of that on reconciliation, privacy amplification and the Bell
    duty, so the emitted-pair request is the bit demand divided by both
    factors. The result is Toeplitz-extracted: the Leftover Hash Lemma
    bound in m2 applies to the extractor output, not to reconciled bits.

    Returns (bits, source) where source is "distilled" or "prng". A
    demand above DISTILLED_POOL_MAX_BITS returns PRNG bits labelled as
    such: padding silently would leave the pool unable to carry an
    entangled-key claim while still looking like one.
    """
    if n_bits <= 0:
        # A single-cell mesh has no inter-cell masks to key.
        return np.zeros(0, dtype=np.int8), "none"
    if force_prng or n_bits > DISTILLED_POOL_MAX_BITS:
        return rng.integers(0, 2, size=n_bits, dtype=np.int8), "prng"
    try:
        from config import OPERATING_POINT, QUANTUM
        from modules.m1_quantum_source import simulate_source
        from modules.m2_post_processing import fast_ec_simulate, toeplitz_hash
    except ImportError:
        from ..config import OPERATING_POINT, QUANTUM
        from .m1_quantum_source import simulate_source
        from .m2_post_processing import fast_ec_simulate, toeplitz_hash

    emitted_per_s = (QUANTUM['source_rep_rate_hz']
                     * QUANTUM['source_brightness'])
    p_coincidence = OPERATING_POINT['raw_edge_rate_hz'] / max(emitted_per_s, 1.0)
    yield_frac = (OPERATING_POINT['secret_rate_bps']
                  / max(OPERATING_POINT['raw_edge_rate_hz'], 1.0))
    # Distil in chunks. simulate_source allocates several float64
    # arrays as long as the coincidence count, so asking it for a whole
    # pool at once costs gigabytes - 300 million coincidences peaked at
    # 6 GB and was enough to exhaust application memory across a sweep.
    # Chunking also matches how a deployment runs: the source feeds
    # reconciliation and privacy amplification continuously, in blocks,
    # rather than accumulating a pool and extracting once.
    try:
        from config import POST_PROC
    except ImportError:
        from ..config import POST_PROC
    block = int(POST_PROC.get('block_size', 50_000))
    out_per_block = max(1, int(block * yield_frac))
    chunk_raw = max(block * 40, 2_000_000)          # ~2 Mbit of raw per pass
    chunk_emitted = int((chunk_raw / max(p_coincidence, 1e-30))
                        * _SOURCE_HEADROOM)

    pieces, taken, passes = [], 0, 0
    max_passes = int(n_bits / max(out_per_block, 1) / 40) + 64
    while taken < n_bits and passes < max_passes:
        passes += 1
        src = simulate_source(
            n_pairs=chunk_emitted,
            rng=np.random.default_rng(rng.integers(2**31)),
            fidelity=QUANTUM['fidelity'],
            source_rep_rate_hz=QUANTUM['source_rep_rate_hz'],
            source_brightness=QUANTUM['source_brightness'],
            fiber_length_m=QUANTUM['fiber_length_m'],
            fiber_loss_db_per_km=QUANTUM['fiber_loss_db_per_km_780nm'],
            coupling_efficiency=QUANTUM['coupling_efficiency'],
            detector_efficiency=QUANTUM['detector_efficiency'],
            detector_dark_count_hz=QUANTUM['detector_dark_count_hz'],
            coincidence_window_ns=QUANTUM['coincidence_window_ns'],
            pbs_extinction_ratio=QUANTUM['pbs_extinction_ratio'],
            fiber_birefringence_rad_per_km_per_C=QUANTUM[
                'fiber_birefringence_rad_per_km_per_C'],
            temperature_variation_C=QUANTUM['temperature_variation_C'],
            colocated=QUANTUM['colocated'],
            local_optics_efficiency=QUANTUM['local_optics_efficiency'],
        )
        raw_a, raw_b = src.raw_bits_alice, src.raw_bits_bob
        if len(raw_a) < block:
            continue
        ec = fast_ec_simulate(raw_a, raw_b, qber_estimate=src.qber, rng=rng)
        off = 0
        for frame in ec:
            if frame.frame_success and frame.n_raw_bits >= block:
                pieces.append(toeplitz_hash(
                    raw_a[off:off + block], out_per_block, rng=rng))
                taken += out_per_block
            off += frame.n_raw_bits
            if taken >= n_bits:
                break
        del src, raw_a, raw_b, ec

    if taken < n_bits:
        raise RuntimeError(
            f"{label}: distilled {taken:,} bits in {passes} passes, short "
            f"of {n_bits:,}. Raise _SOURCE_HEADROOM or reduce the demand; "
            "do not pad with PRNG bits.")
    return np.concatenate(pieces)[:n_bits], "distilled"


def run_federated_training(
    model_fn, client_datasets, test_dataset,
    mode="entangled",  # "entangled" | "gaussian-legacy" | "prng" | "unmasked"
    num_rounds=100, local_epochs=2, batch_size=32, lr=0.01,
    target_accuracy=0.90,
    quantum_key_bits=None,  # Pre-generated distilled shared bits per round
    device='cpu',
    verbose=True,
    _report_pool_source=False,  # print pool provenance even when quiet
    seed=42,  # torch/np seed - vary across repeats for real replicates
):
    """Run complete federated learning training.

    Args:
        model_fn: Callable that returns a fresh model.
        client_datasets: List of client TensorDatasets.
        test_dataset: Test TensorDataset.
        mode: Masking mode.
        num_rounds: Number of training rounds.
        quantum_key_bits: If provided, use these bits for quantum masks.
        seed: seeds torch (model init, DataLoader order) and the driver's
            numpy generator. Passing the SAME seed to different modes
            gives paired runs (identical init/split/shuffling), which is
            what exp4's paired statistics require.

    Returns:
        FLResult with training metrics.
    """
    mode = _validate_mode(mode)
    num_clients = len(client_datasets)
    # Seed torch BEFORE the first model_fn(): the initial global model and
    # DataLoader shuffling otherwise inherit whatever global state the
    # process had, making training metrics irreproducible across suite
    # compositions and standalone runs.
    torch.manual_seed(seed)
    global_model = model_fn()
    n_params = sum(p.numel() for p in global_model.parameters())

    accuracies, losses, mask_gen_times = [], [], []
    quantum_bits_consumed = []
    agg_exact_mismatches, agg_error_inf = [], []
    rounds_to_target = num_rounds
    start_time = time.time()
    rng = np.random.default_rng(seed)

    # Pre-generate PRNG seeds for PRNG mode - symmetric: seed(a,b) == seed(b,a)
    prng_seeds = rng.integers(0, 2**31, size=(num_rounds, num_clients, num_clients))
    for r in range(num_rounds):
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                prng_seeds[r, j, i] = prng_seeds[r, i, j]

    # Pre-generate quantum key material for quantum mode.
    # In a real deployment these bits come from the key-distillation layer
    # (Module 1 source -> Module 2 EC+PA). Here we simulate the key pool with
    # bits drawn from the quantum source simulation, maintaining the pipeline.
    bits_per_mask = n_params * 32
    n_pairs = num_clients * (num_clients - 1) // 2  # pairwise links
    total_bits_needed = num_rounds * n_pairs * bits_per_mask
    if mode in ("gaussian-legacy", "entangled"):
        if quantum_key_bits is not None:
            assert len(quantum_key_bits) >= total_bits_needed, (
                f"Provided pairwise key pool too small: "
                f"{len(quantum_key_bits)} bits < required "
                f"{total_bits_needed}. Pre-size to avoid reuse.")
            quantum_pool = quantum_key_bits
        else:
            # Source -> reconciliation -> privacy amplification, the same
            # chain the multi-cell driver uses. It raises rather than
            # padding, so a pool that cannot be distilled never passes
            # itself off as one that was.
            quantum_pool, _pool_src = distilled_pool(
                total_bits_needed, rng, f"{mode} pairwise pool")
            if verbose or _report_pool_source:
                print(f"      pairwise mask pool: {_pool_src}")

    # Pool-reuse hard audit on the security-bearing modular
    # OTP path. The legacy Gaussian path is not covered by the Bonawitz
    # proof and is excluded by scope.
    pool_audit = (PoolAuditor("pairwise", len(quantum_pool))
                  if mode == "entangled" else None)

    # Communication accounting; modeled-latency inputs
    # from the config single source of truth (lazy import, m5-style).
    comm_rounds = []
    try:
        from config import (OPERATING_POINT as _OP,
                            CLASSICAL_LINK_BPS as _LINK_BPS)
    except ImportError:
        from ..config import (OPERATING_POINT as _OP,
                              CLASSICAL_LINK_BPS as _LINK_BPS)
    _secret_rate_bps = _OP['secret_rate_bps']

    for round_idx in range(num_rounds):
        round_mask_time = 0
        client_updates = []
        global_params = get_model_params(global_model)

        # Local training
        _t_local0 = time.perf_counter()
        for c in range(num_clients):
            local_model = model_fn()
            set_model_params(local_model, global_params.clone())
            train_local(local_model, client_datasets[c], local_epochs, batch_size, lr, device)
            update = get_gradient_update(global_model, local_model)
            client_updates.append(update)
        _local_compute_s = time.perf_counter() - _t_local0

        # Masking & Aggregation
        mask_start = time.time()

        if mode == "entangled":
            # Modular OTP masking on int32-quantized gradients via secure_aggregation_round.
            # Pool is pre-sized -> wrap-around assertion cannot fire.
            round_off = round_idx * n_pairs * bits_per_mask
            pair_pool = _slice_pool(quantum_pool, round_off,
                                    n_pairs * bits_per_mask)
            client_grad_dict = {c: client_updates[c] for c in range(num_clients)}
            local_seeds = {c: shamir_field_element(rng)
                            for c in range(num_clients)}
            agg_res = secure_aggregation_round(
                client_gradients=client_grad_dict,
                cluster_clients=list(range(num_clients)),
                pairwise_pool=pair_pool,
                local_rng_seeds=local_seeds,
                dropouts=set(),
                threshold=max(2, (num_clients + 1) // 2),
                aggregator_rng=rng,
                pool_auditor=pool_audit,
                pool_base_offset=round_off,
            )
            aggregate = agg_res.aggregate / num_clients
            quantum_bits_consumed.append(agg_res.bits_consumed_pairwise)

            # Aggregate-fidelity instrumentation
            # (a) Masking exactness at the INTEGER level: the unmasked
            #     modular sum from the masked pipeline must equal the
            #     direct quantized sum word-for-word (theorem - pairwise
            #     masks cancel exactly in Z_2^32). Zero float rounding in
            #     this comparison.
            _direct_q = np.zeros(n_params, dtype=np.uint32)
            for _c in range(num_clients):
                _direct_q = (_direct_q
                             + quantize_gradient(client_updates[_c])
                             ).astype(np.uint32)
            agg_exact_mismatches.append(
                int((agg_res.aggregate_quantized != _direct_q).sum()))
            # (b) Quantization error of the mean aggregate, measured in
            #     float64 against the exact float mean. Truncation-based
            #     quantization (int cast toward zero) bounds each
            #     client's per-parameter error by 1/QUANT_SCALE, so the
            #     mean-of-n error is < 2^-20 per parameter.
            _direct64 = torch.stack(client_updates).to(torch.float64).mean(dim=0)
            _deq64 = torch.from_numpy(
                _direct_q.view(np.int32).astype(np.float64)
                / QUANT_SCALE / num_clients)
            agg_error_inf.append(
                float((_deq64 - _direct64).abs().max()))

        elif mode == "gaussian-legacy":
            # Legacy: Box-Muller Gaussian masks. Retained for comparison only;
            # NOT covered by Bonawitz's privacy proof. Use mode="entangled" instead.
            pair_masks = {}
            round_bits_used = 0
            pair_idx = 0
            for i in range(num_clients):
                for j in range(i + 1, num_clients):
                    pool_offset = (round_idx * n_pairs + pair_idx) * bits_per_mask
                    pool_end = pool_offset + bits_per_mask
                    assert pool_end <= len(quantum_pool), (
                        f"Pairwise key pool exhausted at round {round_idx} pair ({i},{j}). "
                        "Increase source duration."
                    )
                    pair_bits = quantum_pool[pool_offset:pool_end]
                    pair_masks[(i, j)] = generate_gaussian_mask_legacy(n_params, pair_bits)
                    round_bits_used += bits_per_mask
                    pair_idx += 1
            masked_updates = []
            for c in range(num_clients):
                masked = client_updates[c].clone()
                for c2 in range(num_clients):
                    if c2 == c:
                        continue
                    canon = (min(c, c2), max(c, c2))
                    mask = pair_masks[canon]
                    masked = masked + mask if c < c2 else masked - mask
                masked_updates.append(masked)
            quantum_bits_consumed.append(round_bits_used)
            aggregate = torch.stack(masked_updates).mean(dim=0)

        elif mode == "prng":
            # Classical PRNG pairwise masking (Bonawitz baseline)
            masked_updates = []
            for c in range(num_clients):
                masked = client_updates[c].clone()
                for c2 in range(num_clients):
                    if c2 == c:
                        continue
                    pair_seed = int(prng_seeds[round_idx, min(c,c2), max(c,c2)])
                    mask = generate_prng_mask(n_params, pair_seed)
                    if c < c2:
                        masked = masked + mask
                    else:
                        masked = masked - mask
                masked_updates.append(masked)
            aggregate = torch.stack(masked_updates).mean(dim=0)

        elif mode == "unmasked":
            aggregate = torch.stack(client_updates).mean(dim=0)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        round_mask_time = time.time() - mask_start
        mask_gen_times.append(round_mask_time)

        # Round communication/latency record
        if mode == "entangled":
            cc = agg_res.comm
            rc = RoundComm(
                round_idx=round_idx, n_clients=num_clients,
                payload_bytes=cc["payload_bytes"],
                share_bytes=cc["share_bytes"],
                selfmask_bytes=cc["selfmask_bytes"],
                recovery_share_bytes=cc["recovery_share_bytes"],
                reveal_bytes=cc["reveal_bytes"],
                key_bits_per_edge=dict(cc["key_bits_per_edge"]),
                key_bits_intra=sum(cc["key_bits_per_edge"].values()),
                aggregator_time_s=cc["aggregator_time_s"],
            )
            # Slowest-edge pool-fill; single triangle -> all
            # edges carry exactly bits_per_mask, so this equals the old
            # per-edge formula here, but stays correct if topologies grow.
            rc.key_wait_s = (max(rc.key_bits_per_edge.values())
                             / _secret_rate_bps
                             if rc.key_bits_per_edge else 0.0)
            # Distinguisher instrumentation excluded from every latency
            # bucket.
            rc.masking_s = max(0.0, round_mask_time
                               - rc.aggregator_time_s
                               - cc["instrumentation_time_s"])
        else:
            # Non-entangled modes: payload + downlink only. The prng mode
            # as implemented PRE-SHARES symmetric seeds (no per-round share
            # traffic), so its accounting here is NOT a faithful classical
            # secure-aggregation baseline - the m7 Bonawitz-DH and
            # Flamingo-PRF baselines carry the faithful byte models.
            rc = RoundComm(
                round_idx=round_idx, n_clients=num_clients,
                payload_bytes=(num_clients * n_params
                               * BYTES_PER_MASKED_WORD),
            )
            rc.masking_s = round_mask_time
        rc.downlink_bytes = num_clients * n_params * BYTES_PER_MODEL_WORD
        rc.local_compute_s = _local_compute_s
        rc.comm_s = ((rc.uplink_bytes() + rc.downlink_bytes) * 8
                     / _LINK_BPS)
        comm_rounds.append(rc)

        # Apply aggregate update
        new_params = global_params + aggregate
        set_model_params(global_model, new_params)

        # Evaluate
        acc, loss = evaluate(global_model, test_dataset, device)
        accuracies.append(acc)
        losses.append(loss)

        if acc >= target_accuracy and rounds_to_target == num_rounds:
            rounds_to_target = round_idx + 1

        if verbose and (round_idx + 1) % 10 == 0:
            print(f"  [{mode}] Round {round_idx+1}/{num_rounds}: "
                  f"accuracy={acc:.4f}, loss={loss:.4f}")

    wall_clock = time.time() - start_time

    # Pool-reuse hard audit: every pairwise-mask slice of the
    # run must be disjoint and account for exactly the ledgered burn.
    if mode == "entangled" and pool_audit is not None:
        assert pool_audit.bits_consumed == sum(quantum_bits_consumed), (
            f"pool audit mismatch: {pool_audit.bits_consumed} audited != "
            f"{sum(quantum_bits_consumed)} ledgered")
        assert pool_audit.n_intervals == num_rounds * n_pairs, (
            f"pool audit: expected {num_rounds * n_pairs} disjoint slices, "
            f"saw {pool_audit.n_intervals}")

    return FLResult(
        mode=mode,
        accuracies=accuracies,
        losses=losses,
        final_accuracy=accuracies[-1] if accuracies else 0,
        rounds_to_target=rounds_to_target,
        wall_clock_s=wall_clock,
        mask_gen_times_s=mask_gen_times,
        n_params=n_params,
        quantum_bits_consumed=quantum_bits_consumed,
        comm_rounds=comm_rounds,
        agg_exact_mismatches=agg_exact_mismatches,
        agg_error_inf=agg_error_inf,
    )


# TWO-MASK AGGREGATION ROUND (modular OTP, dropout-aware; entangled-key pool)

@dataclass
class AggregateResult:
    """Output of one two-mask aggregation round."""
    aggregate: torch.Tensor          # Unmasked sum of survivor gradients
    n_survivors: int
    n_dropped: int
    bits_consumed_pairwise: int
    bits_consumed_self: int          # local QRNG bits per survivor
    recovery_invoked: bool
    eavesdrop_distinguishable: bool  # honest-but-curious aggregator gain
    comm: Dict = field(default_factory=dict)  # per-phase bytes + agg time
    # Integer-domain modular sum AFTER unmasking - lets callers verify
    # masking exactness (masked pipeline == direct quantized sum, word
    # for word) without float rounding in the comparison.
    aggregate_quantized: object = None  # np.uint32 array


def _slice_pool(pool, offset, n_bits, auditor=None, base_offset=0):
    """Slice n_bits from pool starting at offset. Raises if exhausted.

    When `auditor` is given, the ABSOLUTE range
    [base_offset + offset, base_offset + offset + n_bits) is registered
    with it, and any overlap with a previously consumed range raises -
    the pool-reuse hard audit. base_offset locates a
    sub-pool view (e.g. one cluster's slice) inside its parent pool so
    all consumption is audited in parent-absolute coordinates.
    """
    if offset + n_bits > len(pool):
        raise ValueError(
            f"Pairwise key pool exhausted at offset {offset} (need {n_bits}, "
            f"have {len(pool) - offset}). Pre-size the pool to avoid bit reuse."
        )
    if auditor is not None:
        auditor.consume(base_offset + offset, n_bits)
    return pool[offset:offset + n_bits]


class PoolAuditor:
    """Hard audit that no distilled-key pool offset is consumed twice
    across a full run.

    OTP integrity is binary: a single reused mask bit voids the
    information-theoretic claim, so overlap is a hard ValueError, not a
    warning. Consumed [start, end) intervals are kept sorted; each
    consume() is checked against both neighbors (O(log n) locate +
    O(n) insert - trivial at simulation scale).

    Discarded-but-reserved material (a threshold-failed cluster's slice)
    is also registered: single-use discipline burns it, so any later
    attempt to reuse those offsets must fail the audit too.
    """

    def __init__(self, name, pool_len):
        self.name = name
        self.pool_len = int(pool_len)
        self._starts = []
        self._ends = []

    def consume(self, offset, n_bits):
        import bisect
        start, end = int(offset), int(offset) + int(n_bits)
        if n_bits <= 0 or start < 0 or end > self.pool_len:
            raise ValueError(
                f"pool '{self.name}': invalid slice [{start}, {end}) of a "
                f"{self.pool_len}-bit pool")
        i = bisect.bisect_left(self._starts, start)
        if i > 0 and self._ends[i - 1] > start:
            raise ValueError(
                f"pool '{self.name}': OTP REUSE - slice [{start}, {end}) "
                f"overlaps already-consumed "
                f"[{self._starts[i-1]}, {self._ends[i-1]})")
        if i < len(self._starts) and self._starts[i] < end:
            raise ValueError(
                f"pool '{self.name}': OTP REUSE - slice [{start}, {end}) "
                f"overlaps already-consumed "
                f"[{self._starts[i]}, {self._ends[i]})")
        self._starts.insert(i, start)
        self._ends.insert(i, end)

    @property
    def bits_consumed(self):
        return sum(e - s for s, e in zip(self._starts, self._ends))

    @property
    def n_intervals(self):
        return len(self._starts)

    def summary(self):
        return {"pool": self.name, "disjoint_slices": self.n_intervals,
                "bits_consumed": self.bits_consumed, "reuse_detected": False}


def secure_aggregation_round(
    client_gradients,           # dict: client_idx -> torch.Tensor (flat)
    cluster_clients,            # list of client indices in this cluster
    pairwise_pool,              # distilled key bits for this cluster, this round
    local_rng_seeds,            # dict: client_idx -> int (local QRNG seed)
    dropouts=None,              # set of client indices that drop after sending
    threshold=None,             # Shamir threshold for self-mask recovery
    aggregator_rng=None,        # used for honest-but-curious distinguisher trial
    pool_auditor=None,          # PoolAuditor: pool-reuse hard audit
    pool_base_offset=0,         # pairwise_pool's absolute offset in the parent pool
):
    """One two-mask (Bonawitz-style) aggregation round with dropout recovery.

    Pairwise masks come from an entanglement-derived pool (source-model
    distilled keys); the round itself is mask-source-agnostic.

    Protocol:
      (1) Each client quantizes its gradient to int32 (modular ring Z_{2^32}).
      (2) For each unordered pair (i, j) in the cluster, both endpoints
          derive the same uint32 mask from shared distilled bits. Client i adds
          mask, client j subtracts mask. Pairwise sum cancels.
      (3) Each client adds a self-mask drawn from local QRNG (seed s_i).
          The seed is Shamir-shared (t-of-n) across the cluster so
          survivors can reconstruct it if the client drops.
      (4) Each client sends one share of its OWN self-mask seed to each of
          its n-1 cluster peers (the n(n-1) x BYTES_PER_SHAMIR_SHARE wire
          charge models exactly these messages), then - after the live-set
          announcement - transmits its masked payload.
      (5) Aggregator sums the masked payloads (modular addition).
      (6) Dropout handling under THIS simulation's drop timing (dropouts
          occur AFTER share distribution, BEFORE payload transmission):
            - A dropped client's payload - and with it its self-mask -
              never entered the sum, so NO self-mask reconstruction is
              protocol-necessary here. The Shamir machinery is still
              distributed every round because the general Bonawitz timing
              (a client dropping AFTER its payload was sent) requires
              >= threshold survivors' shares to reconstruct the dropped
              self-mask seed; this simulation VALIDATES reconstruction
              (asserted below, not charged as protocol traffic) without
              exercising that timing - the m7 baselines do.
            - Each survivor discloses its own self-mask SEED (4 B; the
              mask is PRG-expanded from it - the computationally protected
              leg) so the aggregator can remove surviving self-masks.
            - For the dropped client's pairwise residue: each survivor
              reveals the pairwise bits it shared with the dropped client.
              Safe because bits are per-round-fresh and single-use -
              unconditionally so (cf. Flamingo, IEEE S&P 2023, where the
              same reveal logic rests on computational per-round PRF seeds).
      (7) Aggregator dequantizes the sum, returns aggregate.

    Honest-but-curious aggregator check - VALIDATION, not a security
    result: under uniform modular masks, payload
    indistinguishability from uniform is a THEOREM (one-time pad over
    Z_{2^32}); the chi-squared trial on the lowest byte is therefore an
    implementation-bug detector - it can only fire if the mask pipeline
    is broken (non-uniform bits, reuse, wrong modular arithmetic). A
    False result confirms the implementation matches the theorem; it
    adds no security beyond it.
    """
    if aggregator_rng is None:
        aggregator_rng = np.random.default_rng(0)
    dropouts = dropouts or set()
    survivors = [c for c in cluster_clients if c not in dropouts]
    n_cluster = len(cluster_clients)
    if threshold is None:
        threshold = max(2, (n_cluster + 1) // 2)  # majority threshold

    if not survivors:
        raise ValueError("All cluster members dropped; cannot aggregate.")
    if dropouts and len(survivors) < threshold:
        raise ValueError(
            f"secure_aggregation_round: only {len(survivors)} survivors, Shamir threshold is "
            f"{threshold}. Cannot reconstruct dropped clients' self-mask seeds. "
            "Caller should skip this cluster when survivors < threshold."
        )

    # Reference gradient shape
    sample_grad = client_gradients[cluster_clients[0]]
    n_params = sample_grad.numel()
    bits_per_mask = n_params * 32

    # Step 1: Quantize all gradients.
    quant = {c: quantize_gradient(client_gradients[c]) for c in cluster_clients}

    # Step 2: Pairwise modular masks from distilled bits.
    pair_offset = 0
    pair_masks = {}
    for i_idx, ci in enumerate(cluster_clients):
        for cj in cluster_clients[i_idx + 1:]:
            bits = _slice_pool(pairwise_pool, pair_offset, bits_per_mask,
                               auditor=pool_auditor,
                               base_offset=pool_base_offset)
            pair_offset += bits_per_mask
            mask = derive_modular_mask(n_params, bits)
            pair_masks[(ci, cj)] = mask  # canonical (min, max) order
    bits_pairwise = pair_offset

    # Step 3: Self-masks from local QRNG; Shamir shares of seeds.
    self_masks = {}
    seed_shares = {}  # owner_client -> list of (share_idx, share_value)
    for c in cluster_clients:
        seed = int(local_rng_seeds[c]) % _SHAMIR_PRIME
        local_rng = np.random.default_rng(seed)
        # local QRNG generates n_params uint32s directly
        self_masks[c] = local_rng.integers(0, 1 << 32, size=n_params, dtype=np.uint64).astype(np.uint32)
        # Shamir split the seed into one share per cluster member
        seed_shares[c] = shamir_split(seed, n_cluster, threshold,
                                       np.random.default_rng(seed + 1))

    # Step 4: Each client emits masked payload.
    #         payload_i = quant_i + sum_j sign(i,j)*pair_mask_{i,j} + self_mask_i
    masked_payloads = {}
    for c in cluster_clients:
        if c in dropouts:
            # Dropped clients never actually send.
            continue
        payload = quant[c].copy()
        for c2 in cluster_clients:
            if c2 == c:
                continue
            lo, hi = (c, c2) if c < c2 else (c2, c)
            mask = pair_masks[(lo, hi)]
            payload = apply_modular_mask(payload, mask, sign=+1 if c < c2 else -1)
        payload = apply_modular_mask(payload, self_masks[c], sign=+1)
        masked_payloads[c] = payload

    # Step 5: Aggregator sums.
    _t_agg0 = time.perf_counter()
    agg = np.zeros(n_params, dtype=np.uint32)
    for c, pl in masked_payloads.items():
        agg = (agg + pl).astype(np.uint32)
    _t_agg1 = time.perf_counter()

    # Honest-but-curious distinguisher trial - VALIDATION (implementation-
    # bug detector; indistinguishability itself is a theorem, see the
    # docstring). Instrumentation, not protocol - timed separately so it
    # lands in NO latency bucket.
    _t_chi0 = time.perf_counter()
    sample = masked_payloads[survivors[0]]
    distinguishable = _chi2_distinguishable(sample, aggregator_rng)
    _t_chi1 = time.perf_counter()

    # Step 6: Cancel masks.
    _t_agg2 = time.perf_counter()
    bits_self = 0
    recovery_invoked = bool(dropouts)
    for c in cluster_clients:
        if c in dropouts:
            # VALIDATION ONLY: under this simulation's drop
            # timing the dropped client's self-mask never entered the sum,
            # so this reconstruction is not protocol-necessary and its
            # share traffic is NOT charged. It is retained to assert the
            # Shamir machinery works (load-bearing in the general Bonawitz
            # timing, exercised by the m7 baselines).
            shares = []
            for survivor_idx, s in enumerate(survivors):
                if len(shares) >= threshold:
                    break
                # In real protocol, survivor s sends seed_shares[c][s_pos]
                # - here we model that survivor s holds the share at
                # position (s_pos+1) corresponding to its position.
                pos = cluster_clients.index(s)
                shares.append(seed_shares[c][pos])
            reconstructed_seed = shamir_reconstruct(shares)
            assert reconstructed_seed == (int(local_rng_seeds[c]) % _SHAMIR_PRIME), \
                "Shamir reconstruction failed - protocol bug"
            local_rng = np.random.default_rng(reconstructed_seed)
            dropped_self_mask = local_rng.integers(0, 1 << 32, size=n_params, dtype=np.uint64).astype(np.uint32)
            # Note: dropped client's pairwise masks are still in `agg` from
            # the +mask contributions of its surviving partners. We undo
            # those by replaying the appropriate sign from survivors.
            # Since survivors already applied sign(survivor, dropped) to their
            # payload, the dropped client's contribution is missing. The
            # surviving sum contains an extra mask term per pair.
            # Survivors reveal pairwise bits with dropped (safe: per-round-fresh)
            # -> aggregator subtracts their mask to neutralize the residual.
            for s in survivors:
                lo, hi = (s, c) if s < c else (c, s)
                resid = pair_masks[(lo, hi)]
                # Surviving partner s contributed sign(s, c)*resid to agg.
                # We subtract that sign-respecting contribution.
                if s < c:
                    agg = (agg - resid).astype(np.uint32)
                else:
                    agg = (agg + resid).astype(np.uint32)
            # Dropped client's self-mask never entered agg (it didn't send).
            # No subtraction needed for self-mask of dropped client.
        else:
            # Surviving client's self-mask must be removed.
            agg = (agg - self_masks[c]).astype(np.uint32)
            bits_self += n_params * 32

    # Step 7: Dequantize the modular sum.
    # Note: agg is sum of int32-quantized gradients in Z_{2^32}. Treat as
    # signed int32 for dequantization. Modular wraparound is theoretically
    # possible if many gradients sum to magnitudes > 2^31; in practice
    # gradient updates are small.
    aggregate = dequantize_gradient(agg)
    _t_agg3 = time.perf_counter()

    # Communication accounting for this cluster round.
    # Wire model documented at the BYTES_PER_* constants; phase semantics
    # at RoundComm. The distinguisher trial between _t_agg1 and _t_agg2 is
    # instrumentation, not protocol, and is excluded from aggregator time.
    comm = {
        "payload_bytes": len(survivors) * n_params * BYTES_PER_MASKED_WORD,
        "share_bytes": n_cluster * (n_cluster - 1) * BYTES_PER_SHAMIR_SHARE,
        "selfmask_bytes": len(survivors) * BYTES_PER_SELFMASK_SEED,
        # Zero under this simulation's pre-payload drop timing: the
        # dropped self-mask never entered the sum, so seed reconstruction
        # is validation, not protocol. Recovery costs reveals
        # only. The m7 baselines, which exercise post-payload drops,
        # charge their reconstruction shares.
        "recovery_share_bytes": 0,
        "reveal_bytes": (len(dropouts) * len(survivors)
                         * n_params * BYTES_PER_MASKED_WORD),
        "key_bits_per_edge": {(int(a), int(b)): bits_per_mask
                              for (a, b) in pair_masks},
        "aggregator_time_s": (_t_agg1 - _t_agg0) + (_t_agg3 - _t_agg2),
        "instrumentation_time_s": _t_chi1 - _t_chi0,
    }

    return AggregateResult(
        aggregate=aggregate,
        n_survivors=len(survivors),
        n_dropped=len(dropouts),
        bits_consumed_pairwise=bits_pairwise,
        bits_consumed_self=bits_self,
        recovery_invoked=recovery_invoked,
        eavesdrop_distinguishable=distinguishable,
        comm=comm,
        aggregate_quantized=agg.copy(),
    )


def _chi2_distinguishable(payload_uint32, rng, alpha=0.001):
    """Chi-squared test: is payload's lowest-byte distribution uniform?

    VALIDATION ONLY. Under modular OTP with uniform masks,
    indistinguishability is a theorem, so this test cannot produce a
    security finding - a True return means the implementation is broken
    (non-uniform mask bits, pool reuse, or wrong modular arithmetic),
    nothing more. Returns True if the null "payload is uniform" is
    rejected at significance alpha.
    """
    from scipy.stats import chisquare
    low_byte = (payload_uint32 & 0xFF).astype(np.int64)
    counts, _ = np.histogram(low_byte, bins=256, range=(-0.5, 255.5))
    if counts.sum() < 256:
        return False  # too few samples to test
    expected = counts.sum() / 256
    chi2_stat, p_value = chisquare(counts, [expected] * 256)
    return p_value < alpha


def define_triangle_topology(num_clients):
    """Define triangle topologies for different client counts.
    
    Each triangle has exactly 3 clients. Triangles may share edges (clients).
    Returns:
        triangles: List of (i, j, k) client index tuples.
        shared_edges: List of ((tri_a, tri_b), (client_i, client_j)) for shared edges.
        tree_edges: List of (parent_tri, child_tri) for aggregation tree.
    """
    if num_clients == 3:
        triangles = [(0, 1, 2)]
        shared_edges = []
        tree_edges = []
    
    elif num_clients == 6:
        # 4 unique nodes forming 2 overlapping triangles + 2 extra nodes in a 3rd triangle
        # T1: [0,1,2], T2: [1,2,3], T3: [3,4,5]
        # T1-T2 share edge (1,2). T2-T3 share node 3.
        triangles = [(0, 1, 2), (1, 2, 3), (3, 4, 5)]
        shared_edges = [((0, 1), (1, 2)), ((1, 2), (3,))]
        tree_edges = [(0, 1), (1, 2)]  # T0 is root, T1 child, T2 grandchild
    
    elif num_clients == 9:
        # 4 triangles in a vertex-sharing chain covering all 9 clients
        # T0: (0,1,2), T1: (2,3,4), T2: (4,5,6), T3: (6,7,8)
        # Shared vertices: 2 (T0-T1), 4 (T1-T2), 6 (T2-T3)
        triangles = [(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8)]
        shared_edges = [((0,), (2,)), ((1,), (4,)), ((2,), (6,))]
        tree_edges = [(0, 1), (1, 2), (2, 3)]
    
    elif num_clients == 12:
        # 4 triangles in a mesh
        triangles = [
            (0, 1, 2), (1, 2, 3),   # T1, T2 share edge (1,2)
            (3, 4, 5), (5, 6, 7),   # T3, T4 share node 5
            (7, 8, 9), (9, 10, 11), # T5, T6 share node 9
        ]
        shared_edges = [
            ((0, 1), (1, 2)), ((1, 2), (3,)),
            ((2, 3), (5,)), ((3, 4), (7,)), ((4, 5), (9,))
        ]
        tree_edges = [(0, 1), (0, 2), (2, 3), (3, 4), (4, 5)]
    else:
        # Generic: partition clients into groups of 3
        triangles = []
        for i in range(0, num_clients - 2, 3):
            triangles.append((i, i+1, i+2))
        # Chain tree
        tree_edges = [(i, i+1) for i in range(len(triangles)-1)]
        shared_edges = []
    
    return triangles, shared_edges, tree_edges


def topology_from_mesh(mesh_result):
    """Extract triangle topology and aggregation tree from a Module 3 MeshResult.

    Bridges the physical campus topology (Delaunay) to the FL aggregation layer,
    ensuring the logical triangle assignments match the physical entangled links.

    Args:
        mesh_result: MeshResult from m3_mesh_topology.analyze_mesh().

    Returns:
        triangles: List of (i, j, k) client index tuples.
        tree_edges: List of (parent_tri, child_tri) for aggregation tree.
    """
    triangles = [tuple(t) for t in mesh_result.triangles]
    # Extract tree edges from the directed aggregation tree
    tree_edges = list(mesh_result.aggregation_tree.edges())
    return triangles, tree_edges


def size_pools(triangles, tree_edges, num_rounds, n_params):
    """Compute required distilled-key pool sizes for a given topology and round count.

    Returns (bits_intra, bits_inter). Use these to pre-allocate the pools so
    the assertion in `_slice_pool` cannot fire (and bits cannot wrap around).
    """
    bits_per_mask = n_params * 32
    bits_intra = num_rounds * sum(3 for _ in triangles) * bits_per_mask
    bits_inter = num_rounds * len(tree_edges) * bits_per_mask
    return bits_intra, bits_inter


def run_multitriangle_fl(
    model_fn, client_datasets, test_dataset,
    num_clients, num_rounds=60, local_epochs=2,
    batch_size=32, lr=0.01, target_accuracy=0.92,
    triangles=None, tree_edges=None,
    edge_secret_rates=None,
    dropout_rate=0.0,
    pool_intra=None, pool_inter=None,
    device='cpu', verbose=True,
    seed=42,  # torch/np seed - vary across repeats for real replicates
):
    """two-mask hierarchical FL with two-mask design and dropout recovery.

    Differences from the legacy driver (`run_multitriangle_fl_legacy`):
      - Modular OTP masking on int32-quantized gradients (Bonawitz proof applies)
      - Pairwise masks derived from pre-sized distilled-key bit pool - wrap-around impossible
      - Inter-triangle masks ALSO from distilled bits (not classical PRNG)
      - Two-mask design: self-mask Shamir-shared for dropout recovery
      - Per-round dropouts sampled at `dropout_rate`
      - Honest-but-curious aggregator distinguisher reported per round
        (validation-only implementation-bug detector - see
        _chi2_distinguishable; indistinguishability is a theorem)
    """
    if triangles is None or tree_edges is None:
        # Generate the mesh rather than fall back to a hand-written
        # table. The tables encode topologies the Delaunay construction
        # does not produce - one joins two cells at a single vertex -
        # so a silent fallback would reintroduce them in any caller that
        # forgets to pass a topology.
        try:
            from modules.m3_mesh_topology import analyze_mesh
            from config import OPERATING_POINT
        except ImportError:
            from .m3_mesh_topology import analyze_mesh
            from ..config import OPERATING_POINT
        _m = analyze_mesh(num_clients, 1e5,
                          OPERATING_POINT['secret_rate_bps'],
                          rng=np.random.default_rng(seed))
        triangles, tree_edges = topology_from_mesh(_m)
        if edge_secret_rates is None:
            edge_secret_rates = _m.edge_secret_rates
    num_triangles = len(triangles)
    # All pairwise links that physically exist (triangle sides) - the only
    # pairs with distillation hardware and hence key pools. Used to keep
    # tree-mask key assignment on real links.
    physical_edges = {tuple(sorted((int(t[i]), int(t[j]))))
                      for t in triangles for i, j in ((0, 1), (1, 2), (0, 2))}

    # Seed torch BEFORE the first model_fn(): the initial global model and
    # DataLoader shuffling otherwise inherit whatever global state the
    # process had, making training metrics irreproducible across suite
    # compositions and standalone runs.
    torch.manual_seed(seed)
    global_model = model_fn()
    n_params = sum(p.numel() for p in global_model.parameters())

    client_triangle_count = {}
    for tri in triangles:
        for c in tri:
            client_triangle_count[c] = client_triangle_count.get(c, 0) + 1

    rng = np.random.default_rng(seed)

    # Without a per-edge rate map every edge is charged the flat
    # operating-point rate, which is only correct for a lone triangle at
    # the config fibre length.
    if edge_secret_rates is None:
        edge_secret_rates = {}

    # Pre-allocate distilled-key pools. _slice_pool's bound check prevents
    # wrap-around; the PoolAuditors below hard-fail if any offset is ever
    # consumed twice across the whole run.
    bits_per_mask = n_params * 32
    size_intra, size_inter = size_pools(triangles, tree_edges,
                                        num_rounds, n_params)
    key_source = {'intra': 'caller-supplied', 'inter': 'caller-supplied'}
    # Decide provenance once for the whole run. Distilling only the leg
    # that happens to fit the budget would leave a run whose two mask
    # legs came from different sources, which is harder to describe than
    # it is worth: the legs are interchangeable for every quantity this
    # driver reports.
    need = (size_intra if pool_intra is None else 0) + \
           (size_inter if pool_inter is None else 0)
    over_budget = need > DISTILLED_POOL_MAX_BITS
    if pool_intra is None:
        pool_intra, key_source['intra'] = distilled_pool(
            size_intra, rng, "intra-cell pool", force_prng=over_budget)
    if pool_inter is None:
        pool_inter, key_source['inter'] = distilled_pool(
            size_inter, rng, "inter-cell pool", force_prng=over_budget)
    if verbose and key_source:
        for leg, src_label in sorted(key_source.items()):
            print(f"    {leg}-cell mask pool: {src_label}"
                  + ("" if src_label == "distilled" else
                     f" (demand above the {DISTILLED_POOL_MAX_BITS/1e6:.0f} "
                     "Mbit distillation budget)"))
    audit_intra = PoolAuditor("intra", len(pool_intra))
    audit_inter = PoolAuditor("inter", len(pool_inter))

    accuracies, losses = [], []
    dropout_log, recovery_log, distinguisher_log = [], [], []
    rounds_to_target = num_rounds
    start_time = time.time()
    intra_off, inter_off = 0, 0

    # Communication accounting. Modeled-latency inputs
    # come from the config single source of truth; lazy import keeps m4
    # free of import-order coupling with config.
    comm_rounds = []
    try:
        from config import OPERATING_POINT, CLASSICAL_LINK_BPS
    except ImportError:
        from ..config import OPERATING_POINT, CLASSICAL_LINK_BPS
    secret_rate_bps = OPERATING_POINT['secret_rate_bps']

    for round_idx in range(num_rounds):
        global_params = get_model_params(global_model)

        # Step 1: Local training
        _t_local0 = time.perf_counter()
        client_updates = {}
        for tri in triangles:
            for c in tri:
                if c in client_updates:
                    continue
                local_model = model_fn()
                set_model_params(local_model, global_params.clone())
                if c < len(client_datasets):
                    train_local(local_model, client_datasets[c],
                                local_epochs, batch_size, lr, device)
                upd = get_gradient_update(global_model, local_model)
                upd = upd * (1.0 / client_triangle_count.get(c, 1))
                client_updates[c] = upd
        _local_compute_s = time.perf_counter() - _t_local0

        # Step 2: Sample dropouts
        round_dropouts = set()
        if dropout_rate > 0:
            for c in client_updates:
                if rng.random() < dropout_rate:
                    round_dropouts.add(c)
        dropout_log.append(len(round_dropouts))

        # Step 3: Intra-triangle secure_aggregation_round per cluster
        triangle_aggs = []
        # Weight actually carried into the aggregate. Each client's
        # update is pre-scaled by 1/(cells it belongs to), so a client
        # contributes a total weight of 1 when all its cells succeed.
        # A cell that falls below threshold contributes nothing, and its
        # members' shares of that cell go with it - so the divisor has
        # to be the weight that arrived, not the number of clients that
        # did not drop. Counting non-dropped clients instead shrinks
        # every update in proportion to how much was lost.
        included_weight = 0.0
        round_distinguishable = False
        any_recovery = False
        _t_step3 = time.perf_counter()
        _instr_s = 0.0
        rc = RoundComm(round_idx=round_idx, n_clients=len(client_updates),
                       max_payload_multiplicity=max(
                           client_triangle_count.values()))
        for tri in triangles:
            cluster = list(tri)
            _cluster_base = intra_off  # absolute offset for the pool audit
            cluster_pool = _slice_pool(pool_intra, intra_off, 3 * bits_per_mask)
            intra_off += 3 * bits_per_mask
            local_seeds = {c: shamir_field_element(rng) for c in cluster}
            tri_dropouts = {c for c in cluster if c in round_dropouts}
            survivors_count = len(cluster) - len(tri_dropouts)
            # Skip if fewer survivors than Shamir threshold - cannot recover
            # dropped clients' self-masks. Triangle contributes zero this round.
            if survivors_count < 2:
                triangle_aggs.append(torch.zeros(n_params))
                # Quantified for exp5's recovery-cost analysis.
                rc.threshold_failures += 1
                # the aborted cluster still exchanged its
                # per-round Shamir shares (distribution precedes dropout
                # knowledge), and its reserved key slice is DISCARDED, not
                # returned (single-use discipline - safe against partial
                # derivation). Both are ledgered so this record reconciles
                # with aux['bits_intra_consumed'].
                # Drops are detected by non-arrival, so the abort notice
                # follows payload collection: a client masks against all
                # its partners before it can know who dropped - which is
                # precisely why the reveal leg exists - so the survivor
                # of a below-threshold cluster has already sent. Charge
                # its payload as well as the notice; the cluster's
                # contribution is discarded, not its traffic.
                rc.payload_bytes += (survivors_count * n_params
                                     * BYTES_PER_MASKED_WORD)
                rc.downlink_bytes += survivors_count * BYTES_ABORT_NOTICE
                rc.share_bytes += (len(cluster) * (len(cluster) - 1)
                                   * BYTES_PER_SHAMIR_SHARE)
                rc.key_bits_intra += 3 * bits_per_mask
                # Pool audit: the discarded slice is BURNED (single-use
                # discipline) - register it so any later reuse of these
                # offsets hard-fails too.
                audit_intra.consume(_cluster_base, 3 * bits_per_mask)
                for _i in range(3):
                    _a, _b = int(cluster[_i]), int(cluster[(_i + 1) % 3])
                    _e = (min(_a, _b), max(_a, _b))
                    rc.key_bits_per_edge[_e] = (
                        rc.key_bits_per_edge.get(_e, 0) + bits_per_mask)
                continue
            agg_res = secure_aggregation_round(
                client_gradients=client_updates,
                cluster_clients=cluster,
                pairwise_pool=cluster_pool,
                local_rng_seeds=local_seeds,
                dropouts=tri_dropouts,
                threshold=2,
                aggregator_rng=rng,
                pool_auditor=audit_intra,
                pool_base_offset=_cluster_base,
            )
            triangle_aggs.append(agg_res.aggregate)
            included_weight += sum(
                1.0 / client_triangle_count.get(c, 1)
                for c in cluster if c not in tri_dropouts)
            round_distinguishable = round_distinguishable or agg_res.eavesdrop_distinguishable
            any_recovery = any_recovery or agg_res.recovery_invoked
            # Accumulate this cluster's communication accounting.
            cc = agg_res.comm
            rc.payload_bytes += cc["payload_bytes"]
            rc.share_bytes += cc["share_bytes"]
            rc.selfmask_bytes += cc["selfmask_bytes"]
            rc.recovery_share_bytes += cc["recovery_share_bytes"]
            rc.reveal_bytes += cc["reveal_bytes"]
            rc.aggregator_time_s += cc["aggregator_time_s"]
            _instr_s += cc["instrumentation_time_s"]
            for edge, bits in cc["key_bits_per_edge"].items():
                rc.key_bits_per_edge[edge] = (
                    rc.key_bits_per_edge.get(edge, 0) + bits)
                rc.key_bits_intra += bits
        _step3_s = time.perf_counter() - _t_step3
        distinguisher_log.append(round_distinguishable)
        recovery_log.append(any_recovery)

        # Step 4: Inter-triangle aggregation.
        #
        # Every cell submits its sum MASKED and the masks cancel only in
        # the aggregator's total, so no cell sum is ever exposed. Masks
        # sit on the edges of a spanning tree over the cell dual: each
        # edge's mask is added by one endpoint and subtracted by the
        # other, so it survives in both submissions and vanishes from
        # their sum. A spanning tree gives every cell degree >= 1, so
        # every submitted value carries at least one fresh uniform mask.
        #
        # This is Bonawitz et al. (2017) applied a second time, with the
        # cells as its parties: the pairwise-cancelling construction is
        # theirs, and a connected mask graph is what their cancellation
        # argument requires. K-1 edges is the minimum connected graph on
        # K cells, so this costs no more key than the hop-by-hop scheme
        # it replaces.
        #
        # The earlier scheme unmasked at each hop, handing the aggregator
        # every cell sum. With overlapping cells those sums form a linear
        # system in the individual updates; at and above 12 clients it has
        # full rank, so every client's update was solvable with no
        # collusion. Masking through to the total removes that system:
        # the aggregator observes one number.
        _t_tree0 = time.perf_counter()
        if num_triangles == 1:
            global_aggregate = triangle_aggs[0]
        else:
            # Quantize every cell sum once; masks are applied in the ring.
            q_by_tri = {i: quantize_gradient(triangle_aggs[i])
                        for i in range(num_triangles)}
            for parent, child in reversed(tree_edges):
                edge_bits = _slice_pool(pool_inter, inter_off, bits_per_mask,
                                        auditor=audit_inter)
                inter_off += bits_per_mask
                # One edge-pool consumption per mask edge per round. The
                # mask's key demand is assigned to the PHYSICAL shared
                # edge between the two cells (Delaunay dual edge = two
                # shared vertices) - inter-cell masks do not get implicit
                # dedicated hardware; a shared physical edge serving a
                # cell pair as well as its own triangles fills
                # correspondingly slower.
                rc.key_bits_inter += bits_per_mask
                _shared = sorted(int(v) for v in
                                 (set(triangles[parent]) & set(triangles[child])))
                if len(_shared) >= 2:
                    _phys = (_shared[0], _shared[1])
                else:
                    # Non-adjacent tree edge (does not occur on a Delaunay
                    # dual, where parent and child share a side). Assign
                    # the mask to the lexicographically first PHYSICAL
                    # cross link between the two triangles. The candidate
                    # set excludes self-loops and pairs with no
                    # distillation hardware / key pool.
                    _cross = sorted(
                        tuple(sorted((int(a), int(b))))
                        for a in triangles[parent] for b in triangles[child]
                        if int(a) != int(b)
                        and tuple(sorted((int(a), int(b)))) in physical_edges)
                    if not _cross:
                        raise ValueError(
                            f"Tree edge ({parent},{child}) connects "
                            "triangles with no physical cross link; the "
                            "single-hop tree mask cannot be keyed on this "
                            "topology.")
                    _phys = _cross[0]
                rc.key_bits_per_edge[_phys] = (
                    rc.key_bits_per_edge.get(_phys, 0) + bits_per_mask)
                mask = derive_modular_mask(n_params, edge_bits)
                # Opposite signs on the two endpoints: the mask stays in
                # both submissions and cancels only when they are summed.
                q_by_tri[child] = apply_modular_mask(
                    q_by_tri[child], mask, sign=+1)
                q_by_tri[parent] = apply_modular_mask(
                    q_by_tri[parent], mask, sign=-1)
            # Every cell submits; the aggregator's only observation is
            # this sum, in which all tree masks cancel.
            q_total = np.zeros(n_params, dtype=np.uint32)
            for i in range(num_triangles):
                q_total = (q_total + q_by_tri[i]).astype(np.uint32)
                # One masked cell vector on the wire per cell, not per hop.
                rc.inter_triangle_bytes += n_params * BYTES_PER_MASKED_WORD
            global_aggregate = dequantize_gradient(q_total)
        # Inter-cell stage (masking plus the aggregator's single sum) is
        # charged to the aggregation bucket.
        _tree_s = time.perf_counter() - _t_tree0
        rc.aggregator_time_s += _tree_s

        # Step 5: Apply update, normalised by the weight that arrived.
        if included_weight > 0:
            new_params = global_params + global_aggregate / included_weight
            set_model_params(global_model, new_params)

        # Round communication/latency record
        # += not =: abort notices for threshold-failed clusters were
        # already added in step 3.
        rc.downlink_bytes += (len(client_updates) * n_params
                              * BYTES_PER_MODEL_WORD)
        rc.local_compute_s = _local_compute_s
        # Masking = step-3 wall time minus the cluster-level aggregator
        # work (aggregator_time_s minus the tree stage, which ran after
        # _step3_s closed) minus the distinguisher instrumentation
        #.
        rc.masking_s = max(0.0, _step3_s
                           - (rc.aggregator_time_s - _tree_s)
                           - _instr_s)
        # Key-distillation wait: the slowest physical edge's pool-fill
        # time. Edges distil concurrently on dedicated optics, so the
        # round waits on the worst ratio of bits demanded to that edge's
        # own rate - not on the worst demand at a shared rate. An edge's
        # rate depends on its length and on how many ways each endpoint
        # splits its source, so the two factors compound: a long edge
        # between two well-connected nodes is the usual binding one.
        if rc.key_bits_per_edge:
            rc.key_wait_s = max(
                bits / max(edge_secret_rates.get(e, secret_rate_bps), 1.0)
                for e, bits in rc.key_bits_per_edge.items())
        else:
            rc.key_wait_s = 0.0
        rc.comm_s = ((rc.uplink_bytes() + rc.downlink_bytes) * 8
                     / CLASSICAL_LINK_BPS)
        comm_rounds.append(rc)

        acc, loss = evaluate(global_model, test_dataset, device)
        accuracies.append(acc)
        losses.append(loss)
        if acc >= target_accuracy and rounds_to_target == num_rounds:
            rounds_to_target = round_idx + 1

        if verbose and (round_idx + 1) % 10 == 0:
            print(f"    [entangled {num_clients}c/{num_triangles}t] "
                  f"Round {round_idx+1}/{num_rounds}: "
                  f"acc={acc:.4f} dropouts={len(round_dropouts)} "
                  f"recovery={'Y' if any_recovery else 'N'} "
                  f"distinguishable={'Y' if round_distinguishable else 'N'}")

    wall_clock = time.time() - start_time

    # Pool-reuse hard audit: every consumed slice must be
    # disjoint (enforced en route - any overlap raised immediately) and
    # the audited totals must equal the ledgered burns exactly.
    assert audit_intra.bits_consumed == intra_off, (
        f"intra pool audit mismatch: {audit_intra.bits_consumed} audited "
        f"!= {intra_off} ledgered")
    assert audit_inter.bits_consumed == inter_off, (
        f"inter pool audit mismatch: {audit_inter.bits_consumed} audited "
        f"!= {inter_off} ledgered")
    if verbose:
        print(f"    Pool-reuse audit (OTP single-use): PASS - "
              f"intra {audit_intra.n_intervals} disjoint slices / "
              f"{audit_intra.bits_consumed} bits; "
              f"inter {audit_inter.n_intervals} / "
              f"{audit_inter.bits_consumed} bits")

    fl_result = FLResult(
        mode=f"entangled_{num_clients}c_{num_triangles}t",
        accuracies=accuracies,
        losses=losses,
        final_accuracy=accuracies[-1] if accuracies else 0,
        rounds_to_target=rounds_to_target,
        wall_clock_s=wall_clock,
        mask_gen_times_s=[],
        n_params=n_params,
        comm_rounds=comm_rounds,
    )
    aux = {
        "dropouts_per_round": dropout_log,
        "recovery_invoked_per_round": recovery_log,
        "distinguishable_per_round": distinguisher_log,
        "bits_intra_consumed": intra_off,
        "bits_inter_consumed": inter_off,
        "pool_audit": {"intra": audit_intra.summary(),
                       "inter": audit_inter.summary()},
        # "distilled" or "prng" per leg: a PRNG pool cannot carry an
        # entangled-key claim, so callers that report one must check.
        "key_source": key_source,
    }
    return fl_result, aux


def run_multitriangle_fl_legacy(
    model_fn, client_datasets, test_dataset,
    num_clients, num_rounds=60, local_epochs=2,
    batch_size=32, lr=0.01, target_accuracy=0.92,
    triangles=None, tree_edges=None,
    device='cpu', verbose=True,
):
    """Legacy hierarchical driver; run_multitriangle_fl is the maintained
    driver and adds modular OTP masking, pre-sized distilled-key pools,
    the two-mask design, and per-round dropout sampling. No callers;
    retained for reference.

    Run federated learning with multi-triangle hierarchical aggregation.

    Protocol:
      1. All clients train locally -> produce gradient updates
      2. Intra-triangle: Bonawitz pairwise masking within each triangle
         -> triangle aggregates (individual gradients hidden)
      3. Inter-triangle: hierarchical aggregation along tree with masking.
         Child SENDS masked payload; parent RECEIVES and unmasks.
         Wire payloads are recorded for eavesdropper verification.
      4. Root produces global aggregate -> FedAvg update

    Args:
        triangles: Optional pre-computed triangle list (from Module 3 Delaunay).
                   Falls back to define_triangle_topology() if None.
        tree_edges: Optional pre-computed aggregation tree edges.

    Returns:
        FLResult with training metrics.
        eavesdrop_stats: dict with per-round eavesdropper analysis.
    """
    if triangles is None or tree_edges is None:
        triangles, _, tree_edges = define_triangle_topology(num_clients)
    num_triangles = len(triangles)

    # Same seed-before-model order as the live drivers: keeps this reference implementation reproducible if
    # it is ever revived.
    torch.manual_seed(42)
    global_model = model_fn()
    n_params = sum(p.numel() for p in global_model.parameters())

    # Track which clients belong to which triangles
    client_triangle_count = {}
    for tri in triangles:
        for c in tri:
            client_triangle_count[c] = client_triangle_count.get(c, 0) + 1

    accuracies, losses = [], []
    rounds_to_target = num_rounds
    start_time = time.time()
    rng = np.random.default_rng(42)

    # Pre-generate symmetric pairwise seeds for ALL client pairs
    max_client = max(max(t) for t in triangles)
    all_seeds = rng.integers(0, 2**31, size=(num_rounds, max_client+1, max_client+1))
    for r in range(num_rounds):
        for i in range(max_client+1):
            for j in range(i+1, max_client+1):
                all_seeds[r, j, i] = all_seeds[r, i, j]

    # Inter-triangle seeds (for tree aggregation masking)
    inter_seeds = rng.integers(0, 2**31, size=(num_rounds, num_triangles, num_triangles))
    for r in range(num_rounds):
        for i in range(num_triangles):
            for j in range(i+1, num_triangles):
                inter_seeds[r, j, i] = inter_seeds[r, i, j]

    # Eavesdropper verification: track wire payloads vs true aggregates
    eavesdrop_stats = {
        "wire_true_cosine_sims": [],     # cosine sim between wire payload and true aggregate
        "wire_true_correlations": [],    # Pearson correlation
    }

    for round_idx in range(num_rounds):
        global_params = get_model_params(global_model)

        # Step 1: All clients train locally
        client_updates = {}
        for tri in triangles:
            for c in tri:
                if c not in client_updates:
                    local_model = model_fn()
                    set_model_params(local_model, global_params.clone())
                    if c < len(client_datasets):
                        train_local(local_model, client_datasets[c],
                                   local_epochs, batch_size, lr, device)
                    update = get_gradient_update(global_model, local_model)
                    scale = 1.0 / client_triangle_count.get(c, 1)
                    client_updates[c] = update * scale

        # Step 2: Intra-triangle Bonawitz masking
        triangle_aggregates = []
        for tri_idx, tri in enumerate(triangles):
            clients_in_tri = list(tri)
            masked_updates = []
            for c in clients_in_tri:
                masked = client_updates[c].clone()
                for c2 in clients_in_tri:
                    if c2 == c:
                        continue
                    pair_seed = int(all_seeds[round_idx, min(c, c2), max(c, c2)])
                    mask = generate_prng_mask(n_params, pair_seed)
                    if c < c2:
                        masked = masked + mask
                    else:
                        masked = masked - mask
                masked_updates.append(masked)
            tri_agg = torch.stack(masked_updates).sum(dim=0)
            triangle_aggregates.append(tri_agg)

        # Step 3: Inter-triangle hierarchical aggregation
        # Models actual network communication: child and parent are separate
        # entities that each hold one copy of the shared key.
        if num_triangles == 1:
            global_aggregate = triangle_aggregates[0]
        else:
            # Each triangle holds its own local aggregate
            agg = {i: triangle_aggregates[i] for i in range(num_triangles)}

            if tree_edges:
                round_cosines = []
                round_corrs = []
                for parent, child in reversed(tree_edges):
                    pair_seed = int(inter_seeds[round_idx, min(parent, child), max(parent, child)])
                    inter_mask = generate_prng_mask(n_params, pair_seed)

                    # CHILD NODE scope: only sees own aggregate + shared key
                    wire_payload = agg[child] + inter_mask  # transmitted over network

                    # Eavesdropper verification: analyze wire payload
                    # An eavesdropper intercepting wire_payload should learn nothing
                    # about agg[child]. Measure information leakage:
                    true_agg = agg[child]
                    cos_sim = torch.nn.functional.cosine_similarity(
                        wire_payload.unsqueeze(0), true_agg.unsqueeze(0)
                    ).item()
                    # Pearson correlation
                    wp_centered = wire_payload - wire_payload.mean()
                    ta_centered = true_agg - true_agg.mean()
                    corr_num = (wp_centered * ta_centered).sum().item()
                    corr_den = (torch.sqrt((wp_centered**2).sum() * (ta_centered**2).sum())).item()
                    pearson = corr_num / max(corr_den, 1e-12)
                    round_cosines.append(cos_sim)
                    round_corrs.append(pearson)

                    # PARENT NODE scope: only sees wire_payload + shared key
                    child_recovered = wire_payload - inter_mask
                    agg[parent] = agg[parent] + child_recovered

                eavesdrop_stats["wire_true_cosine_sims"].append(np.mean(round_cosines))
                eavesdrop_stats["wire_true_correlations"].append(np.mean(round_corrs))

            root = tree_edges[0][0] if tree_edges else 0
            global_aggregate = agg[root]

        # Step 4: Compute weighted average
        total_unique_clients = len(client_updates)
        new_params = global_params + global_aggregate / total_unique_clients
        set_model_params(global_model, new_params)

        # Evaluate
        acc, loss = evaluate(global_model, test_dataset, device)
        accuracies.append(acc)
        losses.append(loss)

        if acc >= target_accuracy and rounds_to_target == num_rounds:
            rounds_to_target = round_idx + 1

        if verbose and (round_idx + 1) % 10 == 0:
            print(f"    [{num_clients}c/{num_triangles}t] Round {round_idx+1}/{num_rounds}: "
                  f"acc={acc:.4f}, loss={loss:.4f}")

    wall_clock = time.time() - start_time

    fl_result = FLResult(
        mode=f"multi_tri_{num_clients}c_{num_triangles}t",
        accuracies=accuracies,
        losses=losses,
        final_accuracy=accuracies[-1] if accuracies else 0,
        rounds_to_target=rounds_to_target,
        wall_clock_s=wall_clock,
        mask_gen_times_s=[],
        n_params=n_params,
    )
    return fl_result, eavesdrop_stats

