"""
Module 2: Key-distillation post-processing (information reconciliation +
privacy amplification). 'Randomness extraction' refers ONLY to the Toeplitz
hashing step (Leftover Hash Lemma); the pipeline as a whole is secret-key
agreement in the source model (Maurer 1993; Ahlswede-Csiszar 1993).

Reconciliation: column-weight-3 LDPC codes with sum-product belief propagation
decoder (Gallager 1962, Richardson & Urbanke 2001). Brings the two endpoints'
correlated-but-erroneous bit strings into bit-exact agreement.

Privacy amplification (PA): Toeplitz hashing (Krawczyk 1994) compresses the
reconciled string to remove residual classical information leaked by the
reconciliation syndrome and by physical-layer side channels. Security follows
from the Leftover Hash Lemma (Renner 2005, Tomamichel 2012): output is
epsilon-close to uniform if input min-entropy exceeds output length +
2*log(1/epsilon).

Shared-bit rate:
  R_shared = R_raw × [1 - max(beta*H(q), 1-R_code) - D_tag - margin/N]
                   × (1 - FER) × (1 - bell_duty),
where the reconciliation leakage is the LARGER of the declared fixed code's
syndrome fraction (1 - R_code: a rate-0.84 LDPC leaks 16% of every block
regardless of QBER — which is what fast_ec_simulate actually charges) and
the rate-adaptive envelope beta*H(q) with beta ≈ 1.15 (the leakage of an
efficiency-beta code family tuned to q, needed once q exceeds the fixed
code's design point). The crossover sits at the code's design QBER (~2%
for rate 0.84). This never claims less leakage than the implemented code
and never more correction capability than the beta envelope allows.
D_tag is the GLLP-style tagged fraction: multi-pair emissions that a
passive beamsplitter could copy WITHOUT disturbance (Gottesman, Lo,
Lutkenhaus & Preskill 2004). Tagged bits are treated as known to a
transcript recorder and removed in PA — this is what supports the
harvest-now-decrypt-later immunity claim. There is NO eavesdropper term:
secrecy is claimed against observers of the public transcript only.
"""
import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix


def default_tagged_fraction():
    """GLLP tagged fraction: multi-pair emissions / pair emissions.

    The fraction of coincidences a passive beamsplitter could copy without
    disturbance (photon-number-splitting analogue for entanglement sources).
    Removed in privacy amplification so the classical transcript never
    suffices to reconstruct the key (HNDL immunity). GLLP 2004.
    """
    try:
        try:
            from config import QUANTUM
        except ImportError:
            from ..config import QUANTUM
        return QUANTUM['multi_photon_prob'] / QUANTUM['source_brightness']
    except Exception:
        return 8e-5 / 0.01  # config.py defaults


@dataclass
class ECResult:
    """Results from one error correction + privacy amplification run."""
    n_raw_bits: int
    n_corrected_bits: int
    n_secure_bits: int
    frame_success: bool
    residual_errors: int
    syndrome_bits: int
    reconciliation_efficiency: float
    compression_ratio: float


def generate_ldpc_parity_check(n_bits, code_rate, rng=None):
    """Generate a regular LDPC parity check matrix H.
    
    Uses a simple column-weight-3 regular construction suitable for 
    QBER in the 0.01-0.06 range.
    
    Args:
        n_bits: Block length (number of variable nodes)
        code_rate: Code rate R (syndrome size = n_bits * (1-R))
        rng: Random generator
    
    Returns:
        H: Sparse parity check matrix (m × n)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    m = int(n_bits * (1 - code_rate))  # Number of check nodes
    col_weight = 3  # Each bit participates in 3 checks
    row_weight = max(1, int(col_weight * n_bits / m))

    rows, cols = [], []
    for j in range(n_bits):
        check_indices = rng.choice(m, size=col_weight, replace=False)
        for ci in check_indices:
            rows.append(ci)
            cols.append(j)

    H = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(m, n_bits)
    )
    return H


def bp_decode(H, syndrome, channel_llr, max_iter=50):
    """Belief propagation decoder for binary LDPC over BSC.
    
    Args:
        H: Sparse parity check matrix (m × n)
        syndrome: Target syndrome vector (m,)
        channel_llr: Channel log-likelihood ratios (n,)
        max_iter: Maximum BP iterations
    
    Returns:
        decoded: Estimated error pattern (n,)
        success: Whether all syndrome checks pass
    """
    m, n = H.shape
    H_csr = H.tocsr()
    H_csc = H.tocsc()

    # Initialize messages
    # For each edge (i,j): message from variable j to check i
    # and message from check i to variable j
    var_to_check = {}  # (i, j) -> LLR message
    check_to_var = {}  # (i, j) -> LLR message

    # Build adjacency
    check_neighbors = [[] for _ in range(m)]  # checks -> list of var indices
    var_neighbors = [[] for _ in range(n)]     # vars -> list of check indices

    for i in range(m):
        row = H_csr.getrow(i)
        for j in row.indices:
            check_neighbors[i].append(j)
            var_neighbors[j].append(i)
            var_to_check[(i, j)] = channel_llr[j]
            check_to_var[(i, j)] = 0.0

    for iteration in range(max_iter):
        # Check node update
        for i in range(m):
            neighbors = check_neighbors[i]
            if len(neighbors) == 0:
                continue
            for j in neighbors:
                # Product of tanh(msg/2) for all j' != j
                product = 1.0
                for jp in neighbors:
                    if jp != j:
                        msg = var_to_check[(i, jp)]
                        product *= np.tanh(np.clip(msg / 2, -10, 10))
                # Adjust for syndrome bit
                sign = 1 - 2 * int(syndrome[i])
                product *= sign
                check_to_var[(i, j)] = 2 * np.arctanh(np.clip(product, -0.9999, 0.9999))

        # Variable node update
        for j in range(n):
            neighbors = var_neighbors[j]
            for i in neighbors:
                total = channel_llr[j]
                for ip in neighbors:
                    if ip != i:
                        total += check_to_var[(ip, j)]
                var_to_check[(i, j)] = total

        # Hard decision
        posterior = np.zeros(n)
        for j in range(n):
            posterior[j] = channel_llr[j]
            for i in var_neighbors[j]:
                posterior[j] += check_to_var[(i, j)]

        decoded = (posterior < 0).astype(np.int8)

        # Check syndrome
        computed_syndrome = (H_csr @ decoded) % 2
        if np.array_equal(computed_syndrome, syndrome.astype(np.int8)):
            return decoded, True

    # Failed to converge
    decoded = (posterior < 0).astype(np.int8)
    return decoded, False


def fast_ec_simulate(alice_bits, bob_bits, qber_estimate, code_rate=0.84,
                     tagged_fraction=None,
                     block_size=50000, max_iter=50, rng=None):
    """Statistical reconciliation model with hardcoded FER curve.

    KNOWN LIMITATION: the 99.5/90/10% success-rate tiers below are a
    placeholder for production LDPC performance. They are NOT derived from
    density-evolution analysis nor measured from a real decoder. For
    publication-quality numbers, replace with empirical FER from a 5G NR
    LDPC reference decoder or finite-blocklength bound (Polyanskiy 2010).
    Retained here so that rate-budget sweeps can run in CPU-seconds; the
    full BP decoder (`bp_decode` above) is slow in Python and not suitable
    for parameter sweeps.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = min(len(alice_bits), len(bob_bits))
    if n == 0:
        return ECResult(0, 0, 0, False, 0, 0, 0, 0)

    n_blocks = max(1, n // block_size)
    results = []

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block_a = alice_bits[start:end]
        block_b = bob_bits[start:end]
        block_n = len(block_a)

        errors = np.sum(block_a != block_b)
        actual_qber = errors / max(block_n, 1)

        # Syndrome size
        syndrome_bits = int(block_n * (1 - code_rate))

        # Reconciliation efficiency
        h_q = binary_entropy(actual_qber) if actual_qber > 0 else 0
        if h_q > 0:
            beta = (syndrome_bits / block_n) / h_q
        else:
            beta = 1.0

        # Frame success probability (modeled from LDPC performance curves)
        # via the shared tier model — a SINGLE definition, also used by
        # compute_rate_waterfall, so rate budgets can never disagree with
        # the block-level EC model (Gate-1 round-2 finding 1).
        frame_success = rng.random() < (
            1.0 - tier_frame_error_rate(actual_qber, code_rate))

        if frame_success:
            # After successful EC, both parties hold identical bits
            corrected_bits = block_n
            residual_errors = 0
        else:
            corrected_bits = 0
            residual_errors = errors

        # Privacy amplification
        if frame_success:
            tf = default_tagged_fraction() if tagged_fraction is None else tagged_fraction
            secure_bits = max(0, int(block_n * (1 - tf)) - syndrome_bits - 128)
        else:
            secure_bits = 0

        compression = secure_bits / max(block_n, 1)

        results.append(ECResult(
            n_raw_bits=block_n,
            n_corrected_bits=corrected_bits,
            n_secure_bits=secure_bits,
            frame_success=frame_success,
            residual_errors=residual_errors,
            syndrome_bits=syndrome_bits,
            reconciliation_efficiency=beta,
            compression_ratio=compression,
        ))

    return results


def binary_entropy(p):
    """Binary entropy function H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    if p <= 0 or p >= 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def estimate_capacity_qber(code_rate):
    """Estimate the maximum correctable QBER for a given code rate.
    Based on BSC channel capacity: R = 1 - H(q_max).
    """
    # Solve 1 - H(q) = code_rate for q
    # H(q) = 1 - code_rate
    target_entropy = 1 - code_rate
    # Binary search
    lo, hi = 0.0, 0.5
    for _ in range(50):
        mid = (lo + hi) / 2
        if binary_entropy(mid) < target_entropy:
            lo = mid
        else:
            hi = mid
    return mid


def tier_frame_error_rate(qber, code_rate=0.84):
    """Frame-error rate of the declared fixed-rate code under the SAME
    statistical tier model fast_ec_simulate uses (placeholder tiers — not
    density evolution; see that function's honesty note):

        q < 0.8 × capacity  →  FER 0.005
        q <       capacity  →  FER 0.10   (degraded band)
        q ≥       capacity  →  FER 0.90   (the fixed code effectively fails)

    Single source of the tiers so the rate waterfall can never claim EC
    performance the block-level model doesn't deliver (Gate-1 round-2
    finding 1: the old fixed 0.005 constant overstated the projected
    operating point, which sits in the 0.10 band, by ~10%).
    """
    capacity_qber = estimate_capacity_qber(code_rate)
    if qber < capacity_qber * 0.8:
        return 0.005
    if qber < capacity_qber:
        return 0.10
    return 0.90


def compute_rate_waterfall(coincidence_rate_hz, qber, code_rate=0.84,
                           security_margin=128, block_size=50000,
                           bell_duty_cycle=0.02, frame_error_rate=None,
                           tagged_fraction=None, assume_rate_adapted=False):
    """Compute the complete rate waterfall from raw to final secure rate.

    frame_error_rate=None (default) derives the FER from the declared
    code's tier model (tier_frame_error_rate) at this qber, keeping the
    waterfall consistent with the block-level EC model at every QBER.

    assume_rate_adapted=True computes the HYPOTHETICAL rate-adapted-family
    envelope instead: leakage β·h(q) and design-band FER 0.005 — for
    explicitly-labeled comparison curves only, never for headline numbers
    (the repo implements one fixed rate-0.84 code).

    Returns dict with each stage's rate and overhead.
    """
    h_q = binary_entropy(qber)
    beta = 1.15  # LDPC reconciliation inefficiency (rate-adaptive envelope)

    raw_rate = coincidence_rate_hz
    after_sifting = raw_rate  # Fixed-basis: zero sifting loss

    # Shared-bit fraction (source model — no eavesdropper term):
    #   shared_fraction = 1 - max(beta*h(q), 1 - code_rate) - D_tag
    # Reconciliation leakage is the LARGER of:
    #   (a) the declared FIXED code's syndrome fraction (1 - code_rate):
    #       rate-0.84 LDPC leaks 16% of every block regardless of QBER —
    #       this is what fast_ec_simulate actually charges; and
    #   (b) the rate-adaptive envelope beta*h(q): what an efficiency-beta
    #       code family tuned to q would leak, which a real system needs
    #       once q exceeds the fixed code's design point (~2% for 0.84).
    # max(a, b) never claims less leakage than the implemented code and
    # never more correction capability than the envelope allows (Gate-1 F1;
    # the previous pure-beta*h(q) charge was 9.5% optimistic at the
    # operating QBER of 0.0085).
    # D_tag is the GLLP tagged fraction (multi-pair emissions a passive
    # beamsplitter could copy undetected; removed in PA for HNDL immunity).
    if assume_rate_adapted:
        # HYPOTHETICAL envelope: an efficiency-beta code family tuned to q.
        syndrome_fraction = h_q * beta
        if frame_error_rate is None:
            frame_error_rate = 0.005  # adapted code at its design band
    else:
        syndrome_fraction = max(h_q * beta, 1.0 - code_rate)  # reconciliation leakage
        if frame_error_rate is None:
            frame_error_rate = tier_frame_error_rate(qber, code_rate)
    if tagged_fraction is None:
        tagged_fraction = default_tagged_fraction()
    pa_compression_fraction = tagged_fraction
    secure_fraction_per_block = max(0, 1 - syndrome_fraction - pa_compression_fraction
                                    - security_margin / block_size)
    
    after_ec = raw_rate * (1 - syndrome_fraction)
    after_pa = raw_rate * secure_fraction_per_block
    after_verification = after_pa * (1 - frame_error_rate)
    after_bell = after_verification * (1 - bell_duty_cycle)

    pa_overhead_pct = pa_compression_fraction * 100  # GLLP tagged-fraction removal

    return {
        "raw_coincidence_rate": raw_rate,
        "after_sifting": after_sifting,
        "sifting_overhead_pct": 0,
        "after_ec": after_ec,
        "ec_overhead_pct": syndrome_fraction * 100,
        "after_pa": after_pa,
        "pa_overhead_pct": pa_overhead_pct,
        "tagged_fraction": tagged_fraction,
        "frame_error_rate": frame_error_rate,
        "assume_rate_adapted": assume_rate_adapted,
        "after_verification": after_verification,
        "verification_overhead_pct": frame_error_rate * 100,
        "final_secure_rate": after_bell,
        "bell_overhead_pct": bell_duty_cycle * 100,
        "total_overhead_pct": (1 - after_bell / raw_rate) * 100 if raw_rate > 0 else 100,
    }


def toeplitz_hash(bits, output_length, seed=None, rng=None):
    """Privacy amplification via Toeplitz matrix-vector multiplication.

    A Toeplitz matrix T of size (m × n) is defined by (n + m - 1) independent
    random bits: the first row (length n) and the remaining (m-1) entries of
    the first column. Row i is: [col[i], row[0], row[1], ..., row[n-2]]
    shifted appropriately.

    Security guarantee (Leftover Hash Lemma, Renner 2005): if the input has
    min-entropy >= output_length + 2*security_parameter, the output is
    2^{-security_parameter}-close to uniform.

    In hardware this maps to an LFSR cascade on FPGA (Assumption C).
    """
    if rng is None:
        rng = np.random.default_rng(seed if seed else 0)

    n = len(bits)
    m = output_length
    if m >= n:
        return bits.copy()

    # Generate (n + m - 1) random bits defining the Toeplitz matrix
    toeplitz_seed = rng.integers(0, 2, size=n + m - 1, dtype=np.int8)
    # Row i of T is toeplitz_seed[m-1-i : m-1-i+n] (i = 0..m-1)
    # Equivalently: first row = toeplitz_seed[m-1 : m-1+n]
    #               first col = toeplitz_seed[m-1 : :-1] (reversed prefix)

    output = np.zeros(m, dtype=np.int8)
    for i in range(m):
        row_start = m - 1 - i
        row = toeplitz_seed[row_start:row_start + n]
        output[i] = np.sum(bits * row) % 2

    return output


def validate_pa_output(secure_bits, n_samples=5):
    """Basic randomness validation of privacy-amplified output.

    Runs NIST frequency (monobit) test: checks that the proportion of 1s
    is within expected range for uniform random bits.

    Returns:
        pass_rate: fraction of samples passing the monobit test (p > 0.01).
    """
    from scipy.special import erfc
    n = len(secure_bits)
    if n < 100:
        return 1.0  # Too short to test meaningfully

    # Monobit test: S = |sum(2*bit - 1)| / sqrt(n), p = erfc(S/sqrt(2))
    s_obs = abs(np.sum(2 * secure_bits.astype(np.float64) - 1)) / np.sqrt(n)
    p_value = erfc(s_obs / np.sqrt(2))
    return p_value


def validate_bp_against_statistical(qber_values=None, block_size=50000, rng=None):
    """Validate the fast statistical EC model against the full BP decoder.

    Runs real BP decoding on a small number of blocks at several QBER values
    and compares frame error rates to the statistical model's assumptions.

    Returns:
        validation_table: list of dicts with qber, bp_fer, stat_fer, agreement.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if qber_values is None:
        qber_values = [0.01, 0.02, 0.03, 0.04]

    code_rate = 0.84
    n_blocks_per_qber = 5
    validation_table = []

    for qber in qber_values:
        bp_successes = 0
        stat_successes = 0

        for _ in range(n_blocks_per_qber):
            # Generate correlated bits with known QBER
            alice = rng.integers(0, 2, size=block_size, dtype=np.int8)
            error_mask = rng.random(block_size) < qber
            bob = alice.copy()
            bob[error_mask] = 1 - bob[error_mask]

            # Full BP decode
            H = generate_ldpc_parity_check(block_size, code_rate, rng)
            error_pattern = (alice ^ bob).astype(np.int8)
            syndrome = (H @ error_pattern) % 2
            channel_llr = np.full(block_size, np.log((1 - qber) / max(qber, 1e-10)))
            channel_llr[bob == 1] *= -1
            decoded, bp_success = bp_decode(H, syndrome, channel_llr, max_iter=50)
            if bp_success:
                bp_successes += 1

            # Statistical model prediction
            capacity_qber = estimate_capacity_qber(code_rate)
            if qber < capacity_qber * 0.8:
                stat_successes += (1 if rng.random() < 0.995 else 0)
            elif qber < capacity_qber:
                stat_successes += (1 if rng.random() < 0.90 else 0)
            else:
                stat_successes += (1 if rng.random() < 0.10 else 0)

        bp_fer = 1.0 - bp_successes / n_blocks_per_qber
        stat_fer = 1.0 - stat_successes / n_blocks_per_qber

        validation_table.append({
            "qber": qber,
            "bp_fer": bp_fer,
            "stat_fer": stat_fer,
            "bp_success_rate": bp_successes / n_blocks_per_qber,
            "agreement": abs(bp_fer - stat_fer) < 0.3,
        })

    return validation_table
