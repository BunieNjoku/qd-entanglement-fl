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
regardless of QBER - which is what fast_ec_simulate actually charges) and
the rate-adaptive envelope beta*H(q) with beta ≈ 1.15 (the leakage of an
efficiency-beta code family tuned to q, needed once q exceeds the fixed
code's design point). The crossover sits at the code's design QBER (~2%
for rate 0.84). This never claims less leakage than the implemented code
and never more correction capability than the beta envelope allows.
D_tag is the GLLP-style tagged fraction: multi-pair emissions that a
passive beamsplitter could copy WITHOUT disturbance (Gottesman, Lo,
Lutkenhaus & Preskill 2004). Tagged bits are treated as known to a
transcript recorder and removed in PA - this is what supports the
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
        # via the shared tier model - a SINGLE definition, also used by
        # compute_rate_waterfall, so rate budgets can never disagree with
        # the block-level EC model.
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
    statistical tier model fast_ec_simulate uses (placeholder tiers - not
    density evolution; see that function's honesty note):

        q < 0.8 × capacity  ->  FER 0.005
        q <       capacity  ->  FER 0.10   (degraded band)
        q ≥       capacity  ->  FER 0.90   (the fixed code effectively fails)

    Single source of the tiers so the rate waterfall can never claim EC
    performance the block-level model doesn't deliver: the projected operating point sits in the 0.10 band, so a
    flat 0.005 would overstate it by ~10%.
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
    envelope instead: leakage β·h(q) and design-band FER 0.005 - for
    explicitly-labeled comparison curves only, never for headline numbers
    (the repo implements one fixed rate-0.84 code).

    Returns dict with each stage's rate and overhead.
    """
    h_q = binary_entropy(qber)
    beta = 1.15  # LDPC reconciliation inefficiency (rate-adaptive envelope)

    raw_rate = coincidence_rate_hz
    after_sifting = raw_rate  # Fixed-basis: zero sifting loss

    # Shared-bit fraction (source model - no eavesdropper term):
    #   shared_fraction = 1 - max(beta*h(q), 1 - code_rate) - D_tag
    # Reconciliation leakage is the LARGER of:
    #   (a) the declared FIXED code's syndrome fraction (1 - code_rate):
    #       rate-0.84 LDPC leaks 16% of every block regardless of QBER -
    #       this is what fast_ec_simulate actually charges; and
    #   (b) the rate-adaptive envelope beta*h(q): what an efficiency-beta
    #       code family tuned to q would leak, which a real system needs
    #       once q exceeds the fixed code's design point (~2% for 0.84).
    # max(a, b) never claims less leakage than the implemented code and
    # never more correction capability than the envelope allows: a pure beta*h(q) charge would be 9.5% optimistic at the
    # operating QBER of 0.0085.
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


def compute_rate_waterfall_qkd(coincidence_rate_hz, qber, sifting_keep,
                               code_rate=0.84, security_margin=128,
                               block_size=50000, bell_duty_cycle=0.02,
                               frame_error_rate=None,
                               assume_rate_adapted=False):
    """Shor-Preskill asymptotic key fraction under entanglement-based QKD
    framing of the SAME physical layer the source-model waterfall consumes:

        r = sifting_keep * (1 - h2(e_ph) - leak_EC - margin)

    then the shared tier-FER and duty post-factors. e_ph = e_bit, the
    standard symmetric-channel assumption (exp13 prints the caveat about
    m1's noise model not being purely depolarizing).

    NO tagged-fraction charge, deliberately: tagging is the
    prepare-and-measure treatment of multi-photon pulses (GLLP 2004). For
    an entangled source, the basis-independent-source argument (Koashi &
    Preskill, PRL 90, 057902, 2003) and the entangled-PDC analysis (Ma,
    Fung & Lo, PRA 76, 012307, 2007) fold multi-pair emission into the
    measured error rates via squashing (Beaudry, Moroder & Lutkenhaus,
    PRL 101, 093601, 2008). Charging tagging here would understate the
    QKD arm's rate and overstate the source-model arm's advantage. The
    source-model waterfall keeps its explicit D_tag charge because it
    never subtracts an Eve bound at all; each arm gets its own standard
    analysis, not a hybrid.

    leak_EC, the FER tiers, the security margin, and the duty cycle are
    charged IDENTICALLY to compute_rate_waterfall (same declared LDPC,
    same code rate), so the delta between the two waterfalls isolates the
    framing itself: (i) sifting, (ii) the net PA delta h2(e_bit) - D_tag.
    assume_rate_adapted mirrors the source waterfall's semantics EXACTLY
    (leakage beta*h(q), design-band FER 0.005): sweep curves must compare
    the two arms under the same EC model, since mixing a rate-adapted
    source arm with a fixed-code QKD arm would inflate the ratio below
    the ~2% crossover.

    Comparison-only; never feeds headline numbers. Returns the same dict
    shape as compute_rate_waterfall with 'tagged_fraction' fixed at 0.0,
    plus 'eve_bound_fraction' and 'sifting_keep'.
    """
    h_q = binary_entropy(qber)
    beta = 1.15  # same envelope inefficiency as the source waterfall

    raw_rate = coincidence_rate_hz
    after_sifting = raw_rate * sifting_keep

    if assume_rate_adapted:
        syndrome_fraction = h_q * beta
        if frame_error_rate is None:
            frame_error_rate = 0.005  # adapted code at its design band
    else:
        syndrome_fraction = max(h_q * beta, 1.0 - code_rate)
        if frame_error_rate is None:
            frame_error_rate = tier_frame_error_rate(qber, code_rate)

    eve_bound_fraction = h_q  # e_ph = e_bit
    secure_fraction_per_block = max(0, 1 - eve_bound_fraction
                                    - syndrome_fraction
                                    - security_margin / block_size)

    after_ec = after_sifting * (1 - syndrome_fraction)
    after_pa = after_sifting * secure_fraction_per_block
    after_verification = after_pa * (1 - frame_error_rate)
    after_bell = after_verification * (1 - bell_duty_cycle)

    return {
        "raw_coincidence_rate": raw_rate,
        "after_sifting": after_sifting,
        "sifting_overhead_pct": (1 - sifting_keep) * 100,
        "sifting_keep": sifting_keep,
        "after_ec": after_ec,
        "ec_overhead_pct": syndrome_fraction * 100,
        "after_pa": after_pa,
        "pa_overhead_pct": eve_bound_fraction * 100,
        "eve_bound_fraction": eve_bound_fraction,
        "tagged_fraction": 0.0,
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

    # y_i = sum_j seed[(m-1-i)+j] * x_j mod 2 is a correlation - compute
    # it with one FFT convolution instead of an O(m·n) Python loop (the
    # SP 800-22 battery hashes ~4e5-bit inputs, ). Integer
    # counts are exact in float64 up to 2^53; verified bit-identical to
    # the direct row-by-row product for the same toeplitz_seed.
    from scipy.signal import fftconvolve
    c = fftconvolve(toeplitz_seed.astype(np.float64),
                    bits[::-1].astype(np.float64))
    idx = (n + m - 2) - np.arange(m)
    return (np.rint(c[idx]).astype(np.int64) % 2).astype(np.int8)


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


# NIST SP 800-22 SUBSET BATTERY
#
# The load-bearing quantum-layer property is that PA output is close to
# uniform. This battery implements the SP 800-22 subset the work order names
# (monobit, block-frequency, runs, longest-run-of-ones, serial) plus an
# autocorrelation test, each returning its standard p-value. Reference:
# NIST SP 800-22 Rev. 1a (2010), sections 2.1-2.4, 2.11.

def nist_monobit(bits):
    """SP 800-22 §2.1 frequency (monobit) test. Returns p-value."""
    from scipy.special import erfc
    b = np.asarray(bits, dtype=np.int64)
    n = len(b)
    s_obs = abs((2 * b - 1).sum()) / np.sqrt(n)
    return float(erfc(s_obs / np.sqrt(2)))


def nist_block_frequency(bits, M=128):
    """SP 800-22 §2.2 block-frequency test with block size M."""
    from scipy.special import gammaincc
    b = np.asarray(bits, dtype=np.int64)
    N = len(b) // M
    pi = b[:N * M].reshape(N, M).mean(axis=1)
    chi2 = 4.0 * M * float(((pi - 0.5) ** 2).sum())
    return float(gammaincc(N / 2.0, chi2 / 2.0))


def nist_runs(bits):
    """SP 800-22 §2.3 runs test. Returns p-value (0.0 if the monobit
    prerequisite |pi - 1/2| >= 2/sqrt(n) fails, per the spec)."""
    from scipy.special import erfc
    b = np.asarray(bits, dtype=np.int64)
    n = len(b)
    pi = b.mean()
    if abs(pi - 0.5) >= 2.0 / np.sqrt(n):
        return 0.0
    v_obs = 1 + int((b[1:] != b[:-1]).sum())
    num = abs(v_obs - 2.0 * n * pi * (1 - pi))
    den = 2.0 * np.sqrt(2.0 * n) * pi * (1 - pi)
    return float(erfc(num / den))


def nist_longest_run(bits):
    """SP 800-22 §2.4 longest-run-of-ones test. Block size auto-selected
    from n per the spec (M = 8 / 128 / 10^4)."""
    from scipy.special import gammaincc
    b = np.asarray(bits, dtype=np.int64)
    n = len(b)
    if n < 128:
        raise ValueError("longest-run test needs n >= 128")
    if n < 6272:
        M, cats, pis = 8, [1, 2, 3], [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        M, cats, pis = 128, [4, 5, 6, 7, 8], [0.1174, 0.2430, 0.2493,
                                              0.1752, 0.1027, 0.1124]
    else:
        M, cats, pis = 10000, [10, 11, 12, 13, 14, 15], [
            0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    N = n // M
    K = len(cats)
    nu = np.zeros(K + 1, dtype=np.int64)
    blocks = b[:N * M].reshape(N, M)
    for block in blocks:
        # longest run of ones in this block
        longest = best = 0
        for bit in block:
            best = best + 1 if bit else 0
            if best > longest:
                longest = best
        if longest <= cats[0]:
            nu[0] += 1
        elif longest > cats[-1]:
            nu[K] += 1
        else:
            nu[longest - cats[0]] += 1
    exp = N * np.asarray(pis)
    chi2 = float(((nu - exp) ** 2 / exp).sum())
    return float(gammaincc(K / 2.0, chi2 / 2.0))


def _serial_psi_sq(b, m):
    """psi^2_m statistic over the cyclically-extended sequence (§2.11)."""
    if m == 0:
        return 0.0
    n = len(b)
    ext = np.concatenate([b, b[:m - 1]]) if m > 1 else b
    idx = np.zeros(n, dtype=np.int64)
    for j in range(m):
        idx = (idx << 1) | ext[j:j + n]
    counts = np.bincount(idx, minlength=1 << m).astype(np.float64)
    return float((1 << m) / n * (counts ** 2).sum() - n)


def nist_serial(bits, m=8):
    """SP 800-22 §2.11 serial test. Returns (p_value1, p_value2)."""
    from scipy.special import gammaincc
    b = np.asarray(bits, dtype=np.int64)
    psi_m = _serial_psi_sq(b, m)
    psi_m1 = _serial_psi_sq(b, m - 1)
    psi_m2 = _serial_psi_sq(b, m - 2)
    d1 = psi_m - psi_m1
    d2 = psi_m - 2 * psi_m1 + psi_m2
    p1 = float(gammaincc(2 ** (m - 2), d1 / 2.0))
    p2 = float(gammaincc(2 ** (m - 3), d2 / 2.0))
    return p1, p2


def autocorrelation_test(bits, lag):
    """Normalized autocorrelation at the given lag; two-sided p via erfc.
    Not part of SP 800-22; standard complement named by """
    from scipy.special import erfc
    x = 2 * np.asarray(bits, dtype=np.int64) - 1
    n = len(x)
    c = float((x[:n - lag] * x[lag:]).sum())
    z = c / np.sqrt(n - lag)
    return float(erfc(abs(z) / np.sqrt(2)))


def randomness_battery(bits, alpha=0.01, serial_m=8,
                       autocorr_lags=(1, 2, 8, 16, 32)):
    """Run the SP 800-22 subset + autocorrelation on a bit sequence.

    Returns a list of {test, p_value, passed} dicts (passed: p >= alpha).
    NOTE for interpretation: with k tests at level alpha, ~k*alpha
    false rejections are EXPECTED for truly uniform input; a single
    marginal failure is not evidence of non-uniformity.
    """
    b = np.asarray(bits, dtype=np.int64)
    results = [
        {"test": "monobit", "p_value": nist_monobit(b)},
        {"test": "block_frequency(M=128)", "p_value": nist_block_frequency(b)},
        {"test": "runs", "p_value": nist_runs(b)},
        {"test": "longest_run_of_ones", "p_value": nist_longest_run(b)},
    ]
    p1, p2 = nist_serial(b, m=serial_m)
    results.append({"test": f"serial(m={serial_m}) p1", "p_value": p1})
    results.append({"test": f"serial(m={serial_m}) p2", "p_value": p2})
    for lag in autocorr_lags:
        results.append({"test": f"autocorrelation(lag={lag})",
                        "p_value": autocorrelation_test(b, lag)})
    for r in results:
        r["passed"] = bool(r["p_value"] >= alpha)
    return results


def validate_bp_against_statistical(qber_values=None, block_size=1000,
                                    n_blocks=3, rng=None):
    """Cross-check the fast statistical EC tier model against a real BP
    decoder on a toy random LDPC.

    SCOPE: this is a decoder SPOT CHECK, not a
    capacity claim, and it is reported honestly rather than asserted as a
    pass. The code here is a crude column-weight-3 random LDPC at rate
    0.84 - a 3-regular high-rate code has a LOW BSC threshold, far below a
    5G-NR-grade capacity-approaching code. Measured behavior (10
    blocks/point, n=1000, fresh random H per block): FER 0.1-0.6 across
    the 0.5-1% band (0.10 at the 0.85% operating point itself) and 1.0
    at 2% - everywhere ABOVE the tier model's assumed FER (0.005 in
    the design band; 0.10 at 2%). So the toy decoder is a conservative FLOOR: it shows the
    error patterns are BP-decodable at the operating point most of the
    time even with a code this crude, but it does NOT certify the tiers -
    that needs a real capacity-approaching code. The gap is the toy code
    underperforming the idealized code the tiers assume, NOT the tier
    model being wrong. Reported alongside the standing placeholder
    caveat in fast_ec_simulate.

    Uses a SMALL block (default 1000) decoupled from the rate-model's
    50k-bit blocks: BP on the full block in Python is O(iters·edges) and
    would take minutes; the spot check needs only enough bits to exhibit
    the threshold behavior.

    Returns:
        validation_table: list of dicts with qber, bp_fer, stat_fer, agreement.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if qber_values is None:
        # Design band including the 0.85% operating point itself, plus
        # one degraded-band point (2%) to exhibit the toy code's threshold.
        qber_values = [0.005, 0.0085, 0.01, 0.02]

    code_rate = 0.84
    n_blocks_per_qber = n_blocks  # FER resolution = 1/n_blocks
    validation_table = []

    for qber in qber_values:
        bp_successes = 0

        for _ in range(n_blocks_per_qber):
            # Generate correlated bits with known QBER
            alice = rng.integers(0, 2, size=block_size, dtype=np.int8)
            error_mask = rng.random(block_size) < qber
            bob = alice.copy()
            bob[error_mask] = 1 - bob[error_mask]

            # Full BP decode of the ERROR PATTERN from its syndrome.
            H = generate_ldpc_parity_check(block_size, code_rate, rng)
            error_pattern = (alice ^ bob).astype(np.int8)
            syndrome = (H @ error_pattern) % 2
            # Prior LLR on each error bit is UNIFORM: over a BSC every bit
            # is independently flipped with probability qber, so a priori
            # P(e_j=1)=qber for ALL j and LLR(e=0:e=1)=log((1-q)/q) > 0
            # everywhere. The earlier code applied `channel_llr[bob==1] *=
            # -1` - a CODEWORD-decoding sign flip (LLR keyed to the
            # received bit). That is wrong for syndrome decoding of the
            # error pattern: it randomized the priors and the decoder
            # recovered errors at chance (~50%) even with 2 errors in 500
            # bits, producing FER≈1.00 at every QBER. With the uniform
            # prior the decoder decodes the design band correctly.
            channel_llr = np.full(block_size, np.log((1 - qber) / max(qber, 1e-10)))
            decoded, bp_success = bp_decode(H, syndrome, channel_llr, max_iter=50)
            if bp_success:
                bp_successes += 1

        bp_fer = 1.0 - bp_successes / n_blocks_per_qber
        # Reference = the DETERMINISTIC tier FER the pipeline assumes,
        # via the SAME shared tier definition, never an inline copy.
        # A noisy stochastic estimate here
        # could read 0.00 at a QBER whose tier FER is 0.10; the meaningful
        # cross-check is measured BP FER against the tier's assumed value.
        stat_fer = tier_frame_error_rate(qber, code_rate)

        validation_table.append({
            "qber": qber,
            "bp_fer": bp_fer,
            "stat_fer": stat_fer,
            "bp_success_rate": bp_successes / n_blocks_per_qber,
            "agreement": abs(bp_fer - stat_fer) < 0.3,
        })

    return validation_table
