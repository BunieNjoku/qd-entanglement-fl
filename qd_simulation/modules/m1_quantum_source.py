"""
Module 1: Bell-pair source and detection.

Models entangled photon pair generation from InAs/GaAs QD (Schimpf et al. 2021),
fiber transmission (optionally via QFC to telecom C-band), and SPAD detection.
Werner state density matrix is used as a tractable noise model for the
partially-mixed Bell state; for near-pure sources (F > 0.95) the qualitative
behavior matches depolarizing-channel models.

Co-located architecture: QD source at one node; Photon 1 measured locally,
Photon 2 frequency-converted (780nm -> 1550nm via PPLN, Weber et al. 2019)
and transmitted over standard telecom fiber (0.2 dB/km) to partner node.

This module produces correlated bit pairs at the two endpoints. Under the
trusted-source assumption (Assumption B), the CHSH S-parameter > 2 verifies
that the prepared state is genuinely entangled rather than degraded or
mis-set (source fault / drift). Established concept: an ENTANGLEMENT
WITNESS (Horodecki 1996; Terhal 2000) used as a calibration/QA diagnostic.
It is not self-testing, not device-independent, and not an eavesdropper
bound - there is no eavesdropper in the declared threat model
(source-model secret-key agreement).
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class SourceResult:
    """Results from a quantum source simulation run."""
    coincidence_rate_hz: float
    qber: float
    chsh_s: float
    accidental_ratio: float
    raw_bits_alice: np.ndarray
    raw_bits_bob: np.ndarray
    n_true_coinc: int
    n_accidental: int
    params: dict


def werner_state(fidelity):
    """Construct the 4×4 Werner state density matrix.
    ρ = F|Φ+⟩⟨Φ+| + (1-F)/4 × I₄
    """
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |HH⟩ + |VV⟩
    rho_bell = np.outer(phi_plus, phi_plus.conj())
    rho_mixed = np.eye(4) / 4
    return fidelity * rho_bell + (1 - fidelity) * rho_mixed


def measurement_projector(angle):
    """Polarization measurement projectors for a given waveplate angle.
    Returns projectors for H and V outcomes in the rotated basis.
    """
    cos, sin = np.cos(angle), np.sin(angle)
    h_state = np.array([cos, sin])
    v_state = np.array([-sin, cos])
    return np.outer(h_state, h_state.conj()), np.outer(v_state, v_state.conj())


def two_photon_projector(angle_a, angle_b, outcome_a, outcome_b):
    """Tensor product of single-photon projectors for a joint measurement."""
    projs_a = measurement_projector(angle_a)
    projs_b = measurement_projector(angle_b)
    return np.kron(projs_a[outcome_a], projs_b[outcome_b])


def compute_chsh_s(rho):
    """Compute CHSH S-parameter from density matrix at optimal angles.
    S = |E(a1,b1) - E(a1,b2)| + |E(a2,b1) + E(a2,b2)|
    where E(a,b) = P(same) - P(different)

    SIMULATION HONESTY NOTE: S is computed analytically from the programmed
    density matrix, not estimated from sampled measurement statistics - so
    within this simulation the entanglement-witness check is tautological.
    A hardware implementation must estimate S from the Bell-test duty-cycle
    sample, with finite-statistics uncertainty.
    """
    a_angles = [0, np.pi / 4]                    # 0°, 45°
    b_angles = [np.pi / 8, 3 * np.pi / 8]        # 22.5°, 67.5°

    def correlator(a, b):
        p_same = 0
        for outcome in [0, 1]:
            proj = two_photon_projector(a, b, outcome, outcome)
            p_same += np.real(np.trace(rho @ proj))
        return 2 * p_same - 1  # E = P(same) - P(diff) = 2P(same) - 1

    E = [[correlator(a, b) for b in b_angles] for a in a_angles]
    S = abs(E[0][0] - E[0][1]) + abs(E[1][0] + E[1][1])
    return S


def simulate_source(
    fidelity=0.987,
    source_rep_rate_hz=80e6,
    source_brightness=0.01,
    fiber_length_m=500,
    fiber_loss_db_per_km=3.0,
    coupling_efficiency=0.15,
    spad_efficiency=0.65,
    spad_dark_count_hz=50,
    coincidence_window_ns=2.0,
    pbs_extinction_ratio=1000,
    fiber_birefringence_rad_per_km_per_C=0.1,
    temperature_variation_C=5.0,
    duration_s=1.0,
    n_pairs=None,
    colocated=True,
    local_optics_efficiency=0.90,
    use_qfc=False,
    qfc_efficiency=0.35,
    rng=None,
):
    """Simulate entangled photon pair source with all noise contributions.

    Args:
        duration_s: Simulation duration in seconds (default 1s).
        n_pairs: If set, overrides duration_s to generate exactly this many pairs.
        rng: numpy random generator for reproducibility.

    Returns:
        SourceResult with all metrics and raw bit streams.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Step 1: Pair generation
    if n_pairs is None:
        n_pulses = int(source_rep_rate_hz * duration_s)
        n_emitted = rng.binomial(n_pulses, source_brightness)
    else:
        n_emitted = n_pairs
        n_pulses = int(n_pairs / source_brightness) if source_brightness > 0 else n_pairs

    if n_emitted == 0:
        return SourceResult(0, 0, 0, 0, np.array([]), np.array([]), 0, 0, {})

    # Step 2: Fiber transmission
    fiber_loss_db = fiber_loss_db_per_km * (fiber_length_m / 1000)
    fiber_transmission = 10 ** (-fiber_loss_db / 10)

    if colocated:
        # Co-located architecture: Photon 1 stays local (no fiber loss),
        # Photon 2 traverses fiber to partner node.
        # Alice = local detector: local_optics × spad_eff (no fiber)
        # Bob = remote detector: coupling × [QFC ×] fiber_transmission × spad_eff
        alice_efficiency = local_optics_efficiency * spad_efficiency
        bob_efficiency = coupling_efficiency * fiber_transmission * spad_efficiency
        if use_qfc:
            # QFC stage on remote photon before fiber (780nm -> 1550nm via PPLN)
            bob_efficiency *= qfc_efficiency
    else:
        # Mid-fiber model: both photons traverse fiber (legacy)
        total_link_efficiency = coupling_efficiency * fiber_transmission * spad_efficiency
        if use_qfc:
            total_link_efficiency *= qfc_efficiency
        alice_efficiency = total_link_efficiency
        bob_efficiency = total_link_efficiency

    # Each photon independently survives with its respective probability
    alice_detected = rng.random(n_emitted) < alice_efficiency
    bob_detected = rng.random(n_emitted) < bob_efficiency
    coincidence_mask = alice_detected & bob_detected
    n_true_coinc = int(np.sum(coincidence_mask))

    # Step 3: Dark count accidentals
    # Accidentals = dark counts at A coinciding with real detections at B and vice versa
    dark_rate_per_window = spad_dark_count_hz * (coincidence_window_ns * 1e-9)
    n_accidental = rng.poisson(
        dark_rate_per_window * n_emitted * (alice_efficiency + bob_efficiency)
    )
    n_accidental = min(n_accidental, n_true_coinc // 2)  # Cap at reasonable fraction

    # Step 4: Polarization measurement
    rho = werner_state(fidelity)

    # Fiber birefringence: random rotation angle
    bire_std = (fiber_birefringence_rad_per_km_per_C *
                (fiber_length_m / 1000) * temperature_variation_C)
    birefringence_angle = rng.normal(0, bire_std) if bire_std > 0 else 0

    # Fixed measurement basis (both nodes at 0° + birefringence offset)
    angle_alice = 0.0
    angle_bob = birefringence_angle  # Imperfect compensation

    # Compute outcome probabilities from density matrix
    # P(HH), P(HV), P(VH), P(VV)
    probs = np.zeros(4)
    for oa in range(2):
        for ob in range(2):
            proj = two_photon_projector(angle_alice, angle_bob, oa, ob)
            probs[2 * oa + ob] = max(0, np.real(np.trace(rho @ proj)))
    probs /= probs.sum()  # Normalize

    # Sample outcomes for true coincidences
    outcomes = rng.choice(4, size=n_true_coinc, p=probs)
    alice_bits_true = (outcomes >= 2).astype(np.int8)  # 0=H, 1=V
    bob_bits_true = (outcomes % 2).astype(np.int8)

    # PBS extinction ratio: small probability of wrong output port
    pbs_error_prob = 1.0 / pbs_extinction_ratio
    alice_flips = rng.random(n_true_coinc) < pbs_error_prob
    bob_flips = rng.random(n_true_coinc) < pbs_error_prob
    alice_bits_true[alice_flips] = 1 - alice_bits_true[alice_flips]
    bob_bits_true[bob_flips] = 1 - bob_bits_true[bob_flips]

    # Accidental coincidences: random uncorrelated bits
    alice_bits_acc = rng.integers(0, 2, size=n_accidental, dtype=np.int8)
    bob_bits_acc = rng.integers(0, 2, size=n_accidental, dtype=np.int8)

    # Combine
    alice_bits = np.concatenate([alice_bits_true, alice_bits_acc])
    bob_bits = np.concatenate([bob_bits_true, bob_bits_acc])
    n_total = len(alice_bits)

    # Shuffle (accidentals are interspersed with true coincidences)
    if n_total > 0:
        perm = rng.permutation(n_total)
        alice_bits = alice_bits[perm]
        bob_bits = bob_bits[perm]

    # Step 5: Compute metrics
    qber = float(np.mean(alice_bits != bob_bits)) if n_total > 0 else 0
    coincidence_rate = n_total / duration_s if duration_s > 0 else n_total
    accidental_ratio = n_accidental / max(n_total, 1)

    # CHSH S-parameter (from density matrix, Assumption B)
    chsh_s = compute_chsh_s(rho)

    params = {
        "fidelity": fidelity, "fiber_length_m": fiber_length_m,
        "coupling_efficiency": coupling_efficiency,
        "n_emitted": n_emitted, "n_total_coinc": n_total,
    }

    return SourceResult(
        coincidence_rate_hz=coincidence_rate,
        qber=qber,
        chsh_s=chsh_s,
        accidental_ratio=accidental_ratio,
        raw_bits_alice=alice_bits,
        raw_bits_bob=bob_bits,
        n_true_coinc=n_true_coinc,
        n_accidental=n_accidental,
        params=params,
    )


def compute_qber_analytical(fidelity, pbs_extinction=1000, bire_contribution=0.001):
    """Analytical QBER estimate for quick sweeps.
    QBER ≈ (1-F)/2 + 1/extinction + birefringence_contribution
    """
    return (1 - fidelity) / 2 + 1 / pbs_extinction + bire_contribution


def compute_coincidence_rate(
    source_rep_rate_hz, source_brightness, coupling_efficiency,
    fiber_length_m, fiber_loss_db_per_km, spad_efficiency,
    colocated=True, local_optics_efficiency=0.90,
    use_qfc=False, qfc_efficiency=0.35
):
    """Analytical coincidence rate for quick sweeps.

    Co-located model: Photon 1 stays local (no fiber loss), Photon 2 traverses fiber.
    Mid-fiber model (legacy): both photons traverse fiber.
    With QFC: adds frequency conversion loss on the fiber-traversing photon(s).
    """
    fiber_loss_db = fiber_loss_db_per_km * (fiber_length_m / 1000)
    fiber_trans = 10 ** (-fiber_loss_db / 10)

    if colocated:
        eff_local = local_optics_efficiency * spad_efficiency
        eff_remote = coupling_efficiency * fiber_trans * spad_efficiency
        if use_qfc:
            eff_remote *= qfc_efficiency
        return source_rep_rate_hz * source_brightness * eff_local * eff_remote
    else:
        link_eff = coupling_efficiency * fiber_trans * spad_efficiency
        if use_qfc:
            link_eff *= qfc_efficiency
        return source_rep_rate_hz * source_brightness * link_eff ** 2


# NOISE-MODEL SENSITIVITY AND FINITE-STATISTICS CHSH
#
# The Werner state is the tractable default noise model, but it is not
# physically accurate for QDs (system_description flags depolarizing or
# amplitude damping as closer for F > 0.95). One-sided depolarizing on
# half of |phi+> IS the Werner family - D_p(rho_B) = (1-p)rho + p I/2
# applied to one qubit of a maximally entangled state gives exactly
# werner_state(1-p) in this file's Bell-weight parametrization - so
# the informative alternative is
# amplitude damping, which is non-unital and breaks the e_x = e_z
# symmetry the Werner model enjoys.


def amplitude_damping_state(gamma):
    """|phi+> with a one-sided amplitude-damping channel (Kraus
    operators K0 = diag(1, sqrt(1-g)), K1 = |0><1| sqrt(g)) on the
    fiber photon."""
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_in = np.outer(phi_plus, phi_plus.conj())
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    rho = np.zeros((4, 4))
    for k in (k0, k1):
        op = np.kron(np.eye(2), k)
        rho = rho + op @ rho_in @ op.conj().T
    return rho


def amplitude_damping_gamma_for_fidelity(fidelity):
    """Damping parameter giving <phi+|rho|phi+> = fidelity. From the
    Kraus form, F(g) = ((1 + sqrt(1-g))/2)^2, invertible for
    F >= 1/4: g = 1 - (2 sqrt(F) - 1)^2."""
    if not 0.25 <= fidelity <= 1.0:
        raise ValueError("fidelity must be in [0.25, 1]")
    s = 2 * np.sqrt(fidelity) - 1
    return float(1 - s ** 2)


def qber_in_basis(rho, angle):
    """P(outcomes differ) with both analyzers at `angle` (0 = Z,
    pi/4 = X), from Tr(rho Pi)."""
    p_diff = 0.0
    total = 0.0
    for oa in range(2):
        for ob in range(2):
            p = max(0.0, float(np.real(np.trace(
                rho @ two_photon_projector(angle, angle, oa, ob)))))
            total += p
            if oa != ob:
                p_diff += p
    return p_diff / total if total > 0 else 0.0


def sample_chsh_s(rho, n_events, rng=None, n_repeats=10):
    """CHSH S estimated from finitely many sampled measurement events,
    as a hardware Bell-test duty cycle would: n_events split equally
    over the four analyzer settings, outcomes drawn from Tr(rho Pi),
    correlators from counts. Returns (mean, std) over n_repeats
    independent estimates. Complements compute_chsh_s, which evaluates
    S exactly from the density matrix and therefore cannot fail in
    simulation."""
    if rng is None:
        rng = np.random.default_rng(0)
    a_angles = [0, np.pi / 4]
    b_angles = [np.pi / 8, 3 * np.pi / 8]
    n_per = max(1, n_events // 4)
    setting_probs = []
    for a in a_angles:
        for b in b_angles:
            probs = np.array([max(0.0, float(np.real(np.trace(
                rho @ two_photon_projector(a, b, oa, ob)))))
                for oa in range(2) for ob in range(2)])
            setting_probs.append(probs / probs.sum())
    estimates = []
    for _ in range(n_repeats):
        e = []
        for probs in setting_probs:
            counts = rng.multinomial(n_per, probs)
            same = counts[0] + counts[3]
            e.append((2 * same - n_per) / n_per)
        estimates.append(abs(e[0] - e[1]) + abs(e[2] + e[3]))
    return float(np.mean(estimates)), float(np.std(estimates))


# BBM92 MEASUREMENT LAYER
#
# Same source, fiber, and detection physics as simulate_source; the delta
# is measurement-side only (Bennett, Brassard & Mermin, PRL 68, 557, 1992):
# each site adds a passive basis choice between Z (0 deg) and X (45 deg)
# analyzers, matched-basis events are kept (sifting), and the X-X subset
# estimates the phase error rate of the Z-Z key (squashing argument:
# Beaudry, Moroder & Lutkenhaus, PRL 101, 093601, 2008). Asymmetric basis
# probabilities follow Lo, Chau & Ardehali (J. Cryptology 18, 133, 2005).
# With p_z_alice = p_z_bob = 1.0 this reduces to the fixed-basis
# source-model measurement of simulate_source (keep fraction 1, no test
# set) - exp14 uses that limit as its fixed-basis arm so both protocol
# arms share one sampling path.
#
# The intercept-resend adversary models a tap on the FIBER photon
# (photon 2) only: for a tapped coincidence, Eve measures photon 2 in her
# basis and resends the corresponding eigenstate. Standard analysis
# (Bennett & Brassard 1984; Gisin et al., Rev. Mod. Phys. 74, 145, 2002,
# sec. II.C.3): an Eve fixed in the key basis is invisible to a
# fixed-basis protocol and learns the key; against BBM92 the tapped
# fraction f contributes ~f/2 error in the conjugate basis.

@dataclass
class Bbm92Result:
    """One BBM92 (or fixed-basis, p_z = 1) measurement run on an edge."""
    n_coincidences: int
    n_key: int                 # Z-Z sifted events (key candidates)
    n_test: int                # X-X sifted events (parameter estimation)
    key_bits_alice: np.ndarray
    key_bits_bob: np.ndarray
    e_x_observed: float        # error rate on the X-X test set
    e_z_actual: float          # true Z-Z error rate (hardware finds via EC)
    sifting_keep: float        # (n_key + n_test) / n_coincidences
    eve_fraction: float
    eve_basis_mode: str        # 'z' | 'random' | 'none'
    eve_guess_accuracy: float  # fraction of Bob's key bits Eve predicts
    n_accidental: int
    birefringence_angle: float
    params: dict


def _eigenstate(angle, outcome):
    """Analyzer eigenstate resent by an intercept-resend adversary."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c, s]) if outcome == 0 else np.array([-s, c])


def simulate_bbm92(
    fidelity=0.987,
    source_rep_rate_hz=80e6,
    source_brightness=0.01,
    fiber_length_m=500,
    fiber_loss_db_per_km=3.0,
    coupling_efficiency=0.15,
    spad_efficiency=0.65,
    spad_dark_count_hz=50,
    coincidence_window_ns=2.0,
    pbs_extinction_ratio=1000,
    fiber_birefringence_rad_per_km_per_C=0.1,
    temperature_variation_C=5.0,
    duration_s=1.0,
    n_pairs=None,
    colocated=True,
    local_optics_efficiency=0.90,
    use_qfc=False,
    qfc_efficiency=0.35,
    p_z_alice=0.9,
    p_z_bob=0.9,
    eve_fraction=0.0,
    eve_basis_mode='none',
    rng=None,
):
    """Simulate one edge under BBM92 measurement with optional tap.

    Reuses simulate_source's pair-generation, transmission, dark-count,
    and extinction models bit for bit; only the measurement stage differs.
    p_z_*: probability the passive splitter routes a photon to the Z
    analyzer (0 deg). The X analyzer sits at 45 deg. Bob's analyzers are
    additionally rotated by the sampled birefringence offset, so channel
    rotation errs both bases equally (the e_x = e_z property exp13's
    tests pin analytically).

    eve_basis_mode: 'none', 'z' (Eve fixed in the Z basis - the attack a
    fixed-basis protocol cannot see), or 'random' (basis guessed 50/50
    per photon). eve_fraction is the tapped fraction of fiber photons.

    Returns Bbm92Result. Accidentals join the stream with uniform random
    bases and bits on both sides, as dark counts land in whichever
    analyzer fires.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if not (0.0 <= p_z_alice <= 1.0 and 0.0 <= p_z_bob <= 1.0):
        raise ValueError("basis probabilities must be in [0, 1]")
    if not 0.0 <= eve_fraction <= 1.0:
        raise ValueError("eve_fraction must be in [0, 1]")
    if eve_basis_mode not in ('none', 'z', 'random'):
        raise ValueError("eve_basis_mode must be 'none', 'z' or 'random'")
    if eve_fraction > 0 and eve_basis_mode == 'none':
        eve_basis_mode = 'z'

    # Pair generation and per-photon transmission: identical to
    # simulate_source.
    if n_pairs is None:
        n_pulses = int(source_rep_rate_hz * duration_s)
        n_emitted = rng.binomial(n_pulses, source_brightness)
    else:
        n_emitted = n_pairs

    empty = np.array([], dtype=np.int8)
    if n_emitted == 0:
        return Bbm92Result(0, 0, 0, empty, empty, 0.0, 0.0, 0.0,
                           eve_fraction, eve_basis_mode, 0.0, 0, 0.0, {})

    fiber_loss_db = fiber_loss_db_per_km * (fiber_length_m / 1000)
    fiber_transmission = 10 ** (-fiber_loss_db / 10)
    if colocated:
        alice_efficiency = local_optics_efficiency * spad_efficiency
        bob_efficiency = coupling_efficiency * fiber_transmission * spad_efficiency
        if use_qfc:
            bob_efficiency *= qfc_efficiency
    else:
        total = coupling_efficiency * fiber_transmission * spad_efficiency
        if use_qfc:
            total *= qfc_efficiency
        alice_efficiency = total
        bob_efficiency = total

    alice_detected = rng.random(n_emitted) < alice_efficiency
    bob_detected = rng.random(n_emitted) < bob_efficiency
    n_true_coinc = int(np.sum(alice_detected & bob_detected))

    dark_rate_per_window = spad_dark_count_hz * (coincidence_window_ns * 1e-9)
    n_accidental = rng.poisson(
        dark_rate_per_window * n_emitted * (alice_efficiency + bob_efficiency))
    n_accidental = min(n_accidental, n_true_coinc // 2)

    rho = werner_state(fidelity)
    bire_std = (fiber_birefringence_rad_per_km_per_C *
                (fiber_length_m / 1000) * temperature_variation_C)
    birefringence_angle = rng.normal(0, bire_std) if bire_std > 0 else 0.0

    # Analyzer angles: Alice ideal, Bob offset by the channel rotation.
    a_angles = {0: 0.0, 1: np.pi / 4}                       # 0 = Z, 1 = X
    b_angles = {0: birefringence_angle,
                1: np.pi / 4 + birefringence_angle}

    basis_a = (rng.random(n_true_coinc) >= p_z_alice).astype(np.int8)
    basis_b = (rng.random(n_true_coinc) >= p_z_bob).astype(np.int8)
    tapped = rng.random(n_true_coinc) < eve_fraction
    if eve_basis_mode == 'random':
        eve_basis = rng.integers(0, 2, size=n_true_coinc, dtype=np.int8)
    else:
        eve_basis = np.zeros(n_true_coinc, dtype=np.int8)   # Z

    bits_a = np.zeros(n_true_coinc, dtype=np.int8)
    bits_b = np.zeros(n_true_coinc, dtype=np.int8)
    eve_record = np.full(n_true_coinc, -1, dtype=np.int8)

    # Untapped events: joint outcomes from Tr(rho * Pa x Pb) per basis
    # combination.
    for ba in (0, 1):
        for bb in (0, 1):
            mask = (~tapped) & (basis_a == ba) & (basis_b == bb)
            n_sel = int(np.sum(mask))
            if n_sel == 0:
                continue
            probs = np.zeros(4)
            for oa in range(2):
                for ob in range(2):
                    proj = two_photon_projector(
                        a_angles[ba], b_angles[bb], oa, ob)
                    probs[2 * oa + ob] = max(0, np.real(np.trace(rho @ proj)))
            probs /= probs.sum()
            outcomes = rng.choice(4, size=n_sel, p=probs)
            bits_a[mask] = (outcomes >= 2).astype(np.int8)
            bits_b[mask] = (outcomes % 2).astype(np.int8)

    # Tapped events: Alice-Eve joint outcome from the state, then Bob
    # measures Eve's resent eigenstate.
    if np.any(tapped):
        for ba in (0, 1):
            for be in (0, 1):
                mask = tapped & (basis_a == ba) & (eve_basis == be)
                n_sel = int(np.sum(mask))
                if n_sel == 0:
                    continue
                # Eve taps mid-fiber: she sees the channel rotation too,
                # so her analyzer carries the same offset as Bob's.
                e_angle = b_angles[be]
                probs = np.zeros(4)
                for oa in range(2):
                    for oe in range(2):
                        proj = two_photon_projector(
                            a_angles[ba], e_angle, oa, oe)
                        probs[2 * oa + oe] = max(
                            0, np.real(np.trace(rho @ proj)))
                probs /= probs.sum()
                outcomes = rng.choice(4, size=n_sel, p=probs)
                oa_sel = (outcomes >= 2).astype(np.int8)
                oe_sel = (outcomes % 2).astype(np.int8)
                bits_a[mask] = oa_sel
                eve_record[mask] = oe_sel
                # Resent eigenstate measured by Bob in his basis.
                idx = np.where(mask)[0]
                for bb in (0, 1):
                    sub = idx[basis_b[idx] == bb]
                    if len(sub) == 0:
                        continue
                    p0_proj, _ = measurement_projector(b_angles[bb])
                    for oe in (0, 1):
                        sub_oe = sub[eve_record[sub] == oe]
                        if len(sub_oe) == 0:
                            continue
                        e_vec = _eigenstate(e_angle, oe)
                        p0 = float(np.real(e_vec @ p0_proj @ e_vec))
                        p0 = min(1.0, max(0.0, p0))
                        bits_b[sub_oe] = (
                            rng.random(len(sub_oe)) >= p0).astype(np.int8)

    # Finite PBS extinction: detector-level bit flips, both sides.
    pbs_error_prob = 1.0 / pbs_extinction_ratio
    flips_a = rng.random(n_true_coinc) < pbs_error_prob
    flips_b = rng.random(n_true_coinc) < pbs_error_prob
    bits_a[flips_a] = 1 - bits_a[flips_a]
    bits_b[flips_b] = 1 - bits_b[flips_b]

    # Accidentals: uniform bases and bits, no Eve knowledge.
    if n_accidental > 0:
        acc_basis_a = (rng.random(n_accidental) >= p_z_alice).astype(np.int8)
        acc_basis_b = (rng.random(n_accidental) >= p_z_bob).astype(np.int8)
        basis_a = np.concatenate([basis_a, acc_basis_a])
        basis_b = np.concatenate([basis_b, acc_basis_b])
        bits_a = np.concatenate(
            [bits_a, rng.integers(0, 2, n_accidental, dtype=np.int8)])
        bits_b = np.concatenate(
            [bits_b, rng.integers(0, 2, n_accidental, dtype=np.int8)])
        eve_record = np.concatenate(
            [eve_record, np.full(n_accidental, -1, dtype=np.int8)])
        eve_basis = np.concatenate(
            [eve_basis, np.zeros(n_accidental, dtype=np.int8)])
        tapped = np.concatenate(
            [tapped, np.zeros(n_accidental, dtype=bool)])
        # Accidentals are interspersed with true coincidences in time
        # (same shuffle simulate_source applies).
        perm = rng.permutation(len(bits_a))
        basis_a, basis_b = basis_a[perm], basis_b[perm]
        bits_a, bits_b = bits_a[perm], bits_b[perm]
        eve_record, eve_basis = eve_record[perm], eve_basis[perm]
        tapped = tapped[perm]

    n_total = len(bits_a)

    # Sifting: matched-basis events only.
    key_mask = (basis_a == 0) & (basis_b == 0)
    test_mask = (basis_a == 1) & (basis_b == 1)
    n_key = int(np.sum(key_mask))
    n_test = int(np.sum(test_mask))

    key_a = bits_a[key_mask].copy()
    key_b = bits_b[key_mask].copy()
    e_z = float(np.mean(key_a != key_b)) if n_key > 0 else 0.0
    e_x = float(np.mean(bits_a[test_mask] != bits_b[test_mask])) \
        if n_test > 0 else 0.0

    # Eve's guess of Bob's sifted key: her record where she tapped in the
    # key basis, a coin flip elsewhere.
    if n_key > 0:
        ev = eve_record[key_mask]
        eb = eve_basis[key_mask]
        guess = rng.integers(0, 2, n_key, dtype=np.int8)
        informed = (ev >= 0) & (eb == 0)
        guess[informed] = ev[informed]
        eve_acc = float(np.mean(guess == key_b))
    else:
        eve_acc = 0.5

    return Bbm92Result(
        n_coincidences=n_total,
        n_key=n_key,
        n_test=n_test,
        key_bits_alice=key_a,
        key_bits_bob=key_b,
        e_x_observed=e_x,
        e_z_actual=e_z,
        sifting_keep=(n_key + n_test) / max(n_total, 1),
        eve_fraction=eve_fraction,
        eve_basis_mode=eve_basis_mode if eve_fraction > 0 else 'none',
        eve_guess_accuracy=eve_acc,
        n_accidental=n_accidental,
        birefringence_angle=float(birefringence_angle),
        params={
            "fidelity": fidelity, "fiber_length_m": fiber_length_m,
            "p_z_alice": p_z_alice, "p_z_bob": p_z_bob,
            "n_emitted": n_emitted, "n_true_coinc": n_true_coinc,
        },
    )


# SYMMETRIC BEAMSPLITTER-MULTIPLEXED EDGE KEYING
#
# Architecture (matches the cost model: 1 QD + 1 two-way optical switch per
# node per triangle membership):
#
#   Each node hosts ONE QD source serving BOTH of its incident triangle edges.
#   For each emitted pair, Photon 1 is detected locally; Photon 2 is routed by
#   a two-way optical switch (beamsplitter / Pockels cell) to ONE of the node's
#   two neighbours, chosen at random per pair.
#
# Consequence - availability and role symmetry (under the trusted-source
# model this is an availability/fairness property, NOT a security mechanism):
#
#   Edge (i,j) accumulates shared bits from node i's source (on pairs its
#   switch routes toward j) AND from node j's source (on pairs its switch
#   routes toward i). Every edge is therefore keyed from BOTH endpoints.
#
#   No node is the privileged pair-generator on any edge. Source-hosting
#   confers no asymmetric position, because both endpoints host. There is no
#   external entanglement source that all parties must trust. This holds by
#   construction (hardware layout), not by protocol convention.
#
# Rate consequence:
#
#   A 50/50 switch sends ~R/2 of each source's pairs toward each neighbour.
#   Edge (i,j) receives ~R_i/2 from i plus ~R_j/2 from j. For homogeneous
#   nodes this nets to ~R per edge - the two-way keying compensates for the
#   multiplexing split. Bits from the two directions are CONCATENATED into
#   the edge's pool (they are independent Bell-pair measurements on the same
#   node pair, so both endpoints hold both halves after reconciliation).

def edge_key_rate_symmetric(
    per_source_coincidence_rate_hz,
    switch_split=0.5,
    n_incident_edges=2,
):
    """Per-edge raw coincidence rate under beamsplitter-multiplexed keying.

    Each endpoint's source contributes `switch_split` of its coincidences to
    this edge. Both endpoints contribute, so the edge rate is the sum.

    Args:
        per_source_coincidence_rate_hz: Coincidence rate of ONE node's source
            when all its pairs are routed to a single partner (i.e. the
            unmultiplexed link rate from compute_coincidence_rate()).
        switch_split: Fraction of a source's pairs routed to this edge.
            0.5 for a symmetric 2-way switch.
        n_incident_edges: Edges served by one source (2 in a triangle).

    Returns:
        dict with per-direction and total edge rates.
    """
    if not 0.0 <= switch_split <= 1.0:
        raise ValueError("switch_split must be in [0, 1]")

    # Each endpoint routes `switch_split` of its pairs down this edge.
    rate_from_i = per_source_coincidence_rate_hz * switch_split
    rate_from_j = per_source_coincidence_rate_hz * switch_split

    return {
        "rate_from_endpoint_i_hz": rate_from_i,
        "rate_from_endpoint_j_hz": rate_from_j,
        "edge_total_rate_hz": rate_from_i + rate_from_j,
        # Symmetry witness: fraction of the edge's key material contributed by
        # the node that "hosts" it. 0.5 == perfectly symmetric == no privileged
        # generator. Any deviation from 0.5 is source-asymmetry on this edge.
        "source_symmetry_ratio": rate_from_i / (rate_from_i + rate_from_j)
        if (rate_from_i + rate_from_j) > 0 else 0.0,
        "switch_split": switch_split,
        "n_incident_edges": n_incident_edges,
    }


def simulate_symmetric_edge(
    switch_split=0.5,
    duration_s=1.0,
    rng=None,
    **source_kwargs,
):
    """Simulate one triangle edge keyed from BOTH endpoints.

    Runs the source model twice - once for each endpoint's QD - with each
    contributing `switch_split` of its pairs to this edge. The two resulting
    bit strings are concatenated into the edge's shared pool.

    Both endpoints hold both halves: for pairs sourced at i, node i measured
    Photon 1 locally and j measured Photon 2; for pairs sourced at j, the
    roles reverse. After reconciliation each endpoint holds the full pool.

    Returns a SourceResult for the combined edge, plus a symmetry witness.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Endpoint i's source: only `switch_split` of its pairs come down this edge.
    res_i = simulate_source(
        duration_s=duration_s * switch_split, rng=rng, **source_kwargs
    )
    # Endpoint j's source: same, routed the other way.
    res_j = simulate_source(
        duration_s=duration_s * switch_split, rng=rng, **source_kwargs
    )

    # Concatenate both directions into the edge pool.
    # Convention: "alice" = the endpoint we are accounting for; for pairs
    # sourced at j, that endpoint held the REMOTE photon, so we swap.
    edge_bits_i = np.concatenate([res_i.raw_bits_alice, res_j.raw_bits_bob])
    edge_bits_j = np.concatenate([res_i.raw_bits_bob, res_j.raw_bits_alice])

    n_total = len(edge_bits_i)
    qber = float(np.mean(edge_bits_i != edge_bits_j)) if n_total > 0 else 0.0

    n_i, n_j = len(res_i.raw_bits_alice), len(res_j.raw_bits_alice)
    symmetry = n_i / (n_i + n_j) if (n_i + n_j) > 0 else 0.0

    combined = SourceResult(
        coincidence_rate_hz=n_total / duration_s if duration_s > 0 else n_total,
        qber=qber,
        chsh_s=res_i.chsh_s,  # same state, same source spec
        accidental_ratio=(res_i.n_accidental + res_j.n_accidental) / max(n_total, 1),
        raw_bits_alice=edge_bits_i,
        raw_bits_bob=edge_bits_j,
        n_true_coinc=res_i.n_true_coinc + res_j.n_true_coinc,
        n_accidental=res_i.n_accidental + res_j.n_accidental,
        params={
            **res_i.params,
            "bits_from_endpoint_i": n_i,
            "bits_from_endpoint_j": n_j,
            # 0.5 => neither endpoint is the privileged generator on this edge.
            "source_symmetry_ratio": symmetry,
            "switch_split": switch_split,
        },
    )
    return combined
