"""
Module 1: Bell-pair source and detection.

Models entangled photon pair generation from InAs/GaAs QD (Schimpf et al. 2021),
fiber transmission (optionally via QFC to telecom C-band), and SPAD detection.
Werner state density matrix is used as a tractable noise model for the
partially-mixed Bell state; for near-pure sources (F > 0.95) the qualitative
behavior matches depolarizing-channel models.

Co-located architecture: QD source at one node; Photon 1 measured locally,
Photon 2 frequency-converted (780nm → 1550nm via PPLN, Weber et al. 2019)
and transmitted over standard telecom fiber (0.2 dB/km) to partner node.

This module produces correlated bit pairs at the two endpoints. Under the
trusted-source assumption (Assumption B), the CHSH S-parameter > 2 verifies
that the prepared state is genuinely entangled rather than degraded or
mis-set (source fault / drift). Established concept: an ENTANGLEMENT
WITNESS (Horodecki 1996; Terhal 2000) used as a calibration/QA diagnostic.
It is not self-testing, not device-independent, and not an eavesdropper
bound — there is no eavesdropper in the declared threat model
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
    density matrix, not estimated from sampled measurement statistics — so
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

    # ── Step 1: Pair generation ──
    if n_pairs is None:
        n_pulses = int(source_rep_rate_hz * duration_s)
        n_emitted = rng.binomial(n_pulses, source_brightness)
    else:
        n_emitted = n_pairs
        n_pulses = int(n_pairs / source_brightness) if source_brightness > 0 else n_pairs

    if n_emitted == 0:
        return SourceResult(0, 0, 0, 0, np.array([]), np.array([]), 0, 0, {})

    # ── Step 2: Fiber transmission ──
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
            # QFC stage on remote photon before fiber (780nm → 1550nm via PPLN)
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

    # ── Step 3: Dark count accidentals ──
    # Accidentals = dark counts at A coinciding with real detections at B and vice versa
    dark_rate_per_window = spad_dark_count_hz * (coincidence_window_ns * 1e-9)
    n_accidental = rng.poisson(
        dark_rate_per_window * n_emitted * (alice_efficiency + bob_efficiency)
    )
    n_accidental = min(n_accidental, n_true_coinc // 2)  # Cap at reasonable fraction

    # ── Step 4: Polarization measurement ──
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

    # ── Step 5: Compute metrics ──
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


# ══════════════════════════════════════════════
# SYMMETRIC BEAMSPLITTER-MULTIPLEXED EDGE KEYING
# ══════════════════════════════════════════════
#
# Architecture (matches the cost model: 1 QD + 1 two-way optical switch per
# node per triangle membership):
#
#   Each node hosts ONE QD source serving BOTH of its incident triangle edges.
#   For each emitted pair, Photon 1 is detected locally; Photon 2 is routed by
#   a two-way optical switch (beamsplitter / Pockels cell) to ONE of the node's
#   two neighbours, chosen at random per pair.
#
# Consequence — availability and role symmetry (under the trusted-source
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
#   nodes this nets to ~R per edge — the two-way keying compensates for the
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

    Runs the source model twice — once for each endpoint's QD — with each
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
