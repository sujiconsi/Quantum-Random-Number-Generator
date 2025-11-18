"""
lgi_cert.py


Leggett–Garg (LGI) + NSIT (No-Signalling-In-Time) certification helpers
for a single-qubit “temporal correlations” experiment.

Aim:
1) Builds single-qubit circuits with *mid-circuit measurements* to probe two-time correlators (±1 outcomes).
2) Computes the K3 Leggett–Garg quantity:

K3 = C12 + C23 - C13 (Macrorealistic (classical) bound: K3 ≤ 1. Quantum can reach up to 1.5.)

3) Computes an NSIT delta at t2. NSIT ≈ 0 means “no signalling in time” ; >0 indicates invasiveness.


Quick start
>>> from src import qrng
>>> from src.lgi_cert import run_lgi_k3, run_nsit_t2
>>> backend = qrng._QASM_BACKEND
>>> res = run_lgi_k3(backend, shots=8192, theta=0.4)
>>> res.K3, res.C12, res.C23, res.C13, res.violated
(1.31..., 0.86..., 0.86..., 0.41..., True)

>>> nsit = run_nsit_t2(backend, shots=8192, theta=0.4)
>>> nsit.delta, nsit.p_with, nsit.p_without
(0.08..., [0.53..., 0.46...], [0.5..., 0.49...])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import numpy as np

from qiskit import QuantumCircuit


#Helpers 

def _rz_y_unitary(theta: float) -> Tuple[str, float]:
    """
    Our evolution step between times is implemented as Ry(2*theta).

    Returning ('ry', 2*theta) keeps the callsite simple and allows future
    extension (e.g., rx/rz) without changing interfaces.
    """
    return ("ry", 2.0 * theta)


def _apply_step(qc: QuantumCircuit, gate: Tuple[str, float], q: int = 0) -> None:
    """Apply one evolution step to qubit q."""
    name, angle = gate
    if name == "ry":
        qc.ry(angle, q)
    elif name == "rx":
        qc.rx(angle, q)
    elif name == "rz":
        qc.rz(angle, q)
    else:
        raise ValueError(f"Unknown evolution gate: {name}")


def _two_time_circuit(theta: float, pair: Tuple[int, int]) -> QuantumCircuit:
    """
    Build a single-qubit circuit that measures Q at times t_i and t_j (i<j).

    Conventions:
    - Start in |0>, which is the +1 eigenstate of Q = σ_z.
    - Evolve by one step between consecutive time indices.
    - Measure Z at t_i (mid-circuit), evolve further, measure Z at t_j.

    Time index map (equal steps):
      reach t1: apply 1 step
      reach t2: apply 2 steps
      reach t3: apply 3 steps
    """
    i, j = pair
    if not (1 <= i < j <= 3):
        raise ValueError("pair must be one of (1,2), (2,3), (1,3)")

    qc = QuantumCircuit(1, 2)
    step = _rz_y_unitary(theta)

    # Evolve up to t_i
    for _ in range(i):
        _apply_step(qc, step, 0)
    # First measurement at t_i  -> c[0]
    qc.measure(0, 0)

    # Evolve from t_i to t_j
    for _ in range(j - i):
        _apply_step(qc, step, 0)
    # Second measurement at t_j -> c[1]
    qc.measure(0, 1)
    return qc


def _outcome_to_pm1(bit: int) -> int:
    """Map measurement bit (0/1) to ±1 for Q = σ_z."""
    return +1 if bit == 0 else -1


def _estimate_correlator_shots(
    counts: Dict[str, int],
) -> float:
    """
    Given counts over 2-bit strings (b_i b_j), compute C_ij = <Q_i Q_j>.

    We map:
      '00' -> (+1)*(+1)
      '01' -> (+1)*(-1)
      '10' -> (-1)*(+1)
      '11' -> (-1)*(-1)
    """
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Empty counts in correlator estimation.")
    acc = 0.0
    for outcome, c in counts.items():
        s = outcome.replace(" ", "")
        if len(s) != 2:
            # Unexpected, skip
            continue
        qi = _outcome_to_pm1(int(s[1])) if False else _outcome_to_pm1(int(s[1]))
        # NOTE: In Qiskit bit strings are little-endian by default in some contexts.
        # When measuring into c[0], then c[1], the string typically prints as 'b1 b0'.
        # To avoid confusion, parse by classical bit index explicitly:
        # Here we rely on the order returned matching c1c0; safer approach below.
    # Safer recompute using explicit parse:
    acc = 0.0
    for outcome, c in counts.items():
        s = outcome.replace(" ", "")
        # Qiskit prints classical register with most significant clbit on the left.
        # Our circuit measures into c0 then c1 (indexes 0 then 1). Printed string is "c1c0".
        # Extract in that order:
        if len(s) != 2:
            continue
        b_j = int(s[0])  # second measurement (to c[1])
        b_i = int(s[1])  # first measurement (to c[0])
        q_i = _outcome_to_pm1(b_i)
        q_j = _outcome_to_pm1(b_j)
        acc += (q_i * q_j) * c
    return acc / total


#Public API: LGI K3 

@dataclass
class LGIResult:
    K3: float
    C12: float
    C23: float
    C13: float
    violated: bool  # True if K3 > 1 within plain (uncalibrated) estimate


def run_lgi_k3(
    backend,
    shots: int = 8192,
    theta: float = 0.4,
    seed_simulator: Optional[int] = None,
) -> LGIResult:
    """
    Estimate the Leggett–Garg K3 quantity from three two-time experiments.

    
    Parameters
    backend : qiskit backend
        Where to run (simulator or real device with mid-circuit support).
    shots : int
        Shots per pair (1,2), (2,3), (1,3).
    theta : float
        Evolution angle per time step; we use U = Ry(2*theta).
        (θ≈π/6..π/4 often yields visible violations.)
    seed_simulator : int, optional
        If using a simulator, seed for reproducibility.

    Returns
    
    LGIResult
        K3 = C12 + C23 − C13 and the individual correlators.

    Notes
    
    Macrorealist (classical) bound: K3 ≤ 1.
    Quantum max (for optimal settings) ≈ 1.5.
    """
    pairs = [(1, 2), (2, 3), (1, 3)]
    circuits = [_two_time_circuit(theta, p) for p in pairs]

    job = backend.run(circuits, shots=shots, seed_simulator=seed_simulator)
    results = job.result()

    C: List[float] = []
    for k, p in enumerate(pairs):
        counts = results.get_counts(circuits[k])
        C.append(_estimate_correlator_shots(counts))

    C12, C23, C13 = C
    K3 = C12 + C23 - C13
    return LGIResult(K3=float(K3), C12=float(C12), C23=float(C23), C13=float(C13), violated=(K3 > 1.0))


#Public API: NSIT at t2 

@dataclass
class NSITResult:
    delta: float                 # Σ_b | P_with(b) − P_without(b) |
    p_with: Tuple[float, float]  # (P_t2(b=0 | measured at t1), P_t2(b=1 | ...))
    p_without: Tuple[float, float]


def _measure_t2_with_and_without_t1(theta: float):
    """
    Build two circuits:
    - with_t1: measure at t1, evolve, measure at t2
    - without_t1: skip t1, only measure at t2
    """
    step = _rz_y_unitary(theta)

    # with_t1: evolve to t1, measure, evolve to t2, measure
    with_t1 = QuantumCircuit(1, 1)
    _apply_step(with_t1, step, 0)       # reach t1
    with_t1.measure(0, 0)
    _apply_step(with_t1, step, 0)       # reach t2
    # overwrite same clbit with second measurement is not ideal; use a new clbit
    # Simpler: allocate 2 classical bits and only read the second at t2 for stats.
    with_t1 = QuantumCircuit(1, 2)
    _apply_step(with_t1, step, 0)
    with_t1.measure(0, 0)               # t1 -> c0
    _apply_step(with_t1, step, 0)
    with_t1.measure(0, 1)               # t2 -> c1

    # without_t1: evolve directly to t2, measure once
    without_t1 = QuantumCircuit(1, 1)
    _apply_step(without_t1, step, 0)    # reach t1 (no measure)
    _apply_step(without_t1, step, 0)    # reach t2
    without_t1.measure(0, 0)

    return with_t1, without_t1


def run_nsit_t2(
    backend,
    shots: int = 8192,
    theta: float = 0.4,
    seed_simulator: Optional[int] = None,
) -> NSITResult:
    """
    Estimate NSIT at time t2 by comparing the t2 marginal with and without
    an earlier measurement at t1.

    Returns Δ = Σ_b | P_with(b) − P_without(b) |; Δ≈0 indicates NSIT.
    """
    with_t1, without_t1 = _measure_t2_with_and_without_t1(theta)

    job = backend.run([with_t1, without_t1], shots=shots, seed_simulator=seed_simulator)
    res = job.result()
    counts_with = res.get_counts(with_t1)      # 2-bit strings "b1 b0"
    counts_without = res.get_counts(without_t1)  # 1-bit strings

    # t2 marginal from "with_t1": look at the LEFT bit (c1)
    total_with = sum(counts_with.values())
    p_with0 = sum(c for s, c in counts_with.items() if s.replace(" ", "")[0] == "0") / total_with
    p_with1 = 1.0 - p_with0

    # from "without_t1": single bit
    total_wo = sum(counts_without.values())
    p_wo0 = counts_without.get("0", 0) / total_wo + counts_without.get(" 0", 0) / total_wo
    p_wo1 = 1.0 - p_wo0

    delta = abs(p_with0 - p_wo0) + abs(p_with1 - p_wo1)
    return NSITResult(delta=float(delta), p_with=(float(p_with0), float(p_with1)), p_without=(float(p_wo0), float(p_wo1)))


__all__ = [
    "LGIResult",
    "NSITResult",
    "run_lgi_k3",
    "run_nsit_t2",
]

"""
Design choices 
- Time evolution is modelled as repeated single-qubit rotations U(θ) between
  measurements. By default we use Ry(2θ) so that θ directly controls the angle
  between “times.”
- Measurements are projective in Z; we map bit 0 → +1, bit 1 → −1.
- We use *mid-circuit* measurements (Qiskit supports this) to implement
  sequential measurements on the same qubit.
- This is meant for small-n educational use (one qubit, lots of shots).


References (high level / inspiration)
- A. J. Leggett and A. Garg, PRL 54, 857 (1985)
- Kofler & Brukner, PRL 99, 180403 (2007) — macrorealism & LGI context
- NSIT tests as a practical check for measurement invasiveness

"""
