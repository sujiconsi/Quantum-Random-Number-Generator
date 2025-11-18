"""
mitigation.py: Lightweight readout-error mitigation (toggleable).

Aim
- Calibrates *assignment matrices* for each qubit
- Builds a global assignment matrix A = A_0 ⊗ A_1 ⊗ ... ⊗ A_n-1.
- Applies the linear correction: p_ideal ≈ A^-1 p_noisy
  (using a stable pseudo-inverse + clipping + renormalization).


Quick start

>>> from src import qrng
>>> from src.mitigation import calibrate_qubit_assignment, build_global_assignment, apply_correction

# Calibrate 1-qubit matrix on your backend
>>> A0 = calibrate_qubit_assignment(qubit=0, backend=qrng._QASM_BACKEND, shots=4000)
>>> A0
array([[0.99, 0.01],
       [0.02, 0.98]])

# For multiple qubits, kron them
>>> A = build_global_assignment([A0, A0])  # two qubits, example

# Correct a noisy probability vector
>>> import numpy as np
>>> p_noisy = np.array([0.49, 0.02, 0.03, 0.46])  # 2-qubit example
>>> p_corr = apply_correction(p_noisy, A)
>>> p_corr, p_corr.sum()
(array([...]), 1.0)

Notes:
- For n > ~8, A is 2^n x 2^n; consider per-qubit or local schemes (out of scope).
- This module focuses on small n (demo & teaching use).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.result import Result  # type: ignore


#Calibration: per-qubit assignment matrices 

def _prepare_and_measure_1q(
    qubit: int,
    state: int,
) -> QuantumCircuit:
    """
    Build a 1-qubit calibration circuit for a specific qubit index.

    state = 0 -> prepare |0>
    state = 1 -> prepare |1> (via X)
    Measures into a single classical bit.
    """
    qc = QuantumCircuit(qubit + 1, 1)  # allocate enough qubits to reach index
    if state == 1:
        qc.x(qubit)
    qc.measure(qubit, 0)
    return qc


def calibrate_qubit_assignment(
    qubit: int,
    backend,
    shots: int = 8192,
    seed_simulator: Optional[int] = None,
) -> np.ndarray:
    """
    Estimate a 2x2 assignment matrix A for one qubit.

    A[i, j] = P(read=j | prepared=i), i,j in {0,1}

    We run two circuits:
      - prepare |0>, measure -> counts0
      - prepare |1>, measure -> counts1
    """
    if shots <= 0:
        raise ValueError("shots must be positive")

    qc0 = _prepare_and_measure_1q(qubit, 0)
    qc1 = _prepare_and_measure_1q(qubit, 1)

    job0 = backend.run(qc0, shots=shots, seed_simulator=seed_simulator)
    job1 = backend.run(qc1, shots=shots, seed_simulator=seed_simulator)
    r0 = job0.result()
    r1 = job1.result()

    def _prob_read_one(res: Result) -> float:
        counts = res.get_counts()
        ones = 0
        total = 0
        for k, v in counts.items():
            k = k.replace(" ", "")
            if k == "1":
                ones += v
            total += v
        return ones / total if total > 0 else 0.0

    p1_given_0 = _prob_read_one(r0)      # read 1 when prepared 0
    p1_given_1 = _prob_read_one(r1)      # read 1 when prepared 1

    # P(read=0 | prep=0) = 1 - p1|0
    # P(read=0 | prep=1) = 1 - p1|1
    A = np.array([
        [1.0 - p1_given_0, p1_given_0],
        [1.0 - p1_given_1, p1_given_1],
    ], dtype=float)
    return A


#Build global assignment matrix (Kronecker product) 

def build_global_assignment(per_qubit: Iterable[np.ndarray]) -> np.ndarray:
    """
    Take an iterable of 2x2 per-qubit assignment matrices and build
    the full 2^n x 2^n matrix via Kronecker products.
    """
    per_qubit = list(per_qubit)
    if not per_qubit:
        raise ValueError("Need at least one 2x2 matrix.")
    A = np.array([[1.0]])
    for Aq in per_qubit:
        Aq = np.asarray(Aq, dtype=float)
        if Aq.shape != (2, 2):
            raise ValueError("Each per-qubit matrix must be 2x2.")
        A = np.kron(A, Aq)
    return A


#Apply linear correction 

def apply_correction(
    p_noisy: np.ndarray,
    A: np.ndarray,
    *,
    rcond: float = 1e-10,
    clip: bool = True,
    renorm: bool = True,
) -> np.ndarray:
    """
    Solve p_ideal ≈ A^{-1} p_noisy using a stable pseudo-inverse.

    Parameters
    
    p_noisy : np.ndarray
        Noisy probability vector of length 2^n (sums ~ 1).
    A : np.ndarray
        Assignment matrix (2^n x 2^n).
    rcond : float
        Cutoff for small singular values in the pseudo-inverse.
    clip : bool
        If True, negative entries are clipped to 0 after inversion.
    renorm : bool
        If True, renormalize the result to sum to 1.

    Returns
    
    np.ndarray
        Corrected probability vector.

    Notes
    
    - The result may still be imperfect if A is poorly conditioned.
    - For display as "corrected counts", you may multiply by total shots and round.
    """
    p_noisy = np.asarray(p_noisy, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if A.shape[0] != p_noisy.shape[0]:
        raise ValueError("A and p_noisy shape mismatch.")

    # Moore-Penrose pseudo-inverse (stable)
    A_pinv = np.linalg.pinv(A, rcond=rcond)
    p_est = A_pinv @ p_noisy

    if clip:
        p_est = np.clip(p_est, 0.0, None)
    if renorm:
        s = p_est.sum()
        if s > 0:
            p_est = p_est / s
    return p_est


#Convenience wrapper (toggleable) 

@dataclass
class Mitigator:
    """
    Toggleable readout mitigator.

    Usage
    
    >>> from src import qrng
    >>> from src.mitigation import Mitigator, calibrate_qubit_assignment, build_global_assignment
    >>> backend = qrng._QASM_BACKEND

    # calibrate per-qubit on indices [0..n-1]
    >>> perq = [calibrate_qubit_assignment(q, backend, shots=8000) for q in range(2)]
    >>> A = build_global_assignment(perq)
    >>> mit = Mitigator(enabled=True, A=A)

    # Later: correct a probability vector
    >>> p_corr = mit.correct(p_noisy)
    """

    enabled: bool = False
    A: Optional[np.ndarray] = None
    rcond: float = 1e-10
    clip: bool = True
    renorm: bool = True

    def correct(self, p_noisy: np.ndarray) -> np.ndarray:
        """
        Either pass-through (if disabled or no A), or apply linear correction.
        """
        if not self.enabled or self.A is None:
            # Safe pass-through
            return np.asarray(p_noisy, dtype=float).reshape(-1)
        return apply_correction(
            p_noisy=np.asarray(p_noisy, dtype=float).reshape(-1),
            A=self.A,
            rcond=self.rcond,
            clip=self.clip,
            renorm=self.renorm,
        )


#Utilities: counts <-> prob vector 

def counts_to_prob_vector(
    counts: Dict[str, int],
    n_qubits: int,
) -> np.ndarray:
    """
    Convert bitstring-keyed counts to a probability vector of length 2^n.

    The bitstrings are assumed MSB..LSB as written (e.g., "0101").
    Missing outcomes are treated as 0.
    """
    size = 1 << n_qubits
    v = np.zeros(size, dtype=float)
    total = 0.0
    for s, c in counts.items():
        k = int(str(s).replace(" ", ""), 2)
        v[k] += c
        total += c
    if total <= 0:
        raise ValueError("Empty counts.")
    return v / total


def prob_vector_to_counts(
    p: np.ndarray,
    shots: int,
    n_qubits: int,
) -> Dict[str, int]:
    """
    Turn a probability vector back into bitstring counts (by rounding).

    Useful for producing "corrected counts" for downstream tools or plots.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size != (1 << n_qubits):
        raise ValueError("Probability vector length does not match 2^n.")
    counts = np.rint(p * shots).astype(int)
    # Adjust rounding error to ensure exact sum = shots
    diff = shots - counts.sum()
    if diff != 0:
        # add/subtract to the largest probabilities
        order = np.argsort(-p)
        for i in range(abs(int(diff))):
            idx = order[i % len(order)]
            counts[idx] += 1 if diff > 0 else -1
    out: Dict[str, int] = {}
    for k, c in enumerate(counts):
        s = format(k, f"0{n_qubits}b")
        out[s] = int(c)
    return out


__all__ = [
    "calibrate_qubit_assignment",
    "build_global_assignment",
    "apply_correction",
    "Mitigator",
    "counts_to_prob_vector",
    "prob_vector_to_counts",
]
"""

Inspired by standard measurement mitigation approaches (e.g., IBM's M3). It aims to be:
- Small: single file; no heavy dependencies beyond numpy + qiskit
- Safe: uses pseudo-inverse and guards against negative entries
- Optional: wrapped it behind a Mitigator(enable=True/False)

"""
