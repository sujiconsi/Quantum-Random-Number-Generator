"""
qrng.py


Purpose/Aim:
1) Builds a simple useful quantum circuit for randomness (H on n qubits, then measure)
2) Turns measurement outcomes into a continuous stream of unbiased bits (via a small cache).
3) Provides unbiased integers in [0, n) using rejection sampling.
4) Stays backend-agnostic: runs on Aer by default; can accept a real device backend.

Why this shape?

- The n-qubit H^n circuit gives a uniform distribution over all bitstrings.
- A small "bit pool" avoids rebuilding and running circuits for every single bit.
- Rejection sampling ensures no modulo bias when mapping bits to [0, n).

Quick start

>>> from src.qrng import random_bits, uniform_int
>>> random_bits(64)          # 64 fresh bits (list of 0/1)
>>> uniform_int(10)          # unbiased integer 0..9

If you're generating a lot of randomness:
>>> from src.qrng import BitPool
>>> pool = BitPool(n_qubits=16, refill_shots=4096)   
# tune as you like
>>> pool.uniform_ints(10, size=1000)                 
# 1,000 numbers in [0,10)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math
import numpy as np

#Qiskit imports and backend selection 
try:
    # Fast path if qiskit-aer is available
    from qiskit_aer import Aer
    _QASM_BACKEND = Aer.get_backend("qasm_simulator")
except Exception:
    # Fallback so examples still run without Aer installed
    from qiskit import BasicAer as Aer  # type: ignore
    _QASM_BACKEND = Aer.get_backend("qasm_simulator")  # type: ignore

from qiskit import QuantumCircuit


#Circuit construction 
def _h_superposition_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Build a minimal H^n circuit and measure in the computational basis.

    Parameters
    ----------
    n_qubits : int
        Number of qubits to put into equal superposition.

    Returns
    -------
    QuantumCircuit
        A circuit with n Hadamards, n classical bits, and measurements.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


#Raw bitstring generation 
def generate_bitstrings(
    n_qubits: int,
    shots: int,
    backend=None,
    seed_simulator: Optional[int] = None,
) -> List[str]:
    """
    Run the H^n circuit for a given number of shots and return raw bitstrings.

    Notes
    
    - Outputs are returned as MSB..LSB bitstrings (e.g. "0101").
    - The order of outcomes is randomized (deterministically if `seed_simulator` is set).

    Parameters

    n_qubits : int
        Number of qubits prepared in equal superposition.
    shots : int
        Number of measurement shots to run.
    backend : qiskit backend, optional
        Defaults to Aer qasm_simulator if not provided.
    seed_simulator : int, optional
        Seed for reproducible shuffling on simulators.

    Returns

    List[str]
        A list of length `shots` with bitstrings of width `n_qubits`.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")

    if backend is None:
        backend = _QASM_BACKEND

    qc = _h_superposition_circuit(n_qubits)
    job = backend.run(qc, shots=shots, seed_simulator=seed_simulator)
    result = job.result()
    counts = result.get_counts(qc)  # dict: bitstring -> frequency

    # Expand counts into a flat list of strings, normalized to width n_qubits.
    out: List[str] = []
    for bitstr, c in counts.items():
        s = bitstr.replace(" ", "").zfill(n_qubits)
        out.extend([s] * c)

    # Randomize order so downstream consumers can't rely on dict ordering.
    rng = np.random.default_rng(seed_simulator)
    rng.shuffle(out)
    return out


#Bit cache & unbiased integers 
@dataclass
class BitPool:
    """
    A small, refillable cache of unbiased bits backed by the H^n circuit.

    Typical usage
    
    >>> pool = BitPool(n_qubits=16, refill_shots=4096)
    >>> pool.get_bits(128)          # 128 fresh bits
    >>> pool.uniform_int(1000)      # unbiased integer in [0, 1000)
    >>> pool.uniform_ints(10, 5)    # 5 unbiased integers in [0, 10)

    Parameters
    
    n_qubits : int, default=16
        Number of qubits per circuit shot (i.e., bits per shot).
    refill_shots : int, default=4096
        How many shots to run each time the buffer needs topping up.
    backend : qiskit backend, optional
        Where to run the circuit. Default is Aer simulator.
    seed_simulator : int, optional
        Seed for deterministic behavior in tests/demos on simulators.
    """

    n_qubits: int = 16
    refill_shots: int = 4096
    backend: Optional[object] = None
    seed_simulator: Optional[int] = None

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = _QASM_BACKEND
        self._buf: List[int] = []  # internal bit buffer as ints 0/1

    #internal 
    def _refill(self) -> None:
        """
        Pull another batch of bitstrings from the backend and push the bits
        into the buffer. Called automatically when the buffer runs low.
        """
        bitstrings = generate_bitstrings(
            self.n_qubits,
            self.refill_shots,
            backend=self.backend,
            seed_simulator=self.seed_simulator,
        )
        self._buf.extend(int(b) for s in bitstrings for b in s)

    #public API 
    def get_bits(self, n_bits: int) -> List[int]:
        """
        Return `n_bits` unbiased bits (as ints 0/1). Refills on demand.
        """
        if n_bits <= 0:
            raise ValueError("n_bits must be positive")
        while len(self._buf) < n_bits:
            self._refill()
        out = self._buf[:n_bits]
        del self._buf[:n_bits]
        return out

    def get_uint(self, k_bits: int) -> int:
        """
        Interpret `k_bits` fresh bits as a big-endian, non-negative integer.
        """
        if k_bits <= 0:
            raise ValueError("k_bits must be positive")
        val = 0
        for b in self.get_bits(k_bits):
            val = (val << 1) | b
        return val

    def uniform_int(self, n: int) -> int:
        """
        Unbiased integer in [0, n) via rejection sampling.

        How it works (short version)
        - Choose k = ceil(log2(n)), so values live in [0, 2^k).
        - Accept only when the k-bit value falls in the largest multiple of n.
        - On accept, return value % n; otherwise, try again (rare).
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if n == 1:
            return 0

        k = math.ceil(math.log2(n))
        M = 1 << k
        limit = (M // n) * n  # largest multiple of n less than 2^k

        while True:
            x = self.get_uint(k)
            if x < limit:
                return x % n  # unbiased

    def uniform_ints(self, n: int, size: int) -> List[int]:
        """
        Vectorized convenience: `size` many unbiased integers in [0, n).
        """
        if size < 0:
            raise ValueError("size must be non-negative")
        return [self.uniform_int(n) for _ in range(size)]


#Module-level convenience singletons 
_default_pool: Optional[BitPool] = None


def default_pool() -> BitPool:
    """
    Lazily create (and reuse) a default BitPool.
    Good for quick scripts and notebooks.
    """
    global _default_pool
    if _default_pool is None:
        _default_pool = BitPool()
    return _default_pool


def random_bits(n_bits: int) -> List[int]:
    """Fetch `n_bits` from the default pool."""
    return default_pool().get_bits(n_bits)


def uniform_int(n: int) -> int:
    """Unbiased integer in [0, n) from the default pool."""
    return default_pool().uniform_int(n)


def uniform_ints(n: int, size: int) -> List[int]:
    """Unbiased integers in [0, n) from the default pool."""
    return default_pool().uniform_ints(n, size)


__all__ = [
    "BitPool",
    "generate_bitstrings",
    "default_pool",
    "random_bits",
    "uniform_int",
    "uniform_ints",
]


#Tiny smoke test when run directly 
if __name__ == "__main__":
    pool = BitPool(n_qubits=12, refill_shots=2048, seed_simulator=123)
    xs = pool.uniform_ints(10, size=5000)
    hist = np.bincount(xs, minlength=10)
    print("mod-10 histogram:", hist.tolist())
