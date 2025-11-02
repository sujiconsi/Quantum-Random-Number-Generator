"""
metrics.py

Core statistical utilities for evaluating QRNG output.

What this module does

- Builds histograms over outcomes (integers or bitstrings).
- Runs a Chi-square test against the uniform distribution.
- Computes KL divergence (with safe smoothing).
- Measures bit-frequency bias (how often 0 vs 1 appear).
- Small helpers to move between counts and probability vectors.

Design choices

- "Uniform" means: over all outcomes in the sample space you specify.
- We accept both integer outcomes and bitstrings ("0101") where helpful.
- KL divergence uses additive epsilon smoothing to avoid log(0).

Quick start

>>> from src.metrics import chi_square_uniform, bit_frequency
>>> counts = {0: 102, 1: 98}                   # example
>>> chi_square_uniform(counts, support_size=2)
ChiSquareResult(stat=0.08..., df=1, pvalue=0.77..., expected=[100, 100])

>>> bit_frequency(["010", "111", "000"])
BitFreq(total_bits=9, ones=5, zeros=4, frac_one=0.555..., frac_zero=0.444...)

Dependencies

- numpy
- scipy (for chi-square p-values), but we keep graceful failure if not present
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union, Optional
import math
import numpy as np

try:
    from scipy.stats import chisquare  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# Types 

CountKey = Union[int, str]  # ints for value-space, str for bitstrings
Counts = Mapping[CountKey, int]


#Helpers: counts / probabilities 

def counts_to_vector(
    counts: Mapping[int, int],
    support_size: int,
) -> np.ndarray:
    """
    Convert integer-keyed counts {k: c} into a length-`support_size` vector
    ordered by index (0..support_size-1). Missing entries are treated as 0.
    """
    v = np.zeros(support_size, dtype=float)
    for k, c in counts.items():
        if 0 <= k < support_size:
            v[k] = float(c)
    return v


def normalize_counts(counts: Counts) -> Dict[CountKey, float]:
    """
    Normalize a counts dict into probabilities. Returns a new dict.
    """
    total = float(sum(counts.values()))
    if total <= 0.0:
        raise ValueError("Cannot normalize empty or zero-total counts.")
    return {k: v / total for k, v in counts.items()}


def bitstrings_to_int_counts(bitstrings: Iterable[str]) -> Dict[int, int]:
    """
    Map bitstrings like "0101" to integer counts (MSB..LSB as written).
    """
    out: Dict[int, int] = {}
    for s in bitstrings:
        k = int(s, 2)
        out[k] = out.get(k, 0) + 1
    return out


def outcome_histogram(
    outcomes: Iterable[int],
    support_size: int,
) -> Dict[int, int]:
    """
    Make a histogram over integer outcomes in [0, support_size).

    Any outcome outside this range is ignored.
    """
    hist: Dict[int, int] = {}
    for x in outcomes:
        if 0 <= x < support_size:
            hist[x] = hist.get(x, 0) + 1
    return hist


#Bit-frequency stats 

@dataclass
class BitFreq:
    total_bits: int
    ones: int
    zeros: int
    frac_one: float
    frac_zero: float


def bit_frequency(bitstrings: Iterable[str]) -> BitFreq:
    """
    Count 0/1 frequency across a collection of bitstrings.

    Returns BitFreq with totals and fractions.
    """
    ones = 0
    zeros = 0
    total = 0
    for s in bitstrings:
        for ch in s:
            if ch == "1":
                ones += 1
            elif ch == "0":
                zeros += 1
            else:
                # Ignore whitespace or stray chars silently
                continue
            total += 1
    if total == 0:
        raise ValueError("No bits found.")
    return BitFreq(
        total_bits=total,
        ones=ones,
        zeros=zeros,
        frac_one=ones / total,
        frac_zero=zeros / total,
    )


#Chi-square uniformity test 

@dataclass
class ChiSquareResult:
    stat: float
    df: int
    pvalue: float
    expected: List[float]


def chi_square_uniform(
    counts: Counts,
    support_size: Optional[int] = None,
) -> ChiSquareResult:
    """
    Chi-square goodness-of-fit against a uniform distribution.

    Parameters
    ----------
    counts : Mapping
        Outcome -> frequency (int). Keys may be ints or bitstrings.
        If keys are bitstrings, we assume full support_size = 2^len(bitstring).
    support_size : int, optional
        Total number of categories to test against (e.g., 2^n).
        If omitted AND keys are ints, we use max(counts)+1 as a heuristic.

    Returns
    
    ChiSquareResult(stat, df, pvalue, expected)

    Notes
    
    - df = (support_size - 1)
    - If SciPy is not installed, pvalue is set to NaN, but the statistic is
      still computed correctly.
    """
    # Infer support size if not given
    int_keys = True
    if support_size is None:
        try:
            kmax = max(int(k) for k in counts.keys())
            support_size = kmax + 1
        except Exception:
            int_keys = False

    if not int_keys or support_size is None:
        # Try to infer from bitstring width
        # Grab the first key and infer width
        k0 = next(iter(counts.keys()))
        if not isinstance(k0, str):
            raise ValueError("support_size could not be inferred; please pass it.")
        width = len(k0.replace(" ", ""))
        support_size = 1 << width

    # Build observed vector
    if all(isinstance(k, int) for k in counts.keys()):
        observed = counts_to_vector(counts, support_size)
    else:
        # bitstring keys -> convert to int first
        tmp = bitstrings_to_int_counts(counts.keys() for _ in [None])  # not used
        # better: convert counts explicitly
        as_int: Dict[int, int] = {}
        for k, c in counts.items():
            as_int[int(str(k).replace(" ", ""), 2)] = c
        observed = counts_to_vector(as_int, support_size)

    total = observed.sum()
    if total <= 0:
        raise ValueError("Empty counts supplied.")

    expected = np.ones(support_size, dtype=float) * (total / support_size)
    # Chi-square statistic
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.nansum((observed - expected) ** 2 / expected)

    if _HAS_SCIPY:
        # scipy.stats.chisquare can compute pvalue given expected frequencies
        res = chisquare(f_obs=observed, f_exp=expected)  # type: ignore
        pvalue = float(res.pvalue)
    else:  # pragma: no cover
        # Without SciPy, approximate using survival function of chi2 is not trivial.
        # We return NaN to signal "stat OK, pvalue unavailable".
        pvalue = float("nan")

    df = support_size - 1
    return ChiSquareResult(
        stat=float(stat),
        df=df,
        pvalue=pvalue,
        expected=expected.tolist(),
    )


#KL divergence 

def kl_divergence(
    p: Sequence[float],
    q: Sequence[float],
    eps: float = 1e-12,
) -> float:
    """
    Compute D_KL(p || q) with additive smoothing.

    We apply: p' = normalize(p) + eps, q' = normalize(q) + eps,
    then renormalize again so they sum to 1, and compute sum p' * log2(p'/q').

    Parameters
    
    p, q : sequences of floats
        Distributions. They do not need to be normalized.
    eps : float
        Smoothing constant to avoid log(0).

    Returns
    
    float
        KL divergence in bits.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape.")

    def _norm(x: np.ndarray) -> np.ndarray:
        s = x.sum()
        if s <= 0:
            raise ValueError("Distribution has zero or negative sum.")
        return x / s

    p = _norm(p) + eps
    q = _norm(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


__all__ = [
    "BitFreq",
    "ChiSquareResult",
    "bit_frequency",
    "chi_square_uniform",
    "kl_divergence",
    "counts_to_vector",
    "normalize_counts",
    "bitstrings_to_int_counts",
    "outcome_histogram",
]
