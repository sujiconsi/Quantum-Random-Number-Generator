"""
viz.py: Matplotlib helpers for clean figures in QRNG

Aim:
- Small, dependency-light (matplotlib only).
- Return (fig, ax) so callers can further customize or save.
- Accept plain dicts/arrays from `metrics.py` and Qiskit counts.


Quick start

>>> from src.viz import plot_counts_histogram, plot_bit_frequency, plot_uniformity_residuals
>>> fig, ax = plot_counts_histogram({"00": 120, "01": 130, "10": 121, "11": 129}, title="2-qubit counts")

>>> from src.metrics import bit_frequency
>>> bf = bit_frequency(["0101", "1111", "0000"])
>>> fig, ax = plot_bit_frequency(bf)

>>> obs = [100, 98, 102, 100]
>>> exp = [100, 100, 100, 100]
>>> fig, ax = plot_uniformity_residuals(obs, exp, title="Residuals")
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union, Optional
import math
import numpy as np
import matplotlib.pyplot as plt

try:
    # Optional, for type hints if user has metrics installed
    from .metrics import BitFreq
except Exception:  # pragma: no cover
    class BitFreq:  # fallback typing shell
        total_bits: int
        ones: int
        zeros: int
        frac_one: float
        frac_zero: float


#Basic helpers 

def _autox_labels(ax, labels: Sequence[str]) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)


#Plots 

def plot_counts_histogram(
    counts: Mapping[Union[int, str], int],
    *,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar chart of outcome counts. Keys may be ints or bitstrings.
    """
    labels = [str(k).replace(" ", "") for k in counts.keys()]
    vals = [int(v) for v in counts.values()]

    fig, ax = plt.subplots()
    ax.bar(range(len(vals)), vals)
    _autox_labels(ax, labels)
    ax.set_ylabel("Counts")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_bit_frequency(
    bf: BitFreq,
    *,
    title: Optional[str] = "Bit-frequency (should be ~50/50)",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 0/1 frequency as a simple two-bar chart.

    Parameters
    
    bf : BitFreq
        Output of src.metrics.bit_frequency(...)
    """
    labels = ["0", "1"]
    vals = [bf.frac_zero, bf.frac_one]

    fig, ax = plt.subplots()
    ax.bar(range(2), vals)
    _autox_labels(ax, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_uniformity_residuals(
    observed: Sequence[float],
    expected: Sequence[float],
    *,
    title: Optional[str] = "Uniformity residuals (observed − expected)",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize how far each category is from uniform expectation.

    Inputs can be counts or probabilities, as long as both use the same scale.
    """
    observed = np.asarray(observed, dtype=float).reshape(-1)
    expected = np.asarray(expected, dtype=float).reshape(-1)
    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have same length.")

    resid = observed - expected
    labels = [str(i) for i in range(len(resid))]

    fig, ax = plt.subplots()
    ax.bar(range(len(resid)), resid)
    _autox_labels(ax, labels)
    ax.axhline(0.0)
    ax.set_ylabel("Observed − Expected")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_password_entropy_curve(
    lengths: Sequence[int],
    alphabet_size: int,
    *,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot H = length * log2(alphabet_size) over a set of lengths.

    Useful for README or demos (“how entropy scales with length”).
    """
    lengths = list(lengths)
    if any(L <= 0 for L in lengths):
        raise ValueError("All lengths must be positive.")
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be >= 2")

    H = [L * math.log2(alphabet_size) for L in lengths]

    fig, ax = plt.subplots()
    ax.plot(lengths, H, marker="o")
    ax.set_xlabel("Password length")
    ax.set_ylabel("Entropy (bits)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_lgi_components(
    C12: float,
    C23: float,
    C13: float,
    K3: float,
    *,
    title: Optional[str] = "LGI components",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the three correlators and K3 side-by-side.
    """
    labels = ["C12", "C23", "C13", "K3"]
    vals = [C12, C23, C13, K3]

    fig, ax = plt.subplots()
    ax.bar(range(len(vals)), vals)
    _autox_labels(ax, labels)
    ax.axhline(1.0, linestyle="--")   # macrorealist bound for K3
    ax.set_ylim(min(-1.0, min(vals) - 0.1), max(1.5, max(vals) + 0.1))
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_nsit_delta(
    delta: float,
    p_with: Sequence[float],
    p_without: Sequence[float],
    *,
    title: Optional[str] = "NSIT @ t2",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar plot for NSIT: compare t2 marginals with/without earlier measurement.
    """
    labels = ["b=0 (with)", "b=1 (with)", "b=0 (w/o)", "b=1 (w/o)"]
    vals = [float(p_with[0]), float(p_with[1]), float(p_without[0]), float(p_without[1])]

    fig, ax = plt.subplots()
    ax.bar(range(len(vals)), vals)
    _autox_labels(ax, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probability")
    if title:
        ax.set_title(f"{title}  —  Δ = {delta:.3f}")
    fig.tight_layout()
    return fig, ax


__all__ = [
    "plot_counts_histogram",
    "plot_bit_frequency",
    "plot_uniformity_residuals",
    "plot_password_entropy_curve",
    "plot_lgi_components",
    "plot_nsit_delta",
]
