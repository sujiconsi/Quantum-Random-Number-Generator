"""
quantum_passwords.py

Aim:
1) Defines a sensible default alphabet (Base94: printable ASCII).
2) Provides a PasswordGenerator that draws unbiased indices from a BitPool.
3) Exposes simple helpers to make passwords and estimate their entropy.

Note:
- Unbiased indices come from `qrng.BitPool.uniform_int`, which uses
  rejection sampling under the hood
- Entropy =  length * log2(alphabet_size).

Quick start
>>> from src.quantum_passwords import make_password, estimate_entropy_bits
>>> make_password(16)                      
# 16 chars from Base94
>>> estimate_entropy_bits(16)              
# ~105 bits for Base94

Custom alphabet:
>>> ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
>>> from src.quantum_passwords import PasswordGenerator
>>> gen = PasswordGenerator(alphabet=ALNUM)
>>> gen.passwords(length=12, count=5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math

from . import qrng


#Alphabets 
# Printable ASCII Base94: characters 33 ('!') through 126 ('~')
BASE94 = "".join(chr(c) for c in range(33, 127))


#Password generator 
@dataclass
class PasswordGenerator:
    """
    Generate unbiased passwords over a chosen alphabet.

    Parameters
    
    alphabet : str, default=BASE94
        Characters to sample from. Must have length >= 2.
    pool : qrng.BitPool, optional
        Source of unbiased indices. Defaults to the module's default BitPool.

    Examples
    
    >>> gen = PasswordGenerator()
    >>> gen.password(16)
    'N?iX...'
    >>> gen.entropy_bits(16)
    104.6...

    With a custom alphabet:
    >>> ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    >>> PasswordGenerator(alphabet=ALNUM).password(20)
    'aZ3...'
    """

    alphabet: str = BASE94
    pool: Optional[qrng.BitPool] = None

    def __post_init__(self) -> None:
        if self.pool is None:
            self.pool = qrng.default_pool()
        if len(self.alphabet) < 2:
            raise ValueError("Alphabet must contain at least 2 characters.")

    #Introspection 
    @property
    def alpha_size(self) -> int:
        """Number of symbols in the active alphabet."""
        return len(self.alphabet)

    @property
    def per_char_entropy_bits(self) -> float:
        """Entropy contributed by each character: log2(|alphabet|)."""
        return math.log2(self.alpha_size)

    #Estimates 
    def entropy_bits(self, length: int) -> float:
        """Estimate total entropy for a password of `length`."""
        if length <= 0:
            raise ValueError("length must be positive")
        return length * self.per_char_entropy_bits

    #Generation 
    def password(self, length: int) -> str:
        """
        Create one unbiased password of given length.

        How it works
        
        1) Draw `length` many unbiased integers in [0, |alphabet|).
        2) Map each index to a character.
        """
        if length <= 0:
            raise ValueError("length must be positive")
        idxs = self.pool.uniform_ints(self.alpha_size, size=length)
        return "".join(self.alphabet[i] for i in idxs)

    def passwords(self, length: int, count: int) -> List[str]:
        """Create `count` passwords (each of length `length`)."""
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.password(length) for _ in range(count)]


#Convenience helpers 
def make_password(
    length: int,
    alphabet: str = BASE94,
    pool: Optional[qrng.BitPool] = None,
) -> str:
    """
    One-shot helper to generate a password without creating a class instance.
    """
    gen = PasswordGenerator(alphabet=alphabet, pool=pool)
    return gen.password(length)


def estimate_entropy_bits(length: int, alphabet_size: int = len(BASE94)) -> float:
    """
    Estimate password entropy (in bits): H = length * log2(alphabet_size).
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be >= 2")
    return length * math.log2(alphabet_size)


__all__ = [
    "BASE94",
    "PasswordGenerator",
    "make_password",
    "estimate_entropy_bits",
]


# Tiny demo when run directly 
if __name__ == "__main__":
    gen = PasswordGenerator()
    pwd = gen.password(16)
    print("Password:", pwd)
    print("Entropy (bits) ~", round(gen.entropy_bits(16), 2))
