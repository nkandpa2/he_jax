import jax.numpy as jnp
import jax
from jax import random
from utils import concat_polynomials, ring_polymul, ring_polyadd, symmetric_mod

def encrypt(pt, pk, q, n):
    return concat_polynomials(ring_polyadd(pt, pk[0], q, n), pk[1])

def decrypt(ct, sk, q, n):
    return symmetric_mod(ring_polyadd(ct[0], ring_polymul(ct[1], sk, q, n), q, n), q).astype(jnp.int64)

