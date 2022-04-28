import jax.numpy as jnp
import jax
from jax import random
from utils import concat_polynomials, polymod, ring_polyadd, ring_polymul
import pdb

def sample_hamming(key, shape, norm):
    idx_key, val_key = random.split(key, 2)
    h = jnp.zeros(shape, dtype=jnp.int64).flatten()
    norm = min(h.size, norm)
    indices = random.choice(idx_key, h.size, [norm], replace=False)
    values = (random.bernoulli(val_key, 0.5, [norm]).astype(int) * 2) - 1
    h = h.at[indices].set(values)
    h = h.reshape(shape)
    return h

def sample_triangle(key, shape):
    return random.categorical(key, jnp.log(jnp.array([0.25, 0.5, 0.25])), shape=shape) - 1

def sample_uniform(key, shape, minval, maxval):
    return random.randint(key, shape, minval, maxval, dtype=jnp.int64)

def generate_evaluation_key(random_key, sk, n, p, q):
    a_key, e_key = random.split(random_key, 2)
    a = sample_uniform(a_key, [n], 0, p*q)
    e = sample_triangle(e_key, [n])
    evk = ring_polyadd(ring_polyadd(ring_polymul(-a, sk, p*q, n), e, p*q, n), ring_polymul(sk, sk, p*q, n)*p, p*q, n)
    return concat_polynomials(evk, a)

def generate_secret_key(random_key, n):
    return sample_hamming(random_key, [n], n // 4)

def generate_public_key(random_key, secret_key, n, q):
    coeff_key, error_key = random.split(random_key, 2)
    pk_coeff = sample_uniform(coeff_key, [n], 0, q)
    pk_error = sample_triangle(error_key, [n])
    p0 = ring_polyadd(-ring_polymul(secret_key, pk_coeff, q, n), pk_error, q, n)
    p1 = pk_coeff
    return concat_polynomials(p0, p1)

def keygen(random_key, n, q, p):
    sk_key, pk_key, ek_key = random.split(random_key, 3)
    sk = generate_secret_key(sk_key, n)
    pk = generate_public_key(pk_key, sk, n, q)
    ek = generate_evaluation_key(ek_key, sk, n, p, q)
    return pk, sk, ek
