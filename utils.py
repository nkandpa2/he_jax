import jax.numpy as jnp
import numpy as np
import random
import pdb

def concat_polynomials(p1, p2):
    if p1.size < p2.size:
        return jnp.stack((jnp.concatenate((jnp.zeros(p2.size - p1.size, dtype=p1.dtype), p1)), p2))
    else:
        return jnp.stack((p1, jnp.concatenate((jnp.zeros(p1.size - p2.size, dtype=p2.dtype), p2))))

def polydiv(u, v):
    # w has the common type
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    scale = 1. / v[0]
    q = jnp.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.astype(w.dtype) 
    for k in range(0, m-n+1):
        d = scale * r[k]
        q = q.at[k].set(d)
        r = r.at[k:k+n+1].set(r[k:k+n+1] - d*v)
    #while jnp.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
    #    r = r[1:]
    #r = jnp.trim_zeros(r, 'f')
    return q, r

def symmetric_mod(x, q):
    half_q = q // 2
    return jnp.mod(x + half_q, q) - half_q

def polymod(u, n):
    q, r = polydiv(u, cyclotomic(n))
    return r

def cyclotomic(n):
    #assert jnp.allclose(jnp.round(jnp.log2(n)), jnp.log2(n)), f'n is not a power of 2 ({n})'
    poly = jnp.zeros(n+1)
    poly = poly.at[0].set(1)
    poly = poly.at[n].set(1)
    return poly

def ring_polymul(u, v, q, n):
    return jnp.mod(polymod(jnp.polymul(u, v), n), q).astype(jnp.int64)

def ring_polyadd(u, v, q, n):
    return jnp.mod(jnp.polyadd(u, v), q).astype(jnp.int64)

def relin(d0, d1, d2, p, q, n, evk):
    P0 = jnp.round(ring_polymul(d2, evk[0], p*q, n)/p)
    P1 = jnp.round(ring_polymul(d2, evk[1], p*q, n)/p)
    return concat_polynomials(ring_polyadd(d0, P0, q, n), ring_polyadd(d1, P1, q, n))

def rescale(ct, q, delta):
    ct_new = jnp.round(ct/delta).astype(jnp.int64)
    q_new = q // delta
    return ct_new, q_new
