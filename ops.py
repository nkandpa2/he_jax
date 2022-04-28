import jax
import jax.numpy as jnp
from utils import ring_polymul, ring_polyadd, concat_polynomials, relin, rescale
from encoding import encode
import pdb

##### Add
def add_ct_ct_p(ct1, ct2, q=None, n=None, **kwargs):
    return concat_polynomials(ring_polyadd(ct1[0], ct2[0], q, n), ring_polyadd(ct1[1], ct2[1], q, n)), q, 0

def add_pt_ct_p(pt, ct, q=None, n=None, delta=None, **kwargs):
    pt = encode(jax.random.PRNGKey(0), pt.reshape(-1), delta)
    return concat_polynomials(ring_polyadd(ct[0], pt, q, n), ct[1]), q, 0

def add_p(ct1, ct2, q=None, n=None, delta=None, **kwargs):
    if len(ct1.shape) != 2:
        return add_pt_ct_p(ct1, ct2, q=q, n=n, delta=delta)
    elif len(ct2.shape) != 2:
        return add_pt_ct_p(ct2, ct1, q=q, n=n, delta=delta)
    else:
        return add_ct_ct_p(ct1, ct2, q=q, n=n, delta=delta)

##### Subtract
def sub_ct_ct_p(ct1, ct2, q=None, n=None, **kwargs):
    return concat_polynomials(ring_polyadd(ct1[0], -ct2[0], q, n), ring_polyadd(ct1[1], -ct2[1], q, n)), q, 0

def sub_pt_ct_p(pt, ct, q=None, n=None, delta=None, **kwargs):
    pt = encode(jax.random.PRNGKey(0), pt.reshape(-1), delta)
    return concat_polynomials(ring_polyadd(pt, -ct[0], q, n), ct[1]), q, 0

def sub_ct_pt_p(ct, pt, q=None, n=None, delta=None, **kwargs):
    pt = encode(jax.random.PRNGKey(0), pt.reshape(-1), delta)
    return concat_polynomials(ring_polyadd(ct[0], -pt, q, n), ct[1]), q, 0


def sub_p(ct1, ct2, q=None, n=None, delta=None, **kwargs):
    if len(ct1.shape) != 2:
        return sub_pt_ct_p(ct1, ct2, q=q, n=n, delta=delta)
    elif len(ct2.shape) != 2:
        return sub_ct_pt_p(ct1, ct2, q=q, n=n, delta=delta)
    else:
        return sub_ct_ct_p(ct1, ct2, q=q, n=n, delta=delta)


##### Multiply
def mul_ct_ct_p(ct1, ct2, p=None, q=None, n=None, ek=None, delta=None, **kwargs):
    d0 = ring_polymul(ct1[0], ct2[0], q, n)
    d1 = ring_polyadd(ring_polymul(ct1[0], ct2[1], q, n), ring_polymul(ct1[1], ct2[0], q, n), q, n)
    d2 = ring_polymul(ct1[1], ct2[1], q, n)
    ct_relin = relin(d0, d1, d2, p, q, n, ek)
    ct_rescaled, q_rescaled = rescale(ct_relin, q, delta)
    return ct_rescaled, q_rescaled, 1

def mul_pt_ct_p(pt, ct, q=None, n=None, delta=None, **kwargs):
    pt = encode(jax.random.PRNGKey(0), pt.reshape(-1), delta)
    ct_prod = concat_polynomials(ring_polymul(pt, ct[0], q, n), ring_polymul(pt, ct[1], q, n))
    ct_rescaled, q_rescaled = rescale(ct_prod, q, delta)
    return ct_rescaled, q_rescaled, 1

def mul_p(ct1, ct2, p=None, q=None, n=None, ek=None, delta=None, **kwargs):
    if len(ct1.shape) != 2:
        return mul_pt_ct_p(ct1, ct2, q=q, n=n, delta=delta)
    elif len(ct2.shape) != 2:
        return mul_pt_ct_p(ct2, ct1, q=q, n=n, delta=delta)
    else:
        return mul_ct_ct_p(ct1, ct2, p=p, q=q, n=n, ek=ek, delta=delta)

##### Negate
def neg_p(ct1, q=None, **kwargs):
    return -ct1, q, 0

##### Special Functions
def id(ct, q=None, **kwargs):
    return ct, q, 0

def exp_p(ct, p=None, q=None, n=None, ek=None, delta=None, order=None, **kwargs):
    n_mult = 0
    ct_accum, q, _ = add_p(jnp.array(1.), ct, q=q, n=n, delta=delta)
    ct_power = ct
    denom = jnp.array(2.)
    q_add = q
    for i in range(2,order+1):
        ct_power, q, _ = mul_p(ct_power, ct_power, p=p, q=q, n=n, delta=delta, ek=ek)
        term, q, _ = mul_p(1/denom, ct_power, p=p, q=q, n=n, delta=delta, ek=ek)
        ct_accum, q, _ = add_p(ct_accum, term, q=q, n=n, delta=delta)
        denom = denom * (i+1)
    return ct_accum, q, order+1 
        
         
    n_mult = 0
    ct1, q, mult= add_p(jnp.array(1), ct, q=q, n=n, delta=delta)
    n_mult += mult

    ct2, q, mult = mul_p(ct, ct, p=p, q=q, n=n, ek=ek, delta=delta)
    n_mult += mult
    ct2_div_2, q, mult = mul_p(ct2, jnp.array(0.5), p=p, q=q, n=n, ek=ek, delta=delta)
    n_mult += mult

    ct_out, q, mult = add_p(ct1, ct2_div_2, q=q, n=n, delta=delta)
    n_mult += mult
    return ct_out, q, n_mult

def log_p(ct):
    pass

def integer_pow_p(ct, power):
    pass 

