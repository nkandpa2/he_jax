import jax
import jax.numpy as jnp
from jax import core
from jax import lax
from jax._src.util import safe_map

import numpy as np
from functools import wraps
import ops
import pdb

def examine_jaxpr(closed_jaxpr):
    jaxpr = closed_jaxpr.jaxpr
    print("invars:", jaxpr.invars)
    print("outvars:", jaxpr.outvars)
    print("constvars:", jaxpr.constvars)
    for eqn in jaxpr.eqns:
        print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
    print()
    print("jaxpr:", jaxpr)

def homomorphic(fun, p=None, q=None, n=None, ek=None, delta=None, n_mult=None, order=None):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        out, q_new = homomorphic_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args, p=p, q=q, n=n, ek=ek, delta=delta, n_mult=n_mult, order=order)
        return out[0], q_new
    return wrapped


homomorphic_registry = {lax.add_p: ops.add_p,
                        lax.sub_p: ops.sub_p,
                        lax.mul_p: ops.mul_p,
                        lax.neg_p: ops.neg_p,
                        lax.exp_p: ops.exp_p,
                        lax.convert_element_type_p: ops.id}

def homomorphic_jaxpr(jaxpr, consts, *args, p=None, q=None, n=None, ek=None, delta=None, n_mult=None, order=None):
    # Mapping from variable -> value
    env = {}
  
    def read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val
    
    # Bind args and consts to environment
    write(core.unitvar, core.unit)
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Loop through equations and evaluate primitives using `bind`
    for eqn in jaxpr.eqns:
        # Read inputs to equation from environment
        invals = safe_map(read, eqn.invars)  
        # `bind` is how a primitive is called
        if eqn.primitive not in homomorphic_registry:
            raise NotImplementedError(f'{eqn.primitive} does not have a registered homomorphic implementation')
        
        #pdb.set_trace()
        outval, q, mult = homomorphic_registry[eqn.primitive](*invals, p=p, q=q, n=n, ek=ek, delta=delta, order=order)
        n_mult -= mult
        if n_mult < 0:
            raise Exception('Used up multiplication budget in homomorphic function')

        # Write the results of the primitive into the environment
        safe_map(write, eqn.outvars, [outval]) 
    # Read the final result of the Jaxpr from the environment
    return safe_map(read, jaxpr.outvars), q

