import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import pdb


def pi(z):
    return jnp.real(z[:z.shape[0]//2])

def pi_inverse(z):
    return jnp.concatenate([z, z[::-1]])

def vandermonde(root, size):
    col = jnp.array([root**(2*i+1) for i in range(size)])
    return jnp.vander(col, increasing=True)

def create_sigma_R_basis(root, size):
    """Creates the basis (sigma(1), sigma(X), ..., sigma(X** N-1))."""
    return vandermonde(root, size).T

def compute_basis_coordinates(root, z):
    """Computes the coordinates of a vector with respect to the orthogonal lattice basis."""
    b = create_sigma_R_basis(root, z.shape[0])
    b_conj = jnp.conjugate(b)
    output = jnp.real(jnp.dot(z, b_conj.T) / (b * b_conj).sum(axis=0))
    return output

def coordinate_wise_random_rounding(key, coordinates):
    """Rounds coordinates randonmly."""
    r = coordinates - jnp.floor(coordinates)
    f = jnp.array([random.choice(key, jnp.array([c, c-1]), (1,), p=jnp.array([1-c, c])) for c in r]).reshape(-1)
    
    rounded_coordinates = (coordinates - f).astype(int)
    return rounded_coordinates

def sigma_R_discretization(key, root, z):
    """Projects a vector on the lattice using coordinate wise random rounding."""
    coordinates = compute_basis_coordinates(root, z)
    rounded_coordinates = coordinate_wise_random_rounding(key, coordinates)
    y = jnp.matmul(vandermonde(root, z.shape[0]), rounded_coordinates)
    return y

def sigma_inverse(b: np.array) -> np.array:
    """Encodes the vector b in a polynomial using an M-th root of unity."""
    A = vandermonde(jnp.exp(2*jnp.pi*1j / (b.shape[0]*2)), b.shape[0])
    coeffs = jnp.linalg.solve(A, b)
    return jnp.array(coeffs)[::-1]

def sigma(p: np.array, N: int) -> np.array:
    """Decodes a polynomial by applying it to the N-th roots of unity."""
    xi = jnp.exp(2*jnp.pi*1j / (2*N))
    roots = jnp.array([xi ** (2 * i + 1) for i in range(N)])
    return jnp.polyval(p, roots)

def encode(key: jax.random.PRNGKey, z: np.array, scale: float) -> np.array:
    """Encodes a vector by expanding it first to H,
    scale it, project it on the lattice of sigma(R), and performs
    sigma inverse.
    """
    pi_z = pi_inverse(z)
    scaled_pi_z = scale * pi_z
    rounded_scale_pi_z = sigma_R_discretization(key, jnp.exp(2*jnp.pi*1j/(2*scaled_pi_z.shape[0])), scaled_pi_z)
    p = sigma_inverse(rounded_scale_pi_z)

    coef = jnp.round(jnp.real(p)).astype(int)
    return coef


def decode(p: np.array, scale: float, N: int) -> np.array:
    """Decodes a polynomial by removing the scale, 
    evaluating on the roots, and project it on C^(N/2)"""
    rescaled_p = p / scale
    z = sigma(rescaled_p, N)
    pi_z = pi(z)
    return pi_z

