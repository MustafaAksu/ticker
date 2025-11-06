# src/ot_solver.py
from ott.solvers.linear import unbalanced_sinkhorn
import jax.numpy as jnp

@jit
def solve_uot(p_t, p_tp1, C, epsilon=0.05, rho=0.05):
    geom = pointcloud.PointCloud(C, epsilon=epsilon)
    prob = unbalanced_sinkhorn.UnbalancedSinkhorn(geom, tau=1.0, rho=rho)
    return prob(p_t, p_tp1).matrix