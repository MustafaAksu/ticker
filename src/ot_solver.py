import numpy as np
import ot  # POT

def _normalize(x: np.ndarray) -> np.ndarray:
    s = x.sum()
    return x / s if s > 0 else x

def solve_sinkhorn_balanced(p_t: np.ndarray, p_tp1: np.ndarray,
                            C: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
    """Dengeli (balanced) entropik Sinkhorn → taşıma matrisi Γ."""
    a, b = _normalize(p_t), _normalize(p_tp1)
    # POT: ot.sinkhorn(a,b,M,reg) → Γ
    return ot.sinkhorn(a=a, b=b, M=C, reg=epsilon)

def solve_sinkhorn_unbalanced(p_t: np.ndarray, p_tp1: np.ndarray,
                              C: np.ndarray, epsilon: float = 0.05,
                              rho: float = 0.05) -> np.ndarray:
    """Dengesiz (unbalanced) Sinkhorn – leakage modelleme."""
    a, b = _normalize(p_t), _normalize(p_tp1)
    # reg_m=rho → marjinal sapma cezası
    return ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=epsilon, reg_m=rho)

def solve_ot(p_t: np.ndarray, p_tp1: np.ndarray, C: np.ndarray,
             epsilon: float = 0.05, rho: float | None = 0.05) -> np.ndarray:
    if rho is None:
        return solve_sinkhorn_balanced(p_t, p_tp1, C, epsilon)
    try:
        return solve_sinkhorn_unbalanced(p_t, p_tp1, C, epsilon, rho)
    except Exception:
        # POT versiyon farkı olur ise balanced’a düş
        return solve_sinkhorn_balanced(p_t, p_tp1, C, epsilon)
