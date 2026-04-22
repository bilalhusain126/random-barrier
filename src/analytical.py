"""Analytical baselines for the Lorig random-barrier survival problem.

Uses Lorig's first-passage time density (paper page 3) and adaptive
quadrature to compute survival probabilities.
"""
import numpy as np
from scipy.integrate import quad


def lorig_fpt_density(s, x, z, mu, sigma):
    """Lorig's first-passage time density for X_t = x + mu*t + sigma*W_t hitting z from below."""
    d = z - x
    return (d / (s * np.sqrt(2 * np.pi * sigma**2 * s))) * np.exp(-(d - mu * s)**2 / (2 * sigma**2 * s))


def analytical_survival(x, T, z, mu, sigma):
    """Survival w(x, T, z) = 1 - integral_0^T f_tau(s) ds via scipy adaptive quad.

    Hints the integrator with the density peak location s* ~ (z-x)^2/sigma^2,
    which is essential for x close to z where the peak is sharp.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    T = np.atleast_1d(np.asarray(T, dtype=np.float64))
    x, T = np.broadcast_arrays(x, T)
    out = np.zeros_like(x)
    for i in range(out.size):
        xi_, Ti_ = float(x.flat[i]), float(T.flat[i])
        if xi_ >= z:
            out.flat[i] = 0.0
            continue
        s_peak = max((z - xi_)**2 / sigma**2, 1e-8)
        prob_default, _ = quad(
            lorig_fpt_density, 0.0, Ti_,
            args=(xi_, z, mu, sigma),
            points=[s_peak] if 0 < s_peak < Ti_ else None,
            limit=200, epsabs=1e-10, epsrel=1e-10,
        )
        out.flat[i] = max(1.0 - prob_default, 0.0)
    return out.reshape(x.shape)
