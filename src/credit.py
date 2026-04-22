"""Credit-spread integral and Y(T) for the random-barrier model.

Conditional on no default to date (Z > x_bar, where x_bar is the running min of X):

    u(x, x_bar, T) = integral_{x_bar}^infty w(x, T, z) f_{Z|Z>x_bar}(z) dz
    Y(T) = -(1/T) log u

For exponential Z ~ Exp(eta), memorylessness gives Z|Z>x_bar ~ x_bar + Exp(eta).
"""
import numpy as np


def exponential_f_Z_cond(z_grid, x_bar, eta):
    """Conditional density of Z given Z > x_bar, for Z ~ Exp(eta)."""
    return eta * np.exp(-eta * (z_grid - x_bar))


def compute_u_and_spread(survival_fn, T_arr, x_val, z_grid, f_Z_cond):
    """Compute u(x_val, x_bar, T) and Y(T) = -log u / T over T_arr.

    survival_fn(x, T, z) -> scalar w (must accept python floats).
    """
    T_arr = np.asarray(T_arr, dtype=np.float64)
    u_vals = np.zeros(len(T_arr))
    for i, T_val in enumerate(T_arr):
        w_vals = np.array([float(survival_fn(x_val, T_val, z_val)) for z_val in z_grid])
        u_vals[i] = np.trapz(w_vals * f_Z_cond, z_grid)
    u_vals = np.clip(u_vals, 1e-15, None)
    Y_vals = -(1.0 / T_arr) * np.log(u_vals)
    return u_vals, Y_vals
