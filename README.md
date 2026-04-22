# random-barrier-credit

Numerical experiments for structural credit default with a random barrier, based on:

> Lorig, M. (2024). *Structural Default with a Random Barrier* (draft, October 2024).

## Overview

Let $X_t$ be a firm value diffusion and $Z > X_0$ a random barrier independent of $X$. Default occurs at $\tau = \inf\{t \geq 0 : X_t = Z\}$. The key quantity is

$$u(x, \bar{x}) = E\!\left(e^{-\int_0^\tau \gamma(X_s)\,ds}\,\varphi(\tau, X_\tau) \;\Big|\; X_0 = x,\; Z > \bar{x}\right).$$

**Approach.** Since $Z \perp X$, conditioning on $Z = z$ reduces the problem to a fixed-barrier survival PDE for $w(x, t, z)$. We solve for $w$ using a physics-informed DeepONet (PI-DeepONet), which learns the operator $z \mapsto w(\cdot,\cdot;\,z)$ directly. The quantity $u$ then follows by integrating $w$ against the conditional distribution of $Z$.

## Notebooks

- `notebooks/lorig_deeponet.ipynb` — PI-DeepONet for the BM-with-drift example (Lorig §2): trains on the survival PDE, compares to the analytical solution, and computes the credit spread $Y(x, \bar{x};\,t) = -\frac{1}{t}\log u$.

## Layout

```
notebooks/   experiments
src/         shared modules
  analytical.py   FPT density and analytical survival (Lorig §2)
  models.py       PI-DeepONet architecture
  training.py     physics-informed training loop
  credit.py       F_Z distributions and credit spread computation
  style.py        publication plot style
```

## Install

```
pip install -r requirements.txt
```
