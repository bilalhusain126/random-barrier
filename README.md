# Structural Default with a Random Barrier

PI-DeepONet experiments based on Lorig, M. (2024), *Structural Default with a Random Barrier* (draft).

## Setup

A firm value $X_t$ defaults at the first hitting time $\tau$ of a random barrier $Z$, where $Z \perp X$. The key quantity is

$$u(x, \bar{x}) = E\left[e^{-\int_0^\tau \gamma(X_s) ds} \varphi(\tau, X_\tau) \mid X_0 = x,\ Z > \bar{x}\right].$$

Since $Z \perp X$, conditioning on $Z = z$ reduces the problem to a fixed-barrier PDE for $w(x, t, z)$. A PI-DeepONet learns the operator $z \mapsto w(\cdot,\cdot;\,z)$, and $u$ follows by integrating $w$ against $F_{Z \mid Z > \bar{x}}$.

## Notebooks

- `notebooks/lorig_deeponet.ipynb` — BM-with-drift example (Lorig §2): trains PI-DeepONet on the survival PDE, compares to the analytical solution, computes the credit spread $Y = -\frac{1}{t}\log u$.

## Install

```
pip install -r requirements.txt
```
