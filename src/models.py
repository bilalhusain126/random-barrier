"""Neural-operator architectures for the Lorig survival PDE."""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation=nn.Tanh):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), activation()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class PIDeepONet(nn.Module):
    """PI-DeepONet for w(xi, t; z) on normalised coordinate xi = (x - x_min)/(z - x_min).

    Branch: z -> b(z) in R^p.  Trunk: (xi, t) -> phi(xi, t) in R^p.
    Output: sigmoid(<b, phi> + b0), so w in (0, 1).
    Bias initialised to a positive value so initial output ~ 1, matching the IC.
    """

    def __init__(self, x_min, p=80, branch_hidden=(80, 80, 80), trunk_hidden=(80, 80, 80), bias_init=3.0):
        super().__init__()
        self.x_min = x_min
        self.p = p
        self.branch = MLP(1, list(branch_hidden), p)
        self.trunk = MLP(2, list(trunk_hidden), p)
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, xi, t, z):
        b = self.branch(z.unsqueeze(-1))
        phi = self.trunk(torch.stack([xi, t], dim=-1))
        logit = (b * phi).sum(dim=-1) + self.bias
        return torch.sigmoid(logit)

    def forward_physical(self, x, t, z):
        xi = (x - self.x_min) / (z - self.x_min)
        return self.forward(xi, t, z)


@torch.no_grad()
def deeponet_survival(model, x_arr, T_arr, z_val, device='cpu'):
    """Evaluate trained model on numpy arrays at fixed barrier z_val."""
    x_arr = np.atleast_1d(np.asarray(x_arr, dtype=np.float64))
    T_arr = np.atleast_1d(np.asarray(T_arr, dtype=np.float64))
    x_b, T_b = np.broadcast_arrays(x_arr, T_arr)
    xi = torch.tensor((x_b - model.x_min) / (z_val - model.x_min), dtype=torch.float32, device=device).flatten()
    t = torch.tensor(T_b, dtype=torch.float32, device=device).flatten()
    z = torch.full_like(xi, float(z_val))
    return model(xi, t, z).cpu().numpy().reshape(x_b.shape)
