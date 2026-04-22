"""Training loop for PI-DeepONet on the Lorig survival PDE.

PDE in normalised coords xi = (x - x_min) / (z - x_min), L = z - x_min:

    R = w_t + (mu/L) w_xi + (sigma^2 / 2 L^2) w_xixi

Three soft losses: PDE residual, IC w(xi, 0; z) = 1, BC w(1, t; z) = 0.
"""
import torch


def train_pideeponet(
    model, mu, sigma, x_min, T_max, z_min, z_max,
    n_epochs=15000, lr=1e-3,
    n_pde=4096, n_ic=1024, n_bc=1024,
    lambda_ic=50.0, lambda_bc=50.0,
    sched_step=2000, sched_gamma=0.7,
    device='cpu', log_every=500,
):
    """Train model in place. Returns history dict of losses per epoch."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=sched_step, gamma=sched_gamma)

    history = {'pde': [], 'ic': [], 'bc': [], 'total': []}

    for epoch in range(n_epochs):
        # PDE collocation
        xi = torch.rand(n_pde, device=device, requires_grad=True)
        t = torch.rand(n_pde, device=device, requires_grad=True) * T_max
        z = torch.rand(n_pde, device=device) * (z_max - z_min) + z_min
        L = z - x_min

        w = model(xi, t, z)
        w_t = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_xi = torch.autograd.grad(w, xi, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_xixi = torch.autograd.grad(w_xi, xi, grad_outputs=torch.ones_like(w_xi), create_graph=True)[0]
        R = w_t + (mu / L) * w_xi + (0.5 * sigma**2 / L**2) * w_xixi
        loss_pde = (R ** 2).mean()

        # IC: w(xi, 0, z) = 1, exclude xi near 1 to avoid corner conflict
        xi_ic = torch.rand(n_ic, device=device) * 0.95
        t_ic = torch.zeros(n_ic, device=device)
        z_ic = torch.rand(n_ic, device=device) * (z_max - z_min) + z_min
        loss_ic = ((model(xi_ic, t_ic, z_ic) - 1.0) ** 2).mean()

        # BC: w(1, t, z) = 0
        xi_bc = torch.ones(n_bc, device=device)
        t_bc = torch.rand(n_bc, device=device) * T_max + 1e-3
        z_bc = torch.rand(n_bc, device=device) * (z_max - z_min) + z_min
        loss_bc = (model(xi_bc, t_bc, z_bc) ** 2).mean()

        loss = loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        history['pde'].append(loss_pde.item())
        history['ic'].append(loss_ic.item())
        history['bc'].append(loss_bc.item())
        history['total'].append(loss.item())

        if log_every and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f'epoch {epoch:5d}  pde {loss_pde.item():.3e}  ic {loss_ic.item():.3e}  bc {loss_bc.item():.3e}  total {loss.item():.3e}')

    return history
