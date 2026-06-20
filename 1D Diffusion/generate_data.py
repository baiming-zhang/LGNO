import torch
import numpy as np
import random
import os


# ==================================
# 0. random seed
# ==================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==================================
# 1. PDE parameters
# Discrete model:
#   du_i/dt = kappa * (u_{i-1} - 2u_i + u_{i+1}) / dx^2
# periodic boundary
# ==================================
KAPPA = 0.1


# ==================================
# 2. periodic initial-condition generation
# still keeps relatively rich Fourier initial conditions
# ==================================
def generate_periodic_initial_condition(x):
    """
    Generate a strictly periodic initial condition u0(x)
    """
    u0 = torch.zeros_like(x)

    # low-frequency dominated
    n_modes_low = random.randint(2, 5)
    for _ in range(n_modes_low):
        k = random.randint(1, 6)
        A_sin = random.uniform(-1.0, 1.0)
        A_cos = random.uniform(-1.0, 1.0)
        u0 += A_sin * torch.sin(2.0 * np.pi * k * x)
        u0 += A_cos * torch.cos(2.0 * np.pi * k * x)

    # a small amount of mid/high-frequency content
    n_modes_high = random.randint(1, 4)
    for _ in range(n_modes_high):
        k = random.randint(7, 15)
        A_sin = random.uniform(-0.35, 0.35)
        A_cos = random.uniform(-0.35, 0.35)
        u0 += A_sin * torch.sin(2.0 * np.pi * k * x)
        u0 += A_cos * torch.cos(2.0 * np.pi * k * x)

    # single mode with random phase
    if random.random() < 0.8:
        k = random.randint(1, 10)
        A = random.uniform(-1.0, 1.0)
        phi = random.uniform(0.0, 2.0 * np.pi)
        u0 += A * torch.sin(2.0 * np.pi * k * x + phi)

    # amplitude scaling
    scale = random.uniform(0.5, 2.0)
    u0 = scale * u0

    return u0


# ==================================
# 3. periodic discrete Laplacian
# corresponding stencil: [1, -2, 1] / dx^2
# ==================================
def discrete_laplacian_periodic(u, dx):
    """
    u:
      [Nx] or [Nx, Nt]
    Returns:
      discrete Laplacian with the same shape
    """
    return (
        torch.roll(u, shifts=1, dims=0) - 2.0 * u + torch.roll(u, shifts=-1, dims=0)
    ) / (dx**2)


# ==================================
# 4. generate data with the exact same discrete model
# time marching: explicit Euler
#
#   u^{n+1} = u^n + dt * kappa * L_h u^n
#
# and save in sync:
#   ut[:, n]  = kappa * L_h u^n
#   uxx[:, n] = L_h u^n
#
# so target_operator = ut is exactly consistent at the discrete level
# ==================================
def solve_heat_equation_periodic_discrete(x, t, kappa=KAPPA):
    """
    Returns:
        u_full: [Nx, Nt]
        u0:     [Nx]
        ut:     [Nx, Nt-1]   the derivative corresponding to the step from t_n to t_{n+1}
        uxx:    [Nx, Nt-1]
    """
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    Nx = x.shape[0]
    Nt = t.shape[0]

    # Explicit Euler stability note:
    # for the heat equation in periodic 1D, roughly require kappa*dt/dx^2 <= 1/2
    r = kappa * dt / (dx**2)
    if r > 0.5:
        print(
            f"[Warning] Explicit Euler may be unstable: kappa*dt/dx^2 = {r:.4f} > 0.5"
        )

    u0 = generate_periodic_initial_condition(x)

    u = torch.zeros(Nx, Nt, dtype=x.dtype)
    ut = torch.zeros(Nx, Nt - 1, dtype=x.dtype)
    uxx = torch.zeros(Nx, Nt - 1, dtype=x.dtype)

    u[:, 0] = u0

    for n in range(Nt - 1):
        lap = discrete_laplacian_periodic(u[:, n], dx)  # [Nx]
        dudt = kappa * lap

        uxx[:, n] = lap
        ut[:, n] = dudt
        u[:, n + 1] = u[:, n] + dt * dudt

    return u, u0, ut, uxx


# ==================================
# 5. recompute the residual from the discrete solution
# this should be near zero if generation and validation are fully consistent
#
# Note:
# defined using a forward difference
#   (u^{n+1} - u^n)/dt
# to align with the ut used during generation
# ==================================
def compute_pde_residual_periodic_discrete(u, x, t, kappa=KAPPA):
    """
    u: [Nx, Nt]
    Returns:
        u_now:      [Nx, Nt-1]
        residual:   [Nx, Nt-1]
        u_t_disc:   [Nx, Nt-1]
        u_xx_disc:  [Nx, Nt-1]
    """
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # time forward difference, exactly matching the generation method
    u_t_disc = (u[:, 1:] - u[:, :-1]) / dt  # [Nx, Nt-1]

    # spatial discrete second derivative at each time
    u_xx_list = []
    for n in range(u.shape[1] - 1):
        lap = discrete_laplacian_periodic(u[:, n], dx)
        u_xx_list.append(lap)

    u_xx_disc = torch.stack(u_xx_list, dim=1)  # [Nx, Nt-1]
    residual = u_t_disc - kappa * u_xx_disc
    u_now = u[:, :-1]

    return u_now, residual, u_t_disc, u_xx_disc


# ==================================
# 6. generate training data
# the time length here is now Nt-1
# because derivatives are defined on intervals [t_n, t_{n+1})
# ==================================
def generate_training_file(
    N_x=256,
    N_t=200,
    num_samples=100,
    noise_level=0.0,
    output_file="train_data/train.pt",
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # periodic grid: do not include the endpoint 1.0
    x = torch.linspace(0, 1, N_x + 1)[:-1]  # [0,1)
    t = torch.linspace(0, 1, N_t)

    u_list = []
    residual_list = []
    ut_list = []
    uxx_list = []
    u0_list = []

    for i in range(num_samples):
        u, u0, ut, uxx = solve_heat_equation_periodic_discrete(x, t, kappa=KAPPA)

        if noise_level > 0:
            u = u + noise_level * torch.randn_like(u)

        u_now, residual, u_t_disc, u_xx_disc = compute_pde_residual_periodic_discrete(
            u, x, t, kappa=KAPPA
        )

        u_list.append(u_now)
        residual_list.append(residual)
        ut_list.append(u_t_disc)
        uxx_list.append(u_xx_disc)
        u0_list.append(u0)

        if i % 10 == 0:
            max_res = residual.abs().max().item()
            mean_res = residual.abs().mean().item()
            print(
                f"Generating train sample {i}/{num_samples}, "
                f"max residual = {max_res:.3e}, mean residual = {mean_res:.3e}"
            )

    u_train = torch.stack(u_list)[:, None, :, :]  # [B,1,Nx,Nt-1]
    residual_train = torch.stack(residual_list)[:, None, :, :]
    ut_train = torch.stack(ut_list)[:, None, :, :]
    uxx_train = torch.stack(uxx_list)[:, None, :, :]
    u0_train = torch.stack(u0_list)[:, None, :]  # [B,1,Nx]

    torch.save(
        {
            "u_train": u_train,
            "residual_train": residual_train,
            "ut_train": ut_train,
            "uxx_train": uxx_train,
            "target_operator_train": KAPPA * uxx_train,
            "u0_train": u0_train,
            "x": x,
            "t": t[:-1],  # aligned with u_train and ut_train
            "t_full": t,
            "x_full": x,
            "kappa": KAPPA,
            "equation": "du_i/dt = kappa*(u_{i-1}-2u_i+u_{i+1})/dx^2",
            "source_term": "none",
            "boundary_condition": "periodic",
            "time_discretization": "forward_euler",
            "space_discretization": "periodic central difference",
        },
        output_file,
    )

    print("Training file saved:", output_file)
    print("u_train shape =", u_train.shape)
    print("target_operator_train shape =", (KAPPA * uxx_train).shape)


# ==================================
# 7. generate test data
# ==================================
def generate_test_file(
    N_x=256,
    N_t=200,
    num_samples=20,
    output_file="test_data/test.pt",
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    x = torch.linspace(0, 1, N_x + 1)[:-1]
    t = torch.linspace(0, 1, N_t)

    u_list = []
    residual_list = []
    ut_list = []
    uxx_list = []
    u0_list = []

    for i in range(num_samples):
        u, u0, ut, uxx = solve_heat_equation_periodic_discrete(x, t, kappa=KAPPA)

        u_now, residual, u_t_disc, u_xx_disc = compute_pde_residual_periodic_discrete(
            u, x, t, kappa=KAPPA
        )

        u_list.append(u_now)
        residual_list.append(residual)
        ut_list.append(u_t_disc)
        uxx_list.append(u_xx_disc)
        u0_list.append(u0)

        if i % 10 == 0:
            max_res = residual.abs().max().item()
            mean_res = residual.abs().mean().item()
            print(
                f"Generating test sample {i}/{num_samples}, "
                f"max residual = {max_res:.3e}, mean residual = {mean_res:.3e}"
            )

    u_test = torch.stack(u_list)[:, None, :, :]  # [B,1,Nx,Nt-1]
    residual_test = torch.stack(residual_list)[:, None, :, :]
    ut_test = torch.stack(ut_list)[:, None, :, :]
    uxx_test = torch.stack(uxx_list)[:, None, :, :]
    u0_test = torch.stack(u0_list)[:, None, :]

    torch.save(
        {
            "u": u_test,
            "residual": residual_test,
            "ut": ut_test,
            "uxx": uxx_test,
            "target_operator": KAPPA * uxx_test,
            "u0": u0_test,
            "x": x,
            "t": t[:-1],
            "t_full": t,
            "x_full": x,
            "kappa": KAPPA,
            "equation": "du_i/dt = kappa*(u_{i-1}-2u_i+u_{i+1})/dx^2",
            "source_term": "none",
            "boundary_condition": "periodic",
            "time_discretization": "forward_euler",
            "space_discretization": "periodic central difference",
        },
        output_file,
    )

    print("Test file saved:", output_file)
    print("u_test shape =", u_test.shape)


# ==================================
# 8. simple numerical check
# the residual here should be near machine precision
# ==================================
def quick_check():
    x = torch.linspace(0, 1, 16 + 1)[:-1]
    t = torch.linspace(0, 1, 2000)

    u, u0, ut, uxx = solve_heat_equation_periodic_discrete(x, t, kappa=KAPPA)

    u_now, residual, ut_mid, uxx_mid = compute_pde_residual_periodic_discrete(
        u, x, t, kappa=KAPPA
    )

    total = residual.abs().sum().item()
    deriv = ut_mid.abs().sum().item()
    max_res = residual.abs().max().item()
    mean_res = residual.abs().mean().item()

    rel = total / (deriv + 1e-20)

    print("\n===== Quick Check =====")
    print(f"u shape        = {u.shape}")
    print(f"u_now shape    = {u_now.shape}")
    print(f"residual shape = {residual.shape}")
    print(f"total residual = {total:.6e}")
    print(f"deriv total    = {deriv:.6e}")
    print(f"relative total = {rel:.6e}")
    print(f"max residual   = {max_res:.6e}")
    print(f"mean residual  = {mean_res:.6e}")
    print("=======================\n")


# ==================================
# 9. main program
# ==================================
if __name__ == "__main__":
    set_seed(42)

    quick_check()

    generate_training_file(
        N_x=32,
        N_t=300,
        num_samples=1,
        noise_level=0.0,
        output_file="train_data/train.pt",
    )

    generate_test_file(
        N_x=32,
        N_t=300,
        num_samples=6,
        output_file="test_data/test.pt",
    )
