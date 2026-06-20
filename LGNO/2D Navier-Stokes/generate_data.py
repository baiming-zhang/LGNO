import os
import math
import random
import torch
import matplotlib.pyplot as plt


# ======================================================
# 0. random seed
# ======================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ======================================================
# 1. periodic finite differences over the last two spatial dimensions
# ======================================================
def diff_x(f, dx):
    # x corresponds to the second-to-last dimension
    return (torch.roll(f, shifts=-1, dims=-2) - torch.roll(f, shifts=1, dims=-2)) / (
        2.0 * dx
    )


def diff_y(f, dy):
    # y corresponds to the last dimension
    return (torch.roll(f, shifts=-1, dims=-1) - torch.roll(f, shifts=1, dims=-1)) / (
        2.0 * dy
    )


def laplacian(f, dx, dy):
    f_xx = (
        torch.roll(f, shifts=-1, dims=-2) - 2.0 * f + torch.roll(f, shifts=1, dims=-2)
    ) / (dx * dx)
    f_yy = (
        torch.roll(f, shifts=-1, dims=-1) - 2.0 * f + torch.roll(f, shifts=1, dims=-1)
    ) / (dy * dy)
    return f_xx + f_yy


# ======================================================
# 2. solve the Poisson equation with FFT: Delta psi = -omega
# ======================================================
def solve_poisson_fft(omega, Lx, Ly):
    """
    omega: [..., Nx, Ny]
    solve: Deltapsi = -omega
    periodic BC
    """
    device = omega.device
    Nx = omega.shape[-2]
    Ny = omega.shape[-1]

    omega_hat = torch.fft.fft2(omega, dim=(-2, -1))

    kx = 2.0 * math.pi * torch.fft.fftfreq(Nx, d=Lx / Nx).to(device)
    ky = 2.0 * math.pi * torch.fft.fftfreq(Ny, d=Ly / Ny).to(device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2

    psi_hat = torch.zeros_like(omega_hat)
    mask = K2 != 0
    psi_hat[..., mask] = omega_hat[..., mask] / K2[mask]
    psi_hat[..., ~mask] = 0.0

    psi = torch.fft.ifft2(psi_hat, dim=(-2, -1)).real
    return psi


# ======================================================
# 3. derive velocity from the stream function
#    u = dpsi/dy, v = -dpsi/dx
# ======================================================
def velocity_from_vorticity(omega, dx, dy, Lx, Ly):
    psi = solve_poisson_fft(omega, Lx=Lx, Ly=Ly)
    u = diff_y(psi, dy)
    v = -diff_x(psi, dx)
    return u, v, psi


# ======================================================
# 4. generate periodic initial vorticity fields
# ======================================================
def generate_periodic_vorticity_field(X, Y, device, sample_seed):
    """
    Generate periodic initial vorticity that looks more like fluid texture
    """
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)

    field = torch.zeros_like(X)
    L = 2.0

    def periodic_gaussian(X, Y, x0, y0, amp, sigma_x, sigma_y):
        g = torch.zeros_like(X)
        for mx in (-1, 0, 1):
            for my in (-1, 0, 1):
                dxp = X - (x0 + mx * L)
                dyp = Y - (y0 + my * L)
                g += amp * torch.exp(
                    -0.5 * ((dxp / sigma_x) ** 2 + (dyp / sigma_y) ** 2)
                )
        return g

    n_blob = int(torch.randint(6, 12, (1,), device=device).item())
    for _ in range(n_blob):
        x0 = -1.0 + 2.0 * torch.rand(1, device=device).item()
        y0 = -1.0 + 2.0 * torch.rand(1, device=device).item()

        amp = 2.0 + 4.0 * torch.rand(1, device=device).item()
        if torch.rand(1, device=device).item() < 0.5:
            amp = -amp

        sigma_x = 0.08 + 0.18 * torch.rand(1, device=device).item()
        sigma_y = 0.08 + 0.18 * torch.rand(1, device=device).item()

        field += periodic_gaussian(X, Y, x0, y0, amp, sigma_x, sigma_y)

    for _ in range(8):
        kx = int(torch.randint(1, 6, (1,), device=device).item())
        ky = int(torch.randint(1, 6, (1,), device=device).item())
        a = 0.3 + 0.8 * torch.rand(1, device=device).item()
        px = 2.0 * math.pi * torch.rand(1, device=device).item()
        py = 2.0 * math.pi * torch.rand(1, device=device).item()

        field += (
            a
            * torch.sin(kx * math.pi * (X + 1.0) + px)
            * torch.cos(ky * math.pi * (Y + 1.0) + py)
        )

    field = field - torch.mean(field)
    return field


# ======================================================
# 5. Navier-Stokes right-hand side (vorticity form)
#    omega_t + u omega_x + v omega_y = nu lap(omega)
# ======================================================
def rhs_ns(omega, nu, dx, dy, Lx, Ly):
    u, v, _ = velocity_from_vorticity(omega, dx, dy, Lx, Ly)
    omega_x = diff_x(omega, dx)
    omega_y = diff_y(omega, dy)
    lap_omega = laplacian(omega, dx, dy)

    adv = u * omega_x + v * omega_y
    rhs = -adv + nu * lap_omega
    return rhs, u, v


# ======================================================
# 6. Euler time stepping
# ======================================================
def euler_step(omega, dt, nu, dx, dy, Lx, Ly):
    rhs, _, _ = rhs_ns(omega, nu, dx, dy, Lx, Ly)
    omega_next = omega + dt * rhs
    return omega_next


# ======================================================
# 7. recover g = f - grad p from the true velocity trajectory
#    for the current data, f = 0, so g = -grad p
#
#    Velocity equation:
#    u_t + u u_x + v u_y = nu Delta u + g_u
#    v_t + u v_x + v v_y = nu Delta v + g_v
#
#    therefore
#    g_u = u_t + u u_x + v u_y - nu Delta u
#    g_v = v_t + u v_x + v v_y - nu Delta v
# ======================================================
def compute_g_from_uv(u_all, v_all, dt, nu, dx, dy):
    """
    u_all, v_all: [Ns, Nt, 1, Nx, Ny]
    Returns:
        g_u, g_v, dudt, dvdt
        shape same as u_all / v_all
    """
    Ns, Nt, C, Nx, Ny = u_all.shape
    assert C == 1

    g_u = torch.zeros_like(u_all)
    g_v = torch.zeros_like(v_all)
    dudt = torch.zeros_like(u_all)
    dvdt = torch.zeros_like(v_all)

    # use a forward difference to stay consistent with Euler rollout
    u_now = u_all[:, :-1, 0]  # [Ns, Nt-1, Nx, Ny]
    v_now = v_all[:, :-1, 0]
    u_next = u_all[:, 1:, 0]
    v_next = v_all[:, 1:, 0]

    ut = (u_next - u_now) / dt
    vt = (v_next - v_now) / dt

    u_x = diff_x(u_now, dx)
    u_y = diff_y(u_now, dy)
    v_x = diff_x(v_now, dx)
    v_y = diff_y(v_now, dy)

    lap_u = laplacian(u_now, dx, dy)
    lap_v = laplacian(v_now, dx, dy)

    conv_u = u_now * u_x + v_now * u_y
    conv_v = u_now * v_x + v_now * v_y

    g_u_mid = ut + conv_u - nu * lap_u
    g_v_mid = vt + conv_v - nu * lap_v

    dudt[:, :-1, 0] = ut
    dvdt[:, :-1, 0] = vt
    g_u[:, :-1, 0] = g_u_mid
    g_v[:, :-1, 0] = g_v_mid

    # the last frame has no forward difference, so copy the previous frame to keep shapes consistent
    dudt[:, -1] = dudt[:, -2]
    dvdt[:, -1] = dvdt[:, -2]
    g_u[:, -1] = g_u[:, -2]
    g_v[:, -1] = g_v[:, -2]

    return g_u, g_v, dudt, dvdt


# ======================================================
# 8. save time-series data
#    changed to:
#    - train_data: the first train_frames frames
#    - test_data : the full_frames trajectory starting from the first frame
# ======================================================
def save_ns_dataset(
    omega,
    u,
    v,
    g_u,
    g_v,
    dudt,
    dvdt,
    x,
    y,
    t,
    dt,
    nu,
    base_dir="dataset",
    train_frames=100,
    test_frames=500,
):
    """
    omega: [Ns, Nt, 1, Nx, Ny]
    u,v,g_u,g_v,dudt,dvdt: same shape

    Save rules:
    - only one case
    - train.pt saves the first 100 frames
    - test.pt saves the first 500 frames, i.e. the full trajectory starting from the first frame
    """
    train_dir = os.path.join(base_dir, "train_data")
    test_dir = os.path.join(base_dir, "test_data")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    assert omega.shape[0] == 1, "This version saves only one case as requested."
    Nt_total = omega.shape[1]
    assert (
        train_frames <= Nt_total
    ), f"train_frames={train_frames} exceeds total frames Nt={Nt_total}"
    assert (
        test_frames <= Nt_total
    ), f"test_frames={test_frames} exceeds total frames Nt={Nt_total}"

    metadata = dict(
        equation="2D incompressible Navier-Stokes (velocity form with known g=f-grad p)",
        pde_u="u_t + u*u_x + v*u_y = nu*lap(u) + g_u",
        pde_v="v_t + u*v_x + v*v_y = nu*lap(v) + g_v",
        meaning_of_g="g = f - grad(p)",
        forcing="zero in current dataset, so g = -grad(p)",
        bc="periodic",
        domain="[-1,1]^2",
        time_dependent=True,
        incompressible=True,
        dt=float(dt),
        nu=float(nu),
        saved_fields=["omega", "u", "v", "g_u", "g_v", "dudt", "dvdt"],
        time_integrator="Euler",
    )

    # train: first 100 frames
    train_data = dict(
        omega=omega[:, :train_frames],
        u=u[:, :train_frames],
        v=v[:, :train_frames],
        g_u=g_u[:, :train_frames],
        g_v=g_v[:, :train_frames],
        dudt=dudt[:, :train_frames],
        dvdt=dvdt[:, :train_frames],
        x=x,
        y=y,
        t=t[:train_frames],
        metadata=metadata,
    )

    # test: full 1000-frame trajectory starting from the first frame
    test_data = dict(
        omega=omega[:, :test_frames],
        u=u[:, :test_frames],
        v=v[:, :test_frames],
        g_u=g_u[:, :test_frames],
        g_v=g_v[:, :test_frames],
        dudt=dudt[:, :test_frames],
        dvdt=dvdt[:, :test_frames],
        x=x,
        y=y,
        t=t[:test_frames],
        metadata=metadata,
    )

    torch.save(train_data, os.path.join(train_dir, "train.pt"))
    torch.save(test_data, os.path.join(test_dir, "test.pt"))

    print("Dataset saved.")
    print("Train omega shape:", train_data["omega"].shape)
    print("Train u     shape:", train_data["u"].shape)
    print("Train v     shape:", train_data["v"].shape)
    print("Train g_u   shape:", train_data["g_u"].shape)
    print("Train g_v   shape:", train_data["g_v"].shape)
    print("Train dudt  shape:", train_data["dudt"].shape)
    print("Train dvdt  shape:", train_data["dvdt"].shape)

    print("Test  omega shape:", test_data["omega"].shape)
    print("Test  u     shape:", test_data["u"].shape)
    print("Test  v     shape:", test_data["v"].shape)
    print("Test  g_u   shape:", test_data["g_u"].shape)
    print("Test  g_v   shape:", test_data["g_v"].shape)
    print("Test  dudt  shape:", test_data["dudt"].shape)
    print("Test  dvdt  shape:", test_data["dvdt"].shape)


# ======================================================
# 9. save preview figures
# ======================================================
def save_preview(
    omega_all,
    u_all,
    v_all,
    g_u_all,
    g_v_all,
    x,
    y,
    t,
    save_dir="dataset_preview",
    sample_id=0,
    step_stride=50,
):
    """
    Save one preview figure every step_stride frames
    """
    os.makedirs(save_dir, exist_ok=True)
    extent = [x.min().item(), x.max().item(), y.min().item(), y.max().item()]

    Ns, Nt, _, Nx, Ny = omega_all.shape
    sample_id = min(sample_id, Ns - 1)

    # generate the frame indices to save
    frame_ids = list(range(0, Nt, step_stride))
    if (Nt - 1) not in frame_ids:
        frame_ids.append(Nt - 1)

    for k in frame_ids:
        if k >= Nt:
            continue

        omega = omega_all[sample_id, k, 0]
        u = u_all[sample_id, k, 0]
        v = v_all[sample_id, k, 0]
        g_u = g_u_all[sample_id, k, 0]
        g_v = g_v_all[sample_id, k, 0]

        fields = [
            (omega, "omega"),
            (u, "u"),
            (v, "v"),
            (g_u, "g_u"),
            (g_v, "g_v"),
        ]

        for field, name in fields:
            plt.figure(figsize=(6, 5))
            plt.imshow(
                field.T, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r"
            )
            plt.colorbar()
            plt.title(f"{name}, sample={sample_id}, t={t[k].item():.4f}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"{name}_sample{sample_id}_step{k}.png"), dpi=300
            )
            plt.close()


# ======================================================
# 10. main program
# ======================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # parameters
    # -------------------------
    N = 32
    Nt = 1000  # total frames
    num_samples = 1  # one case only
    nu = 2e-3
    dt = 2e-3
    base_dir = "dataset"

    train_frames = 20  # train_data keeps only the first 20 frames
    test_frames = 1000  # test_data keeps the full 1000-frame trajectory starting from the first frame

    # -------------------------
    # grid
    # -------------------------
    Lx = 2.0
    Ly = 2.0
    dx = Lx / N
    dy = Ly / N

    x = torch.arange(N, device=device) * dx - 1.0
    y = torch.arange(N, device=device) * dy - 1.0
    t = torch.arange(Nt, device=device) * dt
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # -------------------------
    # generate samples
    # -------------------------
    omega_samples = []
    u_samples = []
    v_samples = []

    for s in range(num_samples):
        print(f"\nGenerating sample {s+1}/{num_samples} ...")

        omega = generate_periodic_vorticity_field(
            X, Y, device=device, sample_seed=1234 + s
        )

        omega_traj = []
        u_traj = []
        v_traj = []

        for n in range(Nt):
            u, v, _ = velocity_from_vorticity(omega, dx, dy, Lx, Ly)

            omega_traj.append(omega.detach().cpu())
            u_traj.append(u.detach().cpu())
            v_traj.append(v.detach().cpu())

            if n < Nt - 1:
                omega = euler_step(omega, dt, nu, dx, dy, Lx, Ly)

            if n % 20 == 0 or n == Nt - 1:
                print(
                    f"  step {n:4d}: "
                    f"omega std={omega.std().item():.4e}, "
                    f"u std={u.std().item():.4e}, "
                    f"v std={v.std().item():.4e}"
                )

        omega_traj = torch.stack(omega_traj, dim=0).unsqueeze(1)  # [Nt,1,N,N]
        u_traj = torch.stack(u_traj, dim=0).unsqueeze(1)
        v_traj = torch.stack(v_traj, dim=0).unsqueeze(1)

        omega_samples.append(omega_traj)
        u_samples.append(u_traj)
        v_samples.append(v_traj)

    # [Ns,Nt,1,N,N]
    omega_all = torch.stack(omega_samples, dim=0)
    u_all = torch.stack(u_samples, dim=0)
    v_all = torch.stack(v_samples, dim=0)

    # -------------------------
    # recover g and the time derivative from the true velocity trajectory
    # -------------------------
    g_u_all, g_v_all, dudt_all, dvdt_all = compute_g_from_uv(
        u_all=u_all, v_all=v_all, dt=dt, nu=nu, dx=dx, dy=dy
    )

    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    t_cpu = t.detach().cpu()

    print("\nFinal tensor shapes:")
    print("omega_all =", tuple(omega_all.shape))
    print("u_all     =", tuple(u_all.shape))
    print("v_all     =", tuple(v_all.shape))
    print("g_u_all   =", tuple(g_u_all.shape))
    print("g_v_all   =", tuple(g_v_all.shape))
    print("dudt_all  =", tuple(dudt_all.shape))
    print("dvdt_all  =", tuple(dvdt_all.shape))

    # -------------------------
    # save data
    # -------------------------
    save_ns_dataset(
        omega=omega_all,
        u=u_all,
        v=v_all,
        g_u=g_u_all,
        g_v=g_v_all,
        dudt=dudt_all,
        dvdt=dvdt_all,
        x=x_cpu,
        y=y_cpu,
        t=t_cpu,
        dt=dt,
        nu=nu,
        base_dir=base_dir,
        train_frames=train_frames,
        test_frames=test_frames,
    )

    # -------------------------
    # save preview
    # keep the original preview structure unchanged
    # preview the full 1000-frame data here
    # -------------------------
    # at the end of main(), replace the original save_preview call with:
    save_preview(
        omega_all,
        u_all,
        v_all,
        g_u_all,
        g_v_all,
        x_cpu,
        y_cpu,
        t_cpu,
        save_dir=os.path.join(base_dir, "preview"),
        sample_id=0,
        step_stride=50,  # output once every 50 frames
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
