import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


# ======================================================
# 0. random seed
# ======================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ======================================================
# 1. global parameters
# ======================================================
Lx = 4.0
Ly = 4.0
Lz = 4.0

omega_x = 1.0
omega_y = 1.5
omega_z = 0.8

alpha = 0.3


# ======================================================
# 2. potential field
# ======================================================
def potential(X, Y, Z):
    V_harm = 0.5 * (omega_x**2 * X**2 + omega_y**2 * Y**2 + omega_z**2 * Z**2)

    V_pert = alpha * torch.sin(2 * X) * torch.cos(3 * Y) * torch.sin(Z)

    return V_harm + V_pert


# ======================================================
# 3. initial condition: Gaussian wave packet
# ======================================================
def initial_wavepacket(X, Y, Z):
    sigma = 0.8
    k0 = 2.0

    psi0 = torch.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2)) * torch.exp(1j * k0 * X)

    return psi0


# ======================================================
# 4. 3D central differences (periodic boundary)
# ======================================================
def diff_x(u, dx):
    return (torch.roll(u, shifts=-1, dims=0) - torch.roll(u, shifts=1, dims=0)) / (
        2.0 * dx
    )


def diff_y(u, dy):
    return (torch.roll(u, shifts=-1, dims=1) - torch.roll(u, shifts=1, dims=1)) / (
        2.0 * dy
    )


def diff_z(u, dz):
    return (torch.roll(u, shifts=-1, dims=2) - torch.roll(u, shifts=1, dims=2)) / (
        2.0 * dz
    )


def laplacian_3d(u, dx, dy, dz):
    u_xx = (
        torch.roll(u, shifts=-1, dims=0) - 2.0 * u + torch.roll(u, shifts=1, dims=0)
    ) / (dx * dx)
    u_yy = (
        torch.roll(u, shifts=-1, dims=1) - 2.0 * u + torch.roll(u, shifts=1, dims=1)
    ) / (dy * dy)
    u_zz = (
        torch.roll(u, shifts=-1, dims=2) - 2.0 * u + torch.roll(u, shifts=1, dims=2)
    ) / (dz * dz)
    return u_xx + u_yy + u_zz


# ======================================================
# 5. Hpsi computation
#    Hpsi = -0.5 * Laplacian(psi) + V * psi
# ======================================================
def compute_Hpsi_fd(psi, V, dx, dy, dz):
    lap_psi = laplacian_3d(psi, dx, dy, dz)
    Hpsi = -0.5 * lap_psi + V * psi
    return Hpsi


# ======================================================
# 6. explicit Euler stepping
#    i psi_t = Hpsi  => psi_t = -i Hpsi
#    psi^{n+1} = psi^n + dt * (-i Hpsi^n)
# ======================================================
def time_step_euler(psi, V, dx, dy, dz, dt):
    Hpsi = compute_Hpsi_fd(psi, V, dx, dy, dz)
    dpsi_dt = -1j * Hpsi
    psi_next = psi + dt * dpsi_dt
    return psi_next, Hpsi


# ======================================================
# 7. data generation
#    single case; keep the full time series
#    u: [1, 3, Nx, Ny, Nz, Nt]
#    f: [1, 2, Nx, Ny, Nz, Nt]
# ======================================================
def generate_dataset(Nx=16, Ny=16, Nz=16, Nt=50, dt=1e-4, device="cpu"):
    x = torch.linspace(-Lx / 2, Lx / 2, Nx, device=device)
    y = torch.linspace(-Ly / 2, Ly / 2, Ny, device=device)
    z = torch.linspace(-Lz / 2, Lz / 2, Nz, device=device)
    t = torch.arange(Nt, device=device, dtype=torch.float32) * dt

    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    dx = (x[1] - x[0]).item()
    dy = (y[1] - y[0]).item()
    dz = (z[1] - z[0]).item()

    V = potential(X, Y, Z)
    psi = initial_wavepacket(X, Y, Z)

    u_list = []
    f_list = []

    print("Generating 3D time-dependent Schrodinger dataset with explicit Euler ...")
    print(f"dx = {dx:.6e}, dy = {dy:.6e}, dz = {dz:.6e}, dt = {dt:.6e}")

    for n in range(Nt):
        Hpsi = compute_Hpsi_fd(psi, V, dx, dy, dz)

        # input channels: [Re(psi), Im(psi), V]
        u_n = torch.stack([torch.real(psi), torch.imag(psi), V], dim=0)

        # output channels: [Re(Hpsi), Im(Hpsi)]
        f_n = torch.stack([torch.real(Hpsi), torch.imag(Hpsi)], dim=0)

        u_list.append(u_n)
        f_list.append(f_n)

        if (n + 1) % max(1, Nt // 10) == 0 or n == 0:
            amp = torch.abs(psi)
            mass = (amp**2).mean().item()
            print(
                f"step {n+1:4d}/{Nt} | "
                f"|psi| mean={amp.mean().item():.4e}, "
                f"|psi| max={amp.max().item():.4e}, "
                f"mass(mean |psi|^2)={mass:.4e}"
            )

        psi, _ = time_step_euler(psi, V, dx, dy, dz, dt)

    # [Nt, C, Nx, Ny, Nz] -> [C, Nx, Ny, Nz, Nt]
    u = torch.stack(u_list, dim=0).permute(1, 2, 3, 4, 0).unsqueeze(0).contiguous()
    f = torch.stack(f_list, dim=0).permute(1, 2, 3, 4, 0).unsqueeze(0).contiguous()

    return (
        u.detach().cpu(),
        f.detach().cpu(),
        x.detach().cpu(),
        y.detach().cpu(),
        z.detach().cpu(),
        t.detach().cpu(),
        dt,
    )


# ======================================================
# 8. save
#    train: first 20% of the time window
#    test : full time series
# ======================================================
def save_temporal_dataset(u, f, x, y, z, t, dt, base_dir="dataset", train_ratio=0.2):
    """
    u: [1, 3, Nx, Ny, Nz, Nt]
    f: [1, 2, Nx, Ny, Nz, Nt]
    """
    os.makedirs(os.path.join(base_dir, "train_data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test_data"), exist_ok=True)

    Nt = u.shape[-1]
    Nt_train = max(1, int(train_ratio * Nt))

    metadata = dict(
        equation="3D_time_dependent_schrodinger",
        pde="i * dpsi/dt = -0.5 * Laplacian(psi) + V * psi",
        operator="Hpsi = -0.5 * Laplacian(psi) + V * psi",
        input_channels="u = [Re(psi), Im(psi), V]",
        output_channels="f = [Re(Hpsi), Im(Hpsi)]",
        rollout="explicit_euler",
        dt=float(dt),
        bc="periodic",
        domain=f"[-{Lx/2},{Lx/2}] x [-{Ly/2},{Ly/2}] x [-{Lz/2},{Lz/2}]",
        single_case=True,
        train_split="first_20_percent_time_steps",
        test_contains_full_trajectory=True,
    )

    train_data = dict(
        u=u[..., :Nt_train],
        f=f[..., :Nt_train],
        x=x,
        y=y,
        z=z,
        t=t[:Nt_train],
        metadata=metadata,
    )

    test_data = dict(u=u, f=f, x=x, y=y, z=z, t=t, metadata=metadata)

    torch.save(train_data, os.path.join(base_dir, "train_data", "train.pt"))
    torch.save(test_data, os.path.join(base_dir, "test_data", "test.pt"))

    print("\nDataset saved!")
    print("Train u shape:", train_data["u"].shape)
    print("Train f shape:", train_data["f"].shape)
    print("Test  u shape:", test_data["u"].shape)
    print("Test  f shape:", test_data["f"].shape)
    print("Train time steps:", train_data["u"].shape[-1])
    print("Test  time steps:", test_data["u"].shape[-1])


# ======================================================
# 9. preview figures
#    plot the z-mid slice
# ======================================================
def save_preview(u, f, x, y, z, t, save_dir="dataset/preview", n_show=4):
    os.makedirs(save_dir, exist_ok=True)

    Nt = u.shape[-1]
    z_mid = len(z) // 2
    extent = [x.min().item(), x.max().item(), y.min().item(), y.max().item()]

    if Nt <= n_show:
        time_ids = list(range(Nt))
    else:
        time_ids = np.linspace(0, Nt - 1, n_show, dtype=int).tolist()

    for tid in time_ids:
        re_psi = u[0, 0, :, :, z_mid, tid]
        im_psi = u[0, 1, :, :, z_mid, tid]
        V_mid = u[0, 2, :, :, z_mid, tid]
        re_H = f[0, 0, :, :, z_mid, tid]
        im_H = f[0, 1, :, :, z_mid, tid]

        items = [
            (re_psi, f"Re(psi), t={t[tid].item():.4e}", f"re_psi_t{tid:04d}.png"),
            (im_psi, f"Im(psi), t={t[tid].item():.4e}", f"im_psi_t{tid:04d}.png"),
            (V_mid, f"V, t={t[tid].item():.4e}", f"V_t{tid:04d}.png"),
            (re_H, f"Re(Hpsi), t={t[tid].item():.4e}", f"re_Hpsi_t{tid:04d}.png"),
            (im_H, f"Im(Hpsi), t={t[tid].item():.4e}", f"im_Hpsi_t{tid:04d}.png"),
        ]

        for img, title, fname in items:
            plt.figure(figsize=(6, 5))
            plt.imshow(
                img.T, origin="lower", extent=extent, aspect="equal", cmap="Blues"
            )
            plt.colorbar()
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname), dpi=400)
            plt.close()


# ======================================================
# 10. main program
# ======================================================
if __name__ == "__main__":
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    u, f, x, y, z, t, dt = generate_dataset(
        Nx=16, Ny=16, Nz=16, Nt=5000, dt=1e-4, device=device
    )

    print("\nFinal tensor shapes:")
    print("u shape:", u.shape)  # [1, 3, Nx, Ny, Nz, Nt]
    print("f shape:", f.shape)  # [1, 2, Nx, Ny, Nz, Nt]

    save_temporal_dataset(
        u=u, f=f, x=x, y=y, z=z, t=t, dt=dt, base_dir="dataset", train_ratio=0.2
    )

    save_preview(u=u, f=f, x=x, y=y, z=z, t=t, save_dir="dataset/preview", n_show=11)

    print("\nDone.")
