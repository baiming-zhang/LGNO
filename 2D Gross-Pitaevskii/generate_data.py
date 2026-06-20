# ============================================================
# Static GPE-like Dataset Generator
# ABSOLUTELY PERIODIC VERSION
#
# Train: optical lattice
# Test : hex / deformed / bichromatic / disorder
#
# Input  u = [Re(psi), Im(psi), V]
# Output f = [Re(Hpsi), Im(Hpsi)]
#
# Hpsi = -(hbar^2 / 2m) Δ psi + V psi + g |psi|^2 psi
#
# Periodicity:
#   - psi is strictly periodic
#   - V is strictly periodic
#   - finite-difference Laplacian with torch.roll is periodic
# ============================================================

import os
import math
import random
import torch
import numpy as np
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
# 1. grid + spectral operators
# ======================================================
def build_grid(Nx=128, Ny=128, Lx=1.0, Ly=1.0, device="cpu"):
    x = torch.arange(Nx, device=device) * (Lx / Nx)
    y = torch.arange(Ny, device=device) * (Ly / Ny)

    X, Y = torch.meshgrid(x, y, indexing="ij")

    dx = Lx / Nx
    dy = Ly / Ny

    kx = 2.0 * math.pi * torch.fft.fftfreq(Nx, d=dx).to(device)
    ky = 2.0 * math.pi * torch.fft.fftfreq(Ny, d=dy).to(device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K2 = KX ** 2 + KY ** 2

    return x, y, X, Y, dx, dy, K2, Lx, Ly


# ======================================================
# 2. utilities
# ======================================================
def normalize(psi, dx, dy):
    norm = torch.sqrt(torch.sum(torch.abs(psi) ** 2) * dx * dy)
    return psi / norm.clamp_min(1e-12)


def complex_to_2ch(field):
    return torch.stack([field.real, field.imag], dim=0).float()


def amplitude_from_2ch(field_2ch):
    return torch.sqrt(field_2ch[0] ** 2 + field_2ch[1] ** 2)


def pack_input(psi, V):
    return torch.stack([psi.real, psi.imag, V], dim=0).float()


def periodic_mode(X, Y, mx, my, Lx, Ly, phase=0.0, use_cos=True):
    arg = 2.0 * math.pi * (mx * X / Lx + my * Y / Ly) + phase
    return torch.cos(arg) if use_cos else torch.sin(arg)


# ======================================================
# 3. strictly periodic psi
# ======================================================
def lowfreq_periodic_psi(
    X,
    Y,
    Lx,
    Ly,
    device,
    sample_seed,
    max_mode_x=3,
    max_mode_y=3,
    n_modes_real=8,
    n_modes_imag=8,
    add_constant=True,
    add_small_noise=False
):
    """
    Strictly periodic complex field:
        psi(x,y) = psi_real + i psi_imag
    using only integer Fourier modes.
    """
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)

    Nx, Ny = X.shape

    psi_real = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)
    psi_imag = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)

    if add_constant:
        c_re = 0.3 + 0.7 * torch.rand(1, device=device).item()
        # c_im = -0.2 + 0.4 * torch.rand(1, device=device).item()
        sign_im = 1.0 if torch.rand(1, device=device).item() > 0.5 else -1.0
        sign_im = -1.0
        c_im = sign_im * (0.3 + 0.7 * torch.rand(1, device=device).item())
        psi_real = psi_real + c_re
        psi_imag = psi_imag + c_im

    for _ in range(n_modes_real):
        mx = int(torch.randint(-max_mode_x, max_mode_x + 1, (1,), device=device).item())
        my = int(torch.randint(-max_mode_y, max_mode_y + 1, (1,), device=device).item())
        if mx == 0 and my == 0:
            continue

        amp = 0.45 / ((abs(mx) + 1) ** 1.5 + (abs(my) + 1) ** 1.5)
        amp *= (0.7 + 0.6 * torch.rand(1, device=device).item())
        phase = 2.0 * math.pi * torch.rand(1, device=device).item()
        use_cos = torch.rand(1, device=device).item() > 0.5

        psi_real = psi_real + amp * periodic_mode(X, Y, mx, my, Lx, Ly, phase, use_cos)

    for _ in range(n_modes_imag):
        mx = int(torch.randint(-max_mode_x, max_mode_x + 1, (1,), device=device).item())
        my = int(torch.randint(-max_mode_y, max_mode_y + 1, (1,), device=device).item())
        if mx == 0 and my == 0:
            continue

        amp = 0.45 / ((abs(mx) + 1) ** 1.5 + (abs(my) + 1) ** 1.5)
        amp *= (0.7 + 0.6 * torch.rand(1, device=device).item())
        phase = 2.0 * math.pi * torch.rand(1, device=device).item()
        use_cos = torch.rand(1, device=device).item() > 0.5

        psi_imag = psi_imag + amp * periodic_mode(X, Y, mx, my, Lx, Ly, phase, use_cos)

    if add_small_noise:
        # also periodic: build noise from random Fourier coefficients
        noise = torch.randn((Nx, Ny), device=device)
        noise_hat = torch.fft.fft2(noise)
        kx = torch.fft.fftfreq(Nx, d=Lx / Nx).to(device)
        ky = torch.fft.fftfreq(Ny, d=Ly / Ny).to(device)
        KXn, KYn = torch.meshgrid(kx, ky, indexing="ij")
        filt = torch.exp(-10.0 * (KXn ** 2 + KYn ** 2))
        noise_p = torch.fft.ifft2(noise_hat * filt).real
        noise_p = noise_p / noise_p.std().clamp_min(1e-12)

        psi_real = psi_real + 0.01 * noise_p
        psi_imag = psi_imag + 0.01 * torch.roll(noise_p, shifts=7, dims=1)

    psi = (psi_real + 1j * psi_imag).to(torch.complex64)
    return psi


# ======================================================
# 4. strictly periodic potentials
# ======================================================
def build_optical_lattice(X, Y, Lx, Ly, device):
    mx = int(torch.randint(2, 5, (1,), device=device).item())
    my = int(torch.randint(2, 5, (1,), device=device).item())

    Vx = 1.0 + 0.5 * torch.rand(1, device=device).item()
    Vy = 0.7 + 0.5 * torch.rand(1, device=device).item()

    phix = 2.0 * math.pi * torch.rand(1, device=device).item()
    phiy = 2.0 * math.pi * torch.rand(1, device=device).item()

    V = (
        Vx * torch.cos(2.0 * math.pi * mx * X / Lx + phix) ** 2
        + Vy * torch.cos(2.0 * math.pi * my * Y / Ly + phiy) ** 2
    )
    return V


def build_hex_lattice(X, Y, Lx, Ly, device):
    """
    Strictly periodic 'hex-like' lattice on rectangular torus.
    Use three periodic plane waves with integer wavevectors.
    """
    m = int(torch.randint(2, 5, (1,), device=device).item())
    n = int(torch.randint(2, 5, (1,), device=device).item())

    phi1 = 2.0 * math.pi * torch.rand(1, device=device).item()
    phi2 = 2.0 * math.pi * torch.rand(1, device=device).item()
    phi3 = 2.0 * math.pi * torch.rand(1, device=device).item()

    V1 = 0.9 + 0.4 * torch.rand(1, device=device).item()
    V2 = 0.15 + 0.15 * torch.rand(1, device=device).item()

    p1 = 2.0 * math.pi * ( m * X / Lx +  n * Y / Ly)
    p2 = 2.0 * math.pi * (-m * X / Lx +  n * Y / Ly)
    p3 = 2.0 * math.pi * (      0.0   * X + 2 * n * Y / Ly)

    V = (
        V1 * (torch.cos(p1 + phi1) + torch.cos(p2 + phi2) + torch.cos(p3 + phi3))
        + V2 * (torch.cos(2.0 * p1) + torch.cos(2.0 * p2) + torch.cos(2.0 * p3))
    )
    return V


def build_deformed_lattice(X, Y, Lx, Ly, device):
    mx = int(torch.randint(2, 5, (1,), device=device).item())
    my = int(torch.randint(2, 5, (1,), device=device).item())
    md = int(torch.randint(1, 4, (1,), device=device).item())

    V1 = 1.0 + 0.4 * torch.rand(1, device=device).item()
    V2 = 0.6 + 0.4 * torch.rand(1, device=device).item()
    V3 = 0.3 + 0.3 * torch.rand(1, device=device).item()

    phi1 = 2.0 * math.pi * torch.rand(1, device=device).item()
    phi2 = 2.0 * math.pi * torch.rand(1, device=device).item()
    phi3 = 2.0 * math.pi * torch.rand(1, device=device).item()

    V = (
        V1 * torch.cos(2.0 * math.pi * mx * X / Lx + phi1)
        + V2 * torch.cos(2.0 * math.pi * my * Y / Ly + phi2)
        + V3 * torch.cos(2.0 * math.pi * md * (X / Lx + Y / Ly) + phi3)
    )
    return V


def build_bichromatic_lattice(X, Y, Lx, Ly, device):
    """
    Strictly periodic bichromatic lattice:
    use two different integer harmonics instead of irrational alpha.
    """
    mx1 = int(torch.randint(1, 4, (1,), device=device).item())
    mx2 = int(torch.randint(4, 8, (1,), device=device).item())
    my1 = int(torch.randint(1, 4, (1,), device=device).item())
    my2 = int(torch.randint(4, 8, (1,), device=device).item())

    lamx = 0.4 + 0.4 * torch.rand(1, device=device).item()
    lamy = 0.4 + 0.4 * torch.rand(1, device=device).item()

    phi1 = 2.0 * math.pi * torch.rand(1, device=device).item()
    phi2 = 2.0 * math.pi * torch.rand(1, device=device).item()
    phi3 = 2.0 * math.pi * torch.rand(1, device=device).item()
    phi4 = 2.0 * math.pi * torch.rand(1, device=device).item()

    V = (
        torch.cos(2.0 * math.pi * mx1 * X / Lx + phi1)
        + lamx * torch.cos(2.0 * math.pi * mx2 * X / Lx + phi2)
        + 0.8 * torch.cos(2.0 * math.pi * my1 * Y / Ly + phi3)
        + lamy * torch.cos(2.0 * math.pi * my2 * Y / Ly + phi4)
    )
    return V


def build_disorder_potential(X, Y, K2, device):
    """
    Strictly periodic random field on the discrete torus.
    """
    noise = torch.randn_like(X)

    alpha = 0.002 + 0.006 * torch.rand(1, device=device).item()
    noise_hat = torch.fft.fft2(noise)
    noise_hat = noise_hat * torch.exp(-alpha * K2)

    V = torch.fft.ifft2(noise_hat).real
    V = (V - V.mean()) / V.std().clamp_min(1e-12)

    amp = 0.8 + 0.5 * torch.rand(1, device=device).item()
    V = amp * V
    return V


def sample_potential(potential_type, X, Y, K2, Lx, Ly, device):
    if potential_type == "optical":
        return build_optical_lattice(X, Y, Lx, Ly, device)
    elif potential_type == "hex":
        return build_hex_lattice(X, Y, Lx, Ly, device)
    elif potential_type == "deformed":
        return build_deformed_lattice(X, Y, Lx, Ly, device)
    elif potential_type == "bichromatic":
        return build_bichromatic_lattice(X, Y, Lx, Ly, device)
    elif potential_type == "disorder":
        return build_disorder_potential(X, Y, K2, device)
    else:
        raise ValueError(f"Unknown potential_type: {potential_type}")


# ======================================================
# 5. Hamiltonian
# ======================================================
def H_operator(psi, V, g, dx, dy, hbar2_over_2m=1.0):
    """
    Discrete periodic finite-difference version of

        Hpsi = -(hbar^2/2m) Δ psi + V psi + g |psi|^2 psi

    Parameters
    ----------
    psi : complex tensor, shape [Nx, Ny]
    V   : real tensor,    shape [Nx, Ny]
    g   : float
    dx  : float
    dy  : float
    hbar2_over_2m : float

    Returns
    -------
    Hpsi : complex tensor, shape [Nx, Ny]
    """

    psi_xx = (
        torch.roll(psi, shifts=-1, dims=0)
        - 2.0 * psi
        + torch.roll(psi, shifts=1, dims=0)
    ) / (dx * dx)

    psi_yy = (
        torch.roll(psi, shifts=-1, dims=1)
        - 2.0 * psi
        + torch.roll(psi, shifts=1, dims=1)
    ) / (dy * dy)

    lap_psi = psi_xx + psi_yy

    kinetic = -hbar2_over_2m * lap_psi
    potential = V * psi
    nonlinear = g * torch.abs(psi) ** 2 * psi

    return kinetic + potential + nonlinear


# ======================================================
# 6. generate one split
# ======================================================
def generate_split(
    num_samples,
    potential_type,
    Nx=128,
    Ny=128,
    Lx=1.0,
    Ly=1.0,
    g=5.0,
    hbar2_over_2m=1.0,
    device="cpu",
    sample_seed_offset=0,
    psi_max_mode_x=3,
    psi_max_mode_y=3,
    psi_n_modes_real=8,
    psi_n_modes_imag=8
):
    x, y, X, Y, dx, dy, K2, Lx, Ly = build_grid(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device
    )

    u_all = []
    f_all = []
    psi_all = []
    V_all = []
    labels = []

    for s in range(num_samples):
        print(f"  sample {s + 1}/{num_samples} | potential = {potential_type}")

        V = sample_potential(potential_type, X, Y, K2, Lx, Ly, device)

        psi = lowfreq_periodic_psi(
            X=X,
            Y=Y,
            Lx=Lx,
            Ly=Ly,
            device=device,
            sample_seed=sample_seed_offset + s,
            max_mode_x=psi_max_mode_x,
            max_mode_y=psi_max_mode_y,
            n_modes_real=psi_n_modes_real,
            n_modes_imag=psi_n_modes_imag,
            add_constant=True,
            add_small_noise=False
        )
        psi = normalize(psi, dx, dy)

        Hpsi = H_operator(
            psi,
            V,
            g,
            dx,
            dy,
            hbar2_over_2m=hbar2_over_2m
        )

        u = pack_input(psi, V)
        f = complex_to_2ch(Hpsi)

        u_all.append(u.detach().cpu())
        f_all.append(f.detach().cpu())
        psi_all.append(complex_to_2ch(psi).detach().cpu())
        V_all.append(V.detach().cpu())
        labels.append(potential_type)

        psi_abs = torch.abs(psi)
        Hpsi_abs = torch.abs(Hpsi)

        print(
            f"    |psi| : mean={psi_abs.mean().item(): .4e}, std={psi_abs.std().item(): .4e}, "
            f"max={psi_abs.max().item(): .4e}, min={psi_abs.min().item(): .4e}"
        )
        print(
            f"    |Hpsi|: mean={Hpsi_abs.mean().item(): .4e}, std={Hpsi_abs.std().item(): .4e}, "
            f"max={Hpsi_abs.max().item(): .4e}, min={Hpsi_abs.min().item(): .4e}"
        )

    u_all = torch.stack(u_all, dim=0)
    f_all = torch.stack(f_all, dim=0)
    psi_all = torch.stack(psi_all, dim=0)
    V_all = torch.stack(V_all, dim=0)

    return {
        "u": u_all,
        "f": f_all,
        "psi": psi_all,
        "V": V_all,
        "x": x.detach().cpu(),
        "y": y.detach().cpu(),
        "labels": labels,
    }


# ======================================================
# 7. preview
# ======================================================
def save_preview(dataset_dict, save_dir="preview", max_show=None):
    os.makedirs(save_dir, exist_ok=True)

    u_all = dataset_dict["u"]
    f_all = dataset_dict["f"]
    V_all = dataset_dict["V"]
    labels = dataset_dict["labels"]
    x = dataset_dict["x"]
    y = dataset_dict["y"]

    extent = [x.min().item(), x.max().item(), y.min().item(), y.max().item()]

    if max_show is None:
        n_show = u_all.shape[0]
    else:
        n_show = min(max_show, u_all.shape[0])

    print(f"Saving preview to: {save_dir}")
    print(f"Number of preview samples: {n_show}")

    for i in range(n_show):
        print(f"  saving preview sample {i + 1}/{n_show}")

        u = u_all[i]
        f = f_all[i]
        V = V_all[i]
        label = labels[i]

        psi_amp = torch.sqrt(u[0] ** 2 + u[1] ** 2)
        psi_phase = torch.atan2(u[1], u[0])
        Hpsi_amp = amplitude_from_2ch(f)

        plt.figure(figsize=(6, 5))
        plt.imshow(V.T, origin="lower", extent=extent, aspect="equal", cmap="viridis")
        plt.colorbar()
        plt.title(f"V sample {i} ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"V_sample_{i}_{label}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(psi_amp.T, origin="lower", extent=extent, aspect="equal", cmap="viridis")
        plt.colorbar()
        plt.title(f"|psi| sample {i} ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"psi_amp_sample_{i}_{label}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(
            psi_phase.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="twilight",
            vmin=-math.pi,
            vmax=math.pi
        )
        plt.colorbar()
        plt.title(f"phase(psi) sample {i} ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"psi_phase_sample_{i}_{label}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(Hpsi_amp.T, origin="lower", extent=extent, aspect="equal", cmap="Blues")
        plt.colorbar()
        plt.title(f"|Hpsi| sample {i} ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Hpsi_amp_sample_{i}_{label}.png"), dpi=300)
        plt.close()

        vmax = max(torch.max(torch.abs(u[0])).item(), 1e-12)
        plt.figure(figsize=(6, 5))
        plt.imshow(
            u[0].T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax
        )
        plt.colorbar()
        plt.title(f"Re(psi) sample {i} ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"psi_real_sample_{i}_{label}.png"), dpi=300)
        plt.close()

        vmax = max(torch.max(torch.abs(u[1])).item(), 1e-12)
        plt.figure(figsize=(6, 5))
        plt.imshow(
            u[1].T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax
        )
        plt.colorbar()
        plt.title(f"Im(psi) sample {i} ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"psi_imag_sample_{i}_{label}.png"), dpi=300)
        plt.close()


# ======================================================
# 8. save datasets
# ======================================================
def save_dataset(dataset_dict, save_path, metadata):
    save_obj = {
        "u": dataset_dict["u"],
        "f": dataset_dict["f"],
        "psi": dataset_dict["psi"],
        "V": dataset_dict["V"],
        "x": dataset_dict["x"],
        "y": dataset_dict["y"],
        "labels": dataset_dict["labels"],
        "metadata": metadata,
    }
    torch.save(save_obj, save_path)


# ======================================================
# 9. main
# ======================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Nx = 64
    Ny = 64
    Lx = 1.0
    Ly = 1.0

    g = 0.05
    hbar2_over_2m = 1.0

    psi_max_mode_x = 3
    psi_max_mode_y = 3
    psi_n_modes_real = 8
    psi_n_modes_imag = 8

    n_train_optical = 1
    n_test_each = 1

    base_dir = "dataset"
    train_dir = os.path.join(base_dir, "train_data")
    test_dir = os.path.join(base_dir, "test_data")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("\nGenerating TRAIN set (optical lattice only)")
    train_data = generate_split(
        num_samples=n_train_optical,
        potential_type="optical",
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        g=g,
        hbar2_over_2m=hbar2_over_2m,
        device=device,
        sample_seed_offset=1000,
        psi_max_mode_x=psi_max_mode_x,
        psi_max_mode_y=psi_max_mode_y,
        psi_n_modes_real=psi_n_modes_real,
        psi_n_modes_imag=psi_n_modes_imag
    )

    print("\nGenerating TEST set: hex")
    test_hex = generate_split(
        num_samples=n_test_each,
        potential_type="hex",
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        g=g,
        hbar2_over_2m=hbar2_over_2m,
        device=device,
        sample_seed_offset=2000,
        psi_max_mode_x=psi_max_mode_x,
        psi_max_mode_y=psi_max_mode_y,
        psi_n_modes_real=psi_n_modes_real,
        psi_n_modes_imag=psi_n_modes_imag
    )

    print("\nGenerating TEST set: deformed")
    test_deformed = generate_split(
        num_samples=n_test_each,
        potential_type="deformed",
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        g=g,
        hbar2_over_2m=hbar2_over_2m,
        device=device,
        sample_seed_offset=3000,
        psi_max_mode_x=psi_max_mode_x,
        psi_max_mode_y=psi_max_mode_y,
        psi_n_modes_real=psi_n_modes_real,
        psi_n_modes_imag=psi_n_modes_imag
    )

    print("\nGenerating TEST set: bichromatic")
    test_bichromatic = generate_split(
        num_samples=n_test_each,
        potential_type="bichromatic",
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        g=g,
        hbar2_over_2m=hbar2_over_2m,
        device=device,
        sample_seed_offset=4000,
        psi_max_mode_x=psi_max_mode_x,
        psi_max_mode_y=psi_max_mode_y,
        psi_n_modes_real=psi_n_modes_real,
        psi_n_modes_imag=psi_n_modes_imag
    )

    print("\nGenerating TEST set: disorder")
    test_disorder = generate_split(
        num_samples=n_test_each,
        potential_type="disorder",
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        g=g,
        hbar2_over_2m=hbar2_over_2m,
        device=device,
        sample_seed_offset=5000,
        psi_max_mode_x=psi_max_mode_x,
        psi_max_mode_y=psi_max_mode_y,
        psi_n_modes_real=psi_n_modes_real,
        psi_n_modes_imag=psi_n_modes_imag
    )

    test_data = {
        "u": torch.cat([test_hex["u"], test_deformed["u"], test_bichromatic["u"], test_disorder["u"]], dim=0),
        "f": torch.cat([test_hex["f"], test_deformed["f"], test_bichromatic["f"], test_disorder["f"]], dim=0),
        "psi": torch.cat([test_hex["psi"], test_deformed["psi"], test_bichromatic["psi"], test_disorder["psi"]], dim=0),
        "V": torch.cat([test_hex["V"], test_deformed["V"], test_bichromatic["V"], test_disorder["V"]], dim=0),
        "x": test_hex["x"],
        "y": test_hex["y"],
        "labels": test_hex["labels"] + test_deformed["labels"] + test_bichromatic["labels"] + test_disorder["labels"],
    }

    metadata = {
        "Nx": Nx,
        "Ny": Ny,
        "Lx": Lx,
        "Ly": Ly,
        "g": g,
        "hbar2_over_2m": hbar2_over_2m,
        "psi_max_mode_x": psi_max_mode_x,
        "psi_max_mode_y": psi_max_mode_y,
        "psi_n_modes_real": psi_n_modes_real,
        "psi_n_modes_imag": psi_n_modes_imag,
        "train_samples": n_train_optical,
        "test_samples": n_test_each * 4,
        "train_potential": "optical",
        "test_potentials": ["hex", "deformed", "bichromatic", "disorder"],
        "psi_type": "strictly periodic low-frequency complex field",
        "solver": "direct Hpsi evaluation on periodic domain",
        "boundary_condition": "strict periodic"
    }

    train_path = os.path.join(train_dir, "train_dataset.pt")
    test_path = os.path.join(test_dir, "test_dataset.pt")

    save_dataset(train_data, train_path, metadata)
    save_dataset(test_data, test_path, metadata)

    print("\nDataset saved!")
    print(f"  Train: {train_data['u'].shape[0]} samples -> {train_path}")
    print(f"  Test:  {test_data['u'].shape[0]} samples -> {test_path}")

    save_preview(train_data, save_dir=os.path.join(train_dir, "preview"), max_show=None)
    save_preview(test_data, save_dir=os.path.join(test_dir, "preview"), max_show=None)

    print("\nDone!")


if __name__ == "__main__":
    main()