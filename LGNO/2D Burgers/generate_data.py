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
# 1. save helper: split along the sample dimension and do not use t
# ======================================================
def save_static_dataset(u, f, x, y, base_dir="dataset", train_ratio=0.2):
    """
    u: [Ns, 1, Nx, Ny]
    f: [Ns, 1, Nx, Ny]
    x: [Nx]
    y: [Ny]
    """
    train_dir = os.path.join(base_dir, "train_data")
    test_dir = os.path.join(base_dir, "test_data")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    num_samples = u.shape[0]
    n_train = max(1, int(train_ratio * num_samples))
    n_train = min(n_train, num_samples - 1) if num_samples > 1 else 1

    train_data = dict(
        u=u[:n_train],
        f=f[:n_train],
        x=x,
        y=y,
        metadata=dict(
            equation="static_scalar_burgers_like",
            operator="f = u * du/dx + u * du/dy - nu * lap(u)",
            bc="soft_zero_boundary",
            domain="[-1,1]^2",
            time_dependent=False,
            full_domain_saved=True,
        ),
    )

    test_data = dict(
        u=u[n_train:], f=f[n_train:], x=x, y=y, metadata=train_data["metadata"]
    )

    torch.save(train_data, os.path.join(train_dir, "train.pt"))
    torch.save(test_data, os.path.join(test_dir, "test.pt"))

    print("Dataset saved.")
    print("Train u shape:", train_data["u"].shape)
    print("Train f shape:", train_data["f"].shape)
    print("Test  u shape:", test_data["u"].shape)
    print("Test  f shape:", test_data["f"].shape)


# ======================================================
# 2. full-domain finite differences (torch.roll)
#    so u and f keep the same full-domain size
# ======================================================
def diff_x(u, dx):
    return (torch.roll(u, shifts=-1, dims=0) - torch.roll(u, shifts=1, dims=0)) / (
        2.0 * dx
    )


def diff_y(u, dy):
    return (torch.roll(u, shifts=-1, dims=1) - torch.roll(u, shifts=1, dims=1)) / (
        2.0 * dy
    )


def laplacian(u, dx, dy):
    u_xx = (
        torch.roll(u, shifts=-1, dims=0) - 2.0 * u + torch.roll(u, shifts=1, dims=0)
    ) / (dx * dx)
    u_yy = (
        torch.roll(u, shifts=-1, dims=1) - 2.0 * u + torch.roll(u, shifts=1, dims=1)
    ) / (dy * dy)
    return u_xx + u_yy


# ======================================================
# 3. base random function
# ======================================================
def rand_uniform(low, high, device):
    return low + (high - low) * torch.rand(1, device=device).item()


def rand_int(low, high, device):
    return int(torch.randint(low, high, (1,), device=device).item())


def gaussian_blob(X, Y, x0, y0, amp, sigma_x, sigma_y=None, theta=0.0):
    if sigma_y is None:
        sigma_y = sigma_x

    ct = math.cos(theta)
    st = math.sin(theta)

    Xc = X - x0
    Yc = Y - y0

    Xr = ct * Xc + st * Yc
    Yr = -st * Xc + ct * Yc

    return amp * torch.exp(-0.5 * ((Xr / sigma_x) ** 2 + (Yr / sigma_y) ** 2))


def generate_liquid_like_field(X, Y, device, sample_seed):
    """
    True periodic-boundary version:
    1. multiple periodic Gaussian vortices (implemented by summing periodic images)
    2. no boundary decay
    3. smoother for large sigma and better matched to periodic boundaries
    """

    import torch

    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)

    field = torch.zeros_like(X)

    # periodic domain length: [-1, 1] -> L = 2
    L = 2.0

    def periodic_gaussian(X, Y, x0, y0, amp, sigma):
        """
        True periodic Gaussian:
        sum over adjacent periodic images
        for sigma <= 0.5, m,n in {-1,0,1} is sufficient
        """
        g = torch.zeros_like(X)

        for mx in (-1, 0, 1):
            for my in (-1, 0, 1):
                dx = X - (x0 + mx * L)
                dy = Y - (y0 + my * L)
                r2 = dx**2 + dy**2
                g += amp * torch.exp(-r2 / (2.0 * sigma**2))

        return g

    # ======================================================
    # 1. main vortex
    # ======================================================
    n_main = int(torch.randint(4, 8, (1,), device=device).item())

    for _ in range(n_main):
        x0 = -1.0 + 2.0 * torch.rand(1, device=device).item()
        y0 = -1.0 + 2.0 * torch.rand(1, device=device).item()

        amp = 3.0 + 0.8 * torch.rand(1, device=device).item()
        if torch.rand(1, device=device).item() < 0.5:
            amp = -amp

        sigma = 0.3 + 0.2 * torch.rand(1, device=device).item()

        field += periodic_gaussian(X, Y, x0, y0, amp, sigma)

    field = field - torch.mean(field)

    return field


# ======================================================
# 5. save preview figures(for quick texture inspection)
# ======================================================
def save_preview(u_all, f_all, x, y, save_dir="dataset_preview", max_show=4):
    os.makedirs(save_dir, exist_ok=True)

    n_show = min(max_show, u_all.shape[0])
    extent = [x.min().item(), x.max().item(), y.min().item(), y.max().item()]

    for i in range(n_show):
        u = u_all[i, 0]
        f = f_all[i, 0]

        plt.figure(figsize=(6, 5))
        plt.imshow(u.T, origin="lower", extent=extent, aspect="equal", cmap="Blues")
        plt.colorbar()
        plt.title(f"u sample {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"u_sample_{i}.png"), dpi=400)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(f.T, origin="lower", extent=extent, aspect="equal", cmap="Blues")
        plt.colorbar()
        plt.title(f"f sample {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"f_sample_{i}.png"), dpi=400)
        plt.close()


# ======================================================
# 6. main program
# ======================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # parameters
    # -------------------------
    N = 512
    num_samples = 5
    nu = 0.01
    base_dir = "dataset"

    # -------------------------
    # grid
    # -------------------------
    L = 2.0
    dx = L / N

    x = torch.arange(N, device=device) * dx - 1.0
    y = torch.arange(N, device=device) * dx - 1.0
    X, Y = torch.meshgrid(x, y, indexing="ij")

    dx = (x[1] - x[0]).item()
    dy = (y[1] - y[0]).item()

    # -------------------------
    # generate samples
    # -------------------------
    u_all = []
    f_all = []

    for s in range(num_samples):
        print(f"Generating sample {s + 1}/{num_samples} ...")

        # static 2D field u(x, y)
        u = generate_liquid_like_field(X, Y, device=device, sample_seed=1234 + s)

        # full-domain derivatives
        ux = diff_x(u, dx)
        uy = diff_y(u, dy)
        lap_u = laplacian(u, dx, dy)

        # static target f(x, y)
        # keep the original formulation: f = u*ux + u*uy - nu*lap(u)
        f = u * ux + u * uy - nu * lap_u

        u_all.append(u.detach().cpu())
        f_all.append(f.detach().cpu())

        print(
            f"  u: mean={u.mean().item(): .4e}, std={u.std().item(): .4e}, "
            f"max={u.max().item(): .4e}, min={u.min().item(): .4e}"
        )
        print(
            f"  f: mean={f.mean().item(): .4e}, std={f.std().item(): .4e}, "
            f"max={f.max().item(): .4e}, min={f.min().item(): .4e}"
        )

    # [Ns, 1, N, N]
    u_all = torch.stack(u_all, dim=0).unsqueeze(1)
    f_all = torch.stack(f_all, dim=0).unsqueeze(1)

    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()

    print("\nFinal tensor shapes:")
    print("u_all =", tuple(u_all.shape))
    print("f_all =", tuple(f_all.shape))

    # -------------------------
    # save data
    # -------------------------
    save_static_dataset(
        u=u_all,
        f=f_all,
        x=x_cpu,
        y=y_cpu,
        base_dir=base_dir,
        train_ratio=0.2,
    )

    # -------------------------
    # save preview figures
    # -------------------------
    save_preview(
        u_all,
        f_all,
        x_cpu,
        y_cpu,
        save_dir=os.path.join(base_dir, "preview"),
        max_show=4,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
