import os
import math
import random
import torch
import matplotlib.pyplot as plt

# ======================================================
# 0. 随机种子
# ======================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ======================================================
# 1. 保存函数：按样本维切分，不再使用 t
# ======================================================
def make_metadata(noise_level=None, seed=None, split=None):
    metadata = dict(
        equation="static_scalar_burgers_like",
        operator="f = u * du/dx + u * du/dy - nu * lap(u)",
        bc="soft_zero_boundary",
        domain="[-1,1]^2",
        time_dependent=False,
        full_domain_saved=True,
    )
    if noise_level is not None:
        metadata["noise_level"] = float(noise_level)
    if seed is not None:
        metadata["seed"] = int(seed)
    if split is not None:
        metadata["split"] = split
    return metadata


def save_static_dataset(u, f, x, y, base_dir="dataset", train_ratio=0.2, noise_level=None):
    """
    u: [Ns, 1, Nx, Ny]
    f: [Ns, 1, Nx, Ny]
    x: [Nx]
    y: [Ny]
    """
    train_dir = os.path.join(base_dir, "train_data")
    test_dir  = os.path.join(base_dir, "test_data")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    num_samples = u.shape[0]
    n_train = max(1, int(train_ratio * num_samples))
    n_train = min(n_train, num_samples - 1) if num_samples > 1 else 1

    metadata = make_metadata(noise_level=noise_level)

    train_data = dict(
        u=u[:n_train],
        f=f[:n_train],
        x=x,
        y=y,
        metadata=metadata
    )

    test_data = dict(
        u=u[n_train:],
        f=f[n_train:],
        x=x,
        y=y,
        metadata=train_data["metadata"]
    )

    torch.save(train_data, os.path.join(train_dir, "train.pt"))
    torch.save(test_data,  os.path.join(test_dir,  "test.pt"))

    print("Dataset saved.")
    print("Train u shape:", train_data["u"].shape)
    print("Train f shape:", train_data["f"].shape)
    print("Test  u shape:", test_data["u"].shape)
    print("Test  f shape:", test_data["f"].shape)


def save_seed_test_dataset(u, f, x, y, base_dir, noise_level, seed):
    test_dir = os.path.join(base_dir, f"eval_seed_{seed:02d}", "test_data")
    os.makedirs(test_dir, exist_ok=True)

    test_data = dict(
        u=u.unsqueeze(0).unsqueeze(0).detach().cpu(),
        f=f.unsqueeze(0).unsqueeze(0).detach().cpu(),
        x=x.detach().cpu(),
        y=y.detach().cpu(),
        metadata=make_metadata(noise_level=noise_level, seed=seed, split="eval_seed"),
    )

    torch.save(test_data, os.path.join(test_dir, "test.pt"))
    print(
        f"Seed test saved: noise={noise_level:.2f}, seed={seed:02d}, "
        f"u={tuple(test_data['u'].shape)}, f={tuple(test_data['f'].shape)}"
    )

# ======================================================
# 2. 全域有限差分（torch.roll）
#    这样 u 和 f 都能保持全域同尺寸
# ======================================================
def diff_x(u, dx):
    return (torch.roll(u, shifts=-1, dims=0) - torch.roll(u, shifts=1, dims=0)) / (2.0 * dx)

def diff_y(u, dy):
    return (torch.roll(u, shifts=-1, dims=1) - torch.roll(u, shifts=1, dims=1)) / (2.0 * dy)

def laplacian(u, dx, dy):
    u_xx = (torch.roll(u, shifts=-1, dims=0) - 2.0 * u + torch.roll(u, shifts=1, dims=0)) / (dx * dx)
    u_yy = (torch.roll(u, shifts=-1, dims=1) - 2.0 * u + torch.roll(u, shifts=1, dims=1)) / (dy * dy)
    return u_xx + u_yy

# ======================================================
# 3. 基础随机函数
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

    Xr =  ct * Xc + st * Yc
    Yr = -st * Xc + ct * Yc

    return amp * torch.exp(
        -0.5 * ((Xr / sigma_x) ** 2 + (Yr / sigma_y) ** 2)
    )

def generate_liquid_like_field(X, Y, device, sample_seed):
    """
    真正周期边界版本：
    1. 多个周期高斯涡（通过周期镜像求和实现）
    2. 不做边界衰减
    3. 对大 sigma 更平滑、更吻合周期边界
    """

    import torch

    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)

    field = torch.zeros_like(X)

    # 周期域长度：[-1,1] -> L = 2
    L = 2.0

    def periodic_gaussian(X, Y, x0, y0, amp, sigma):
        """
        真正的周期高斯：
        对相邻周期镜像求和
        对 sigma <= 0.5 这种范围，取 m,n in {-1,0,1} 已经够用了
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
    # 1. 主涡
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
# 5. 保存预览图（方便检查纹理）
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
# 6. 主程序
# ======================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # 参数
    # -------------------------
    N = 512
    num_samples = 2
    nu = 0.01
    base_dir = "dataset"
    noise_levels = [round(0.01 * i, 2) for i in range(11)]

    # -------------------------
    # 网格
    # -------------------------
    L = 2.0
    dx = L / N

    x = torch.arange(N, device=device) * dx - 1.0
    y = torch.arange(N, device=device) * dx - 1.0
    X, Y = torch.meshgrid(x, y, indexing="ij")

    dx = (x[1] - x[0]).item()
    dy = (y[1] - y[0]).item()

    # -------------------------
    # 生成样本
    # -------------------------
    u_by_noise = {level: [] for level in noise_levels}
    f_by_noise = {level: [] for level in noise_levels}

    for s in range(num_samples):
        print(f"Generating sample {s + 1}/{num_samples} ...")

        # 静态二维场 u(x,y)
        u_clean = generate_liquid_like_field(X, Y, device=device, sample_seed=1234 + s)

        # 1. 生成 [-1,1] 噪声
        torch.manual_seed(9000 + s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(9000 + s)
            torch.cuda.manual_seed_all(9000 + s)
        noise = 2 * torch.rand_like(u_clean) - 1.0

        # 2. 空间平滑（关键！！！）
        paddings = 10
        noise = torch.nn.functional.avg_pool2d(
            noise.unsqueeze(0).unsqueeze(0),
            kernel_size=2*paddings+1,  # 推荐 7~15
            stride=1,
            padding=paddings
        ).squeeze()

        # 3. 乘性噪声
        for noise_level in noise_levels:
            # 3. Multiplicative noise with fixed noise field and varied amplitude.
            u = u_clean * (1.0 + noise_level * noise)


        # 全域导数
            ux = diff_x(u, dx)
            uy = diff_y(u, dy)
            lap_u = laplacian(u, dx, dy)

        # 静态目标 f(x,y)
        # 保持与你原先思路一致：f = u*ux + u*uy - nu*lap(u)
            f = u * ux + u * uy - nu * lap_u


            u_by_noise[noise_level].append(u.detach().cpu())
            f_by_noise[noise_level].append(f.detach().cpu())

            print(
                f"  noise={noise_level:.2f} | "
                f"u std={u.std().item(): .4e}, f std={f.std().item(): .4e}"
            )

    # [Ns, 1, N, N]
    # Save each noise level as an independent dataset.

    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()

    for noise_level in noise_levels:
        u_all = torch.stack(u_by_noise[noise_level], dim=0).unsqueeze(1)
        f_all = torch.stack(f_by_noise[noise_level], dim=0).unsqueeze(1)
        noise_label = f"noise_{noise_level:.2f}"
        noise_dir = os.path.join(base_dir, noise_label)

        print(f"\nSaving {noise_label}:")
        print("u_all =", tuple(u_all.shape))
        print("f_all =", tuple(f_all.shape))

    # -------------------------
    # 保存数据
    # -------------------------
        save_static_dataset(
            u=u_all,
            f=f_all,
            x=x_cpu,
            y=y_cpu,
            base_dir=noise_dir,
            train_ratio=0.5,
            noise_level=noise_level,
        )

    # -------------------------
    # 保存预览图
    # -------------------------
        save_preview(
            u_all,
            f_all,
            x_cpu,
            y_cpu,
            save_dir=os.path.join(noise_dir, "preview"),
            max_show=4,
        )

    # -------------------------
    # Extra evaluation sets:
    # for each noise level, save independent seeded test samples.
    # These do not overwrite the trained train/test split above.
    # -------------------------
    eval_seeds = list(range(447, 452))

    for eval_seed in eval_seeds:
        print(f"\nGenerating eval seed {eval_seed} ...")

        u_clean = generate_liquid_like_field(X, Y, device=device, sample_seed=eval_seed)

        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(eval_seed)
            torch.cuda.manual_seed_all(eval_seed)
        noise = 2 * torch.rand_like(u_clean) - 1.0

        paddings = 10
        noise = torch.nn.functional.avg_pool2d(
            noise.unsqueeze(0).unsqueeze(0),
            kernel_size=2*paddings+1,
            stride=1,
            padding=paddings
        ).squeeze()

        for noise_level in noise_levels:
            u = u_clean * (1.0 + noise_level * noise)
            ux = diff_x(u, dx)
            uy = diff_y(u, dy)
            lap_u = laplacian(u, dx, dy)
            f = u * ux + u * uy - nu * lap_u

            noise_label = f"noise_{noise_level:.2f}"
            noise_dir = os.path.join(base_dir, noise_label)
            save_seed_test_dataset(
                u=u,
                f=f,
                x=x,
                y=y,
                base_dir=noise_dir,
                noise_level=noise_level,
                seed=eval_seed,
            )

    print("\nDone.")

if __name__ == "__main__":
    main()
