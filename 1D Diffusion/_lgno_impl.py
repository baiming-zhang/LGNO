# ============================================================
# 1D Diffusion
# Local 3-point MLP Stencil (space only)
#
# Periodic BC version
#
# Changes:
# 1) Use periodic spatial boundaries.
# 2) Implement the full-domain periodic stencil with torch.roll.
# 3) Compute loss, rollout, and test error over the full domain.
# 4) Read x from data and compute dx and coefficient = kappa / dx^2.
# 5) Keep compatibility with the generated periodic data.
# 6) Save total loss to txt for later curve plotting.
# 7) Save all figures under save_dir/figure with subfolders.
# 8) Save the training curve as svg and pdf.
# 9) Save dudt, rollout, and slice figures separately.
# 10) Normalize true/prediction figures consistently to [-1, 1].
# 11) Use signed relative error: error_norm = (pred - true) / max|true|.
# 12) Remove all norm and nonlinear remapping.
# 13) Export square field figures without colorbars, text, ticks, or borders.
#
# Network input/output format remains unchanged:
#   Input : [B,1,Nx,Nt]
#   Output: [B,1,Nx,Nt]
# ============================================================

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# 0. Settings
# ============================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

save_dir = "LGNO_4hidden_results"
os.makedirs(save_dir, exist_ok=True)

figure_dir = os.path.join(save_dir, "figure")
training_figure_dir = os.path.join(figure_dir, "training")
dudt_figure_dir = os.path.join(figure_dir, "dudt")
rollout_figure_dir = os.path.join(figure_dir, "rollout")
slice_figure_dir = os.path.join(figure_dir, "slice")

os.makedirs(figure_dir, exist_ok=True)
os.makedirs(training_figure_dir, exist_ok=True)
os.makedirs(dudt_figure_dir, exist_ok=True)
os.makedirs(rollout_figure_dir, exist_ok=True)
os.makedirs(slice_figure_dir, exist_ok=True)


# ============================================================
# 1. Data loading
# ============================================================


def load_all_train(folder):
    u_list = []
    ut_list = []

    files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
    if len(files) == 0:
        raise FileNotFoundError(f"No .pt files found in folder: {folder}")

    dt_ref = None
    dx_ref = None
    kappa_ref = None
    bc_ref = None

    for file in files:
        path = os.path.join(folder, file)
        data = torch.load(path, map_location="cpu")

        if "u_train" not in data:
            raise KeyError(f"{path} does not contain 'u_train'")
        u_train = data["u_train"]

        if "ut_train" in data:
            ut_train = data["ut_train"]
        elif "target_operator_train" in data:
            ut_train = data["target_operator_train"]
        elif "f_train" in data:
            ut_train = data["f_train"]
            print(f"[Warning] {file}: fallback to old key 'f_train'")
        else:
            raise KeyError(
                f"{path} must contain one of ['ut_train', 'target_operator_train', 'f_train']"
            )

        if "t" not in data:
            raise KeyError(f"{path} does not contain 't'")
        t = data["t"]
        if len(t) < 2:
            raise ValueError(f"{path} has invalid time array length < 2")
        dt = float(t[1] - t[0])

        if "x" not in data:
            raise KeyError(f"{path} does not contain 'x'")
        x = data["x"]
        if len(x) < 2:
            raise ValueError(f"{path} has invalid x array length < 2")
        dx = float(x[1] - x[0])

        kappa = float(data.get("kappa", 0.1))
        bc = data.get("boundary_condition", "unknown")

        if dt_ref is None:
            dt_ref = dt
        if dx_ref is None:
            dx_ref = dx
        if kappa_ref is None:
            kappa_ref = kappa
        if bc_ref is None:
            bc_ref = bc

        u_list.append(u_train)
        ut_list.append(ut_train)

    min_t = min(u.shape[-1] for u in u_list)
    u_list = [u[..., :min_t] for u in u_list]
    ut_list = [ut[..., :min_t] for ut in ut_list]

    u_train = torch.cat(u_list, dim=0)
    ut_train = torch.cat(ut_list, dim=0)

    print("Final training u shape :", u_train.shape)
    print("Final training ut shape:", ut_train.shape)
    print("dt =", dt_ref)
    print("dx =", dx_ref)
    print("kappa =", kappa_ref)
    print("boundary_condition =", bc_ref)

    return (
        u_train.to(device),
        ut_train.to(device),
        dt_ref,
        dx_ref,
        kappa_ref,
        bc_ref,
    )


def load_test_data(path="test_data/test.pt"):
    data = torch.load(path, map_location="cpu")

    if "u" not in data:
        raise KeyError(f"{path} does not contain 'u'")
    u = data["u"]

    if "ut" in data:
        ut = data["ut"]
    elif "target_operator" in data:
        ut = data["target_operator"]
    elif "f" in data:
        ut = data["f"]
        print(f"[Warning] {path}: fallback to old key 'f'")
    else:
        ut = None

    if "t" not in data:
        raise KeyError(f"{path} does not contain 't'")
    t = data["t"]
    if len(t) < 2:
        raise ValueError(f"{path} has invalid time array length < 2")
    dt = float(t[1] - t[0])

    if "x" not in data:
        raise KeyError(f"{path} does not contain 'x'")
    x = data["x"]
    if len(x) < 2:
        raise ValueError(f"{path} has invalid x array length < 2")
    dx = float(x[1] - x[0])

    kappa = float(data.get("kappa", 0.1))
    bc = data.get("boundary_condition", "unknown")

    print("Test u shape :", u.shape)
    if ut is not None:
        print("Test ut shape:", ut.shape)
    print("Test dt =", dt)
    print("Test dx =", dx)
    print("Test kappa =", kappa)
    print("Test boundary_condition =", bc)

    return (
        u.to(device),
        (None if ut is None else ut.to(device)),
        t.to(device),
        dt,
        dx,
        kappa,
        bc,
    )


# ============================================================
# 2. Model
# ============================================================


class LocalCoeffStencilPeriodic(nn.Module):
    def __init__(self, dx, kappa=0.1, hidden=4, print_every=1000, learn_coeff=False):
        super().__init__()

        self.dx = float(dx)
        self.kappa = float(kappa)
        self.print_every = print_every
        self.learn_coeff = learn_coeff
        self._forward_count = 0

        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, u):
        """
        Input:
            u: [B,1,Nx,Nt]
        Output:
            dudt_pred: [B,1,Nx,Nt]
        """
        B, C, Nx, Nt = u.shape
        assert C == 1, "Only single-channel input is supported."

        u_left = torch.roll(u, shifts=1, dims=2)
        u_mid = u
        u_right = torch.roll(u, shifts=-1, dims=2)

        local_u = torch.cat([u_left, u_mid, u_right], dim=1)
        local_u = local_u.permute(0, 2, 3, 1).reshape(-1, 3)

        coeff = self.mlp(local_u).view(B, Nx, Nt, 3)

        c0 = coeff[..., 0]
        c1 = coeff[..., 1]
        c2 = coeff[..., 2]

        if self.training:
            self._forward_count += 1
            if self._forward_count % self.print_every == 0:
                with torch.no_grad():
                    print(
                        "[Stencil stats] "
                        f"c0={c0.mean().item():.3e}+/-{c0.std().item():.3e}, "
                        f"c1={c1.mean().item():.3e}+/-{c1.std().item():.3e}, "
                        f"c2={c2.mean().item():.3e}+/-{c2.std().item():.3e}"
                    )

        # Keep the original network definition logic.
        coefficient = 0.1 * 32 * 32
        dudt = coefficient * (
            c0.unsqueeze(1) * u_left
            + c1.unsqueeze(1) * u_mid
            + c2.unsqueeze(1) * u_right
        )

        return dudt


# ============================================================
# 3. Utility functions
# ============================================================


def relative_l2(pred, true, eps=1e-20):
    return torch.norm(pred - true) / (torch.norm(true) + eps)


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def get_curriculum_rollout_steps(
    epoch, total_epochs, max_rollout_steps, warmup_ratio=0.7
):
    if max_rollout_steps <= 1:
        return 1

    warmup_epochs = max(1, int(total_epochs * warmup_ratio))
    progress = min(1.0, epoch / warmup_epochs)
    steps = 1 + int(progress * (max_rollout_steps - 1))
    return max(1, min(max_rollout_steps, steps))


def rollout_from_state(model, u0, dt, steps):
    """
    u0: [B,1,Nx,1]
    Returns:
        u_roll: [B,1,Nx,steps+1]
    """
    states = [u0]
    current = u0

    for _ in range(steps):
        dudt_pred = model(current)
        current = current + dt * dudt_pred
        states.append(current)

    return torch.cat(states, dim=-1)


def compute_rollout_loss(
    model,
    u_train,
    dt,
    rollout_steps,
    num_rollout_starts,
    loss_fn,
):
    B, C, Nx, Nt = u_train.shape

    rollout_steps = min(rollout_steps, Nt - 1)
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    max_start = Nt - rollout_steps - 1
    if max_start < 0:
        max_start = 0

    all_starts = list(range(max_start + 1))
    if len(all_starts) == 0:
        all_starts = [0]

    if num_rollout_starts >= len(all_starts):
        starts = all_starts
    else:
        starts = random.sample(all_starts, num_rollout_starts)

    total_rollout_loss = 0.0

    for s in starts:
        u0 = u_train[:, :, :, s : s + 1]
        true_seg = u_train[:, :, :, s : s + rollout_steps + 1]
        pred_seg = rollout_from_state(model, u0, dt, rollout_steps)

        loss_roll = loss_fn(pred_seg[:, :, :, 1:], true_seg[:, :, :, 1:])
        total_rollout_loss = total_rollout_loss + loss_roll

    total_rollout_loss = total_rollout_loss / len(starts)
    return total_rollout_loss, starts


@torch.no_grad()
def predict_dudt_sequence(model, u_seq):
    model.eval()
    return model(u_seq)


@torch.no_grad()
def rollout_time_marching(model, u_true, dt):
    model.eval()

    B, C, Nx, Nt = u_true.shape
    u_roll = torch.zeros_like(u_true)
    u_roll[:, :, :, 0] = u_true[:, :, :, 0]

    for n in range(Nt - 1):
        u_now = u_roll[:, :, :, n : n + 1]
        dudt_pred = model(u_now)
        u_next = u_now + dt * dudt_pred
        u_roll[:, :, :, n + 1 : n + 2] = u_next

    return u_roll


# ============================================================
# 4. Training
# ============================================================


def train_model(
    model,
    u_train,
    ut_train,
    dt,
    lr=1e-3,
    epochs=20000,
    patience=500,
    lambda_deriv=1.0,
    lambda_roll=1.0,
    max_rollout_steps=10,
    num_rollout_starts=2,
    warmup_ratio=0.7,
    grad_clip=1.0,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=1,
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    loss_fn = nn.MSELoss()

    best_loss = 1e20
    wait = 0

    history = {
        "total": [],
        "deriv": [],
        "rollout": [],
        "lr": [],
        "rollout_steps": [],
    }

    loss_txt_path = os.path.join(save_dir, "loss_total.txt")
    with open(loss_txt_path, "w", encoding="utf-8") as f:
        f.write("total_loss\n")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        ut_pred = model(u_train)
        deriv_loss = loss_fn(ut_pred, ut_train)

        curr_rollout_steps = get_curriculum_rollout_steps(
            epoch=epoch,
            total_epochs=epochs,
            max_rollout_steps=max_rollout_steps,
            warmup_ratio=warmup_ratio,
        )

        rollout_loss, starts = compute_rollout_loss(
            model=model,
            u_train=u_train,
            dt=dt,
            rollout_steps=curr_rollout_steps,
            num_rollout_starts=num_rollout_starts,
            loss_fn=loss_fn,
        )

        total_loss = lambda_deriv * deriv_loss + lambda_roll * rollout_loss
        total_loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        loss_total_val = total_loss.item()
        loss_deriv_val = deriv_loss.item()
        loss_roll_val = rollout_loss.item()
        lr_now = get_current_lr(optimizer)

        with open(loss_txt_path, "a", encoding="utf-8") as f:
            f.write(f"{loss_total_val:.8e}\n")

        history["total"].append(loss_total_val)
        history["deriv"].append(loss_deriv_val)
        history["rollout"].append(loss_roll_val)
        history["lr"].append(lr_now)
        history["rollout_steps"].append(curr_rollout_steps)

        if loss_total_val < best_loss:
            best_loss = loss_total_val
            wait = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            wait += 1

        if epoch % 100 == 0:
            print(
                f"[{epoch:5d}] "
                f"total = {loss_total_val:.6e}, "
                f"deriv = {loss_deriv_val:.6e}, "
                f"roll = {loss_roll_val:.6e}, "
                f"roll_steps = {curr_rollout_steps}, "
                f"lr = {lr_now:.3e}, "
                f"starts = {starts}"
            )

        if wait > patience:
            print("Early stopping at epoch", epoch)
            break

    train_time = time.time() - start_time
    print("Training time:", train_time)

    return history, best_loss, train_time


# ============================================================
# 5. Visualization
# ============================================================


def save_figure(fig, subdir, filename):
    folder = os.path.join(figure_dir, subdir)
    os.makedirs(folder, exist_ok=True)

    svg_path = os.path.join(folder, f"{filename}.svg")
    pdf_path = os.path.join(folder, f"{filename}.pdf")
    fig.savefig(svg_path, bbox_inches="tight", dpi=400)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=400)
    plt.close(fig)


def save_clean_field(data, subdir, filename, vmin, vmax, cmap="RdBu_r"):
    folder = os.path.join(figure_dir, subdir)
    os.makedirs(folder, exist_ok=True)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    plt.imshow(
        data,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    svg_path = os.path.join(folder, f"{filename}.svg")
    pdf_path = os.path.join(folder, f"{filename}.pdf")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0, dpi=400)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0, dpi=400)
    plt.close(fig)


def plot_training_curves(history):
    fig = plt.figure(figsize=(7, 5))
    plt.semilogy(history["total"], label="Total Loss")
    plt.semilogy(history["deriv"], label="dudt Loss")
    plt.semilogy(history["rollout"], label="Rollout Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    save_figure(fig, "training", "training_curve")

    fig = plt.figure(figsize=(7, 4))
    plt.plot(history["lr"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("OneCycleLR")
    plt.tight_layout()
    save_figure(fig, "training", "lr_curve")

    fig = plt.figure(figsize=(7, 4))
    plt.plot(history["rollout_steps"])
    plt.xlabel("Epoch")
    plt.ylabel("Rollout Steps")
    plt.title("Curriculum Rollout Steps")
    plt.tight_layout()
    save_figure(fig, "training", "rollout_steps_curve")


def save_error_scale(subdir, filename, err_lim, scale_true):
    folder = os.path.join(figure_dir, subdir)
    os.makedirs(folder, exist_ok=True)

    txt_path = os.path.join(folder, f"{filename}_scale.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"scale_true = {scale_true:.8e}\n")
        f.write(f"err_lim = {err_lim:.8e}\n")
        f.write(f"error_range = [{-err_lim:.8e}, {err_lim:.8e}]\n")
        f.write("definition: error_norm = (pred - true) / max(abs(true))\n")

    summary_path = os.path.join(folder, "error_scales.txt")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(
            f"{filename}: "
            f"scale_true={scale_true:.8e}, "
            f"err_lim={err_lim:.8e}, "
            f"range=[{-err_lim:.8e}, {err_lim:.8e}]\n"
        )


def plot_dudt_field_comparison(ut_true, ut_pred, sample_id=0):
    ut_true_np = ut_true[sample_id, 0].detach().cpu().numpy()
    ut_pred_np = ut_pred[sample_id, 0].detach().cpu().numpy()

    coeff_true = np.max(np.abs(ut_true_np))
    coeff_true = max(coeff_true, 1e-12)

    ut_true_norm = ut_true_np / coeff_true
    ut_pred_norm = ut_pred_np / coeff_true

    error = ut_pred_np - ut_true_np
    error_norm = error / coeff_true

    err_lim = np.max(np.abs(error_norm))
    err_lim = max(err_lim, 1e-12)

    save_clean_field(
        ut_true_norm,
        subdir="dudt",
        filename=f"sample{sample_id}_true",
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
    )

    save_clean_field(
        ut_pred_norm,
        subdir="dudt",
        filename=f"sample{sample_id}_predict",
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
    )

    save_clean_field(
        error_norm,
        subdir="dudt",
        filename=f"sample{sample_id}_error",
        vmin=-err_lim,
        vmax=err_lim,
        cmap="RdBu_r",
    )

    save_error_scale(
        subdir="dudt",
        filename=f"sample{sample_id}_error",
        err_lim=err_lim,
        scale_true=coeff_true,
    )


def plot_rollout_comparison(u_true, u_roll, sample_id=0):
    u_true_np = u_true[sample_id, 0].detach().cpu().numpy()
    u_roll_np = u_roll[sample_id, 0].detach().cpu().numpy()

    coeff_true = np.max(np.abs(u_true_np))
    coeff_true = max(coeff_true, 1e-12)

    u_true_norm = u_true_np / coeff_true * 3
    u_roll_norm = u_roll_np / coeff_true * 3

    error = u_roll_np - u_true_np
    error_norm = error / coeff_true

    err_lim = np.max(np.abs(error_norm))
    err_lim = max(err_lim, 1e-12)

    save_clean_field(
        u_true_norm,
        subdir="rollout",
        filename=f"sample{sample_id}_true",
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
    )

    save_clean_field(
        u_roll_norm,
        subdir="rollout",
        filename=f"sample{sample_id}_predict",
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
    )

    save_clean_field(
        error_norm,
        subdir="rollout",
        filename=f"sample{sample_id}_error",
        vmin=-err_lim,
        vmax=err_lim,
        cmap="RdBu_r",
    )

    save_error_scale(
        subdir="rollout",
        filename=f"sample{sample_id}_error",
        err_lim=err_lim,
        scale_true=coeff_true,
    )


def plot_six_test_comparisons(u_true, u_roll):
    n = min(6, u_true.shape[0])
    for i in range(n):
        plot_rollout_comparison(u_true, u_roll, sample_id=i)


def plot_slice_comparison(u_true, u_roll, sample_id=0, time_index=None):
    if time_index is None:
        time_index = u_true.shape[-1] // 2

    true_slice = u_true[sample_id, 0, :, time_index].detach().cpu().numpy()
    pred_slice = u_roll[sample_id, 0, :, time_index].detach().cpu().numpy()

    Nx = len(true_slice)
    x_norm = np.linspace(0.0, 1.0, Nx)

    coeff = max(np.max(np.abs(true_slice)), np.max(np.abs(pred_slice)))
    coeff = max(coeff, 1e-12)

    true_slice_norm = np.clip(true_slice / coeff, -1.0, 1.0)
    pred_slice_norm = np.clip(pred_slice / coeff, -1.0, 1.0)

    norm_time = 0.0
    if u_true.shape[-1] > 1:
        norm_time = time_index / (u_true.shape[-1] - 1)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    plt.plot(x_norm, true_slice_norm, label="True")
    plt.plot(x_norm, pred_slice_norm, "--", label="Rollout Pred")
    plt.xlabel("Normalized space")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.title(f"Slice at normalized time = {norm_time:.3f}")
    ax.set_xticks([0.0, 0.5, 1.0])
    plt.tight_layout()
    save_figure(fig, "slice", "slice_compare")


# ============================================================
# 6. Main program
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1D Diffusion Time-Marching Training (Periodic BC)"
    )

    parser.add_argument("--hidden", type=int, default=4, help="hidden layer size")
    parser.add_argument(
        "--epochs", type=int, default=20000, help="number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="max learning rate for OneCycleLR"
    )
    parser.add_argument(
        "--patience", type=int, default=500, help="early stopping patience"
    )
    parser.add_argument(
        "--data_dir", type=str, default="train_data", help="training data directory"
    )
    parser.add_argument(
        "--test_path", type=str, default="test_data/test.pt", help="test file path"
    )
    parser.add_argument(
        "--save_dir", type=str, default="LGNO_4hidden_results", help="save directory"
    )

    parser.add_argument(
        "--lambda_deriv",
        type=float,
        default=1.0,
        help="weight of dudt supervision loss",
    )
    parser.add_argument(
        "--lambda_roll", type=float, default=1.0, help="weight of rollout loss"
    )
    parser.add_argument(
        "--max_rollout_steps",
        type=int,
        default=10,
        help="maximum rollout steps used in training",
    )
    parser.add_argument(
        "--num_rollout_starts",
        type=int,
        default=2,
        help="how many random rollout start points per epoch",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.7,
        help="curriculum warmup ratio for rollout steps",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="gradient clipping norm, <=0 means disable",
    )

    parser.add_argument(
        "--learn_coeff",
        action="store_true",
        help="whether to use MLP-predicted coefficients instead of fixed periodic Laplacian",
    )

    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    figure_dir = os.path.join(save_dir, "figure")
    training_figure_dir = os.path.join(figure_dir, "training")
    dudt_figure_dir = os.path.join(figure_dir, "dudt")
    rollout_figure_dir = os.path.join(figure_dir, "rollout")
    slice_figure_dir = os.path.join(figure_dir, "slice")

    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(training_figure_dir, exist_ok=True)
    os.makedirs(dudt_figure_dir, exist_ok=True)
    os.makedirs(rollout_figure_dir, exist_ok=True)
    os.makedirs(slice_figure_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Training configuration:")
    print(f"hidden              : {args.hidden}")
    print(f"epochs              : {args.epochs}")
    print(f"lr (max_lr)         : {args.lr}")
    print(f"patience            : {args.patience}")
    print(f"data_dir            : {args.data_dir}")
    print(f"test_path           : {args.test_path}")
    print(f"save_dir            : {args.save_dir}")
    print(f"lambda_deriv        : {args.lambda_deriv}")
    print(f"lambda_roll         : {args.lambda_roll}")
    print(f"max_rollout_steps   : {args.max_rollout_steps}")
    print(f"num_rollout_starts  : {args.num_rollout_starts}")
    print(f"warmup_ratio        : {args.warmup_ratio}")
    print(f"grad_clip           : {args.grad_clip}")
    print(f"learn_coeff         : {args.learn_coeff}")
    print("=" * 60 + "\n")

    u_train, ut_train, dt_train, dx_train, kappa_train, bc_train = load_all_train(
        args.data_dir
    )

    if str(bc_train).lower() != "periodic":
        print(
            f"[Warning] training data boundary_condition = {bc_train}, but this script assumes periodic BC."
        )

    model = LocalCoeffStencilPeriodic(
        dx=dx_train, kappa=kappa_train, hidden=args.hidden, learn_coeff=args.learn_coeff
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("Model information:")
    print(f"Network Structure : 3-layer MLP, hidden={args.hidden}")
    print(f"Total parameters  : {total_params}")
    print(f"Trainable params  : {trainable_params}")
    print(f"dx                : {dx_train}")
    print(f"kappa             : {kappa_train}")
    print("=" * 60 + "\n")

    history, best_loss, training_time = train_model(
        model=model,
        u_train=u_train,
        ut_train=ut_train,
        dt=dt_train,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        lambda_deriv=args.lambda_deriv,
        lambda_roll=args.lambda_roll,
        max_rollout_steps=args.max_rollout_steps,
        num_rollout_starts=args.num_rollout_starts,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
    )

    plot_training_curves(history)

    model.load_state_dict(
        torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device)
    )
    model.eval()

    with torch.no_grad():
        ut_pred_train = predict_dudt_sequence(model, u_train)
        rel_err_train_dudt = relative_l2(ut_pred_train, ut_train).item()
        plot_dudt_field_comparison(ut_train, ut_pred_train, sample_id=0)

    u_test, ut_test, t_test, dt_test, dx_test, kappa_test, bc_test = load_test_data(
        args.test_path
    )

    if str(bc_test).lower() != "periodic":
        print(
            f"[Warning] test data boundary_condition = {bc_test}, but this script assumes periodic BC."
        )

    with torch.no_grad():
        if ut_test is not None:
            ut_pred_test = predict_dudt_sequence(model, u_test)
            rel_err_test_dudt = relative_l2(ut_pred_test, ut_test).item()
            plot_dudt_field_comparison(ut_test, ut_pred_test, sample_id=0)
        else:
            ut_pred_test = None
            rel_err_test_dudt = None

        u_roll = rollout_time_marching(model, u_test, dt_test)
        rel_err_rollout = relative_l2(u_roll, u_test).item()

        plot_slice_comparison(
            u_true=u_test, u_roll=u_roll, sample_id=0, time_index=u_test.shape[-1] // 2
        )

        plot_six_test_comparisons(u_test, u_roll)

    print("\n" + "=" * 60)
    print("Training result summary:")
    print(f"Best training loss               : {best_loss:.6e}")
    print(f"Training time                    : {training_time:.2f} seconds")
    print(f"Train relative L2 error of du/dt : {rel_err_train_dudt:.6e}")
    if rel_err_test_dudt is not None:
        print(f"Test relative L2 error of du/dt  : {rel_err_test_dudt:.6e}")
    print(f"Test rollout relative L2 error   : {rel_err_rollout:.6e}")
    print(f"Total parameters                 : {total_params}")
    print("=" * 60 + "\n")

    with open(os.path.join(save_dir, "training_log.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Model training log (Periodic BC)\n")
        f.write("=" * 60 + "\n")

        f.write("Training configuration:\n")
        f.write(f"hidden              : {args.hidden}\n")
        f.write(f"epochs              : {args.epochs}\n")
        f.write(f"lr (max_lr)         : {args.lr}\n")
        f.write(f"patience            : {args.patience}\n")
        f.write(f"data_dir            : {args.data_dir}\n")
        f.write(f"test_path           : {args.test_path}\n")
        f.write(f"save_dir            : {args.save_dir}\n")
        f.write(f"lambda_deriv        : {args.lambda_deriv}\n")
        f.write(f"lambda_roll         : {args.lambda_roll}\n")
        f.write(f"max_rollout_steps   : {args.max_rollout_steps}\n")
        f.write(f"num_rollout_starts  : {args.num_rollout_starts}\n")
        f.write(f"warmup_ratio        : {args.warmup_ratio}\n")
        f.write(f"grad_clip           : {args.grad_clip}\n")
        f.write(f"learn_coeff         : {args.learn_coeff}\n")
        f.write("\n")

        f.write("Data parameters:\n")
        f.write(f"dx_train            : {dx_train}\n")
        f.write(f"dt_train            : {dt_train}\n")
        f.write(f"kappa_train         : {kappa_train}\n")
        f.write(f"bc_train            : {bc_train}\n")
        f.write(f"dx_test             : {dx_test}\n")
        f.write(f"dt_test             : {dt_test}\n")
        f.write(f"kappa_test          : {kappa_test}\n")
        f.write(f"bc_test             : {bc_test}\n")
        f.write("\n")

        f.write("Model information:\n")
        f.write(f"Network Structure   : 3-layer MLP, hidden={args.hidden}\n")
        f.write(f"Total parameters    : {total_params}\n")
        f.write(f"Trainable params    : {trainable_params}\n")
        f.write("\n")

        f.write("Training result summary:\n")
        f.write(f"Best training loss               : {best_loss:.6e}\n")
        f.write(f"Training time                    : {training_time:.2f} seconds\n")
        f.write(f"Train relative L2 error of du/dt : {rel_err_train_dudt:.6e}\n")
        if rel_err_test_dudt is not None:
            f.write(f"Test relative L2 error of du/dt  : {rel_err_test_dudt:.6e}\n")
        else:
            f.write("Test relative L2 error of du/dt  : None\n")
        f.write(f"Test rollout relative L2 error   : {rel_err_rollout:.6e}\n")
        f.write("\n")

        f.write("Output files:\n")
        f.write(f"loss txt            : {os.path.join(save_dir, 'loss_total.txt')}\n")
        f.write(f"best model          : {os.path.join(save_dir, 'best_model.pth')}\n")
        f.write(f"figure dir          : {figure_dir}\n")

    print("All results saved in folder:", save_dir)
