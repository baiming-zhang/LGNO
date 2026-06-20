# ============================================================
# 2D vector time-marching operator learning
# Learn local term only: (u, v) -> local part of (du/dt, dv/dt)
# Total derivative used in loss / test / rollout:
#     dudt_total = local_pred + g
# Rollout: Euler time stepping
# Keep structure: periodic 3x3 stencil + MLP-produced local weights
# + linear dot product
# ============================================================

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.colors import TwoSlopeNorm
from torch.utils.data import TensorDataset, DataLoader

hidden = 16
# ============================================================
# 0. Global settings
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

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 15
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.02

# ============================================================
# 1. Create output directories
# ============================================================

def get_run_dir_from_cli():
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--run_dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--run_dir="):
            return arg.split("=", 1)[1]
    return None


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = get_run_dir_from_cli() or os.path.join(
    f"LGNO_{hidden}hidden_results", f"run_{timestamp}"
)
os.makedirs(base_dir, exist_ok=True)

fig_dir = os.path.join(base_dir, "figures")
pred_dir = os.path.join(fig_dir, "predict")
truth_dir = os.path.join(fig_dir, "truth")
error_dir = os.path.join(fig_dir, "error")
colorbar_dir = os.path.join(fig_dir, "colorbar")
loss_dir = os.path.join(fig_dir, "loss")

for d in [fig_dir, pred_dir, truth_dir, error_dir, colorbar_dir, loss_dir]:
    os.makedirs(d, exist_ok=True)

log_file = os.path.join(base_dir, "train_log.txt")


def log_print(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


log_print(f"Device: {device}")
log_print(f"Result directory: {base_dir}")

# ============================================================
# 2. Data loading
# Expected dataset keys:
#   u    : [Ns, Nt, 1, H, W]
#   v    : [Ns, Nt, 1, H, W]
#   g_u  : [Ns, Nt, 1, H, W]
#   g_v  : [Ns, Nt, 1, H, W]
#   dudt : [Ns, Nt, 1, H, W]
#   dvdt : [Ns, Nt, 1, H, W]
#   x, y, t
#
# Internal format:
#   vel_seq        = cat([u, v], dim=2)        -> [Ns, Nt, 2, H, W]
#   g_seq          = cat([g_u, g_v], dim=2)    -> [Ns, Nt, 2, H, W]
#   dudt_total_seq = cat([dudt, dvdt], dim=2)  -> [Ns, Nt, 2, H, W]
# ============================================================


def load_sequence_dataset(folder, mode="train"):
    file_path = os.path.join(folder, f"{mode}.pt")
    data = torch.load(file_path, map_location="cpu")

    required_keys = ["u", "v", "g_u", "g_v", "dudt", "dvdt"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {file_path}")

    u = data["u"].float()
    v = data["v"].float()
    g_u = data["g_u"].float()
    g_v = data["g_v"].float()
    dudt = data["dudt"].float()
    dvdt = data["dvdt"].float()

    for name, tensor in [
        ("u", u),
        ("v", v),
        ("g_u", g_u),
        ("g_v", g_v),
        ("dudt", dudt),
        ("dvdt", dvdt),
    ]:
        if tensor.dim() != 5:
            raise ValueError(
                f"Expected {name} shape [Ns, Nt, 1, H, W], got {tuple(tensor.shape)}"
            )
        if tensor.shape[2] != 1:
            raise ValueError(
                f"Expected {name} channel dim = 1, got shape {tuple(tensor.shape)}"
            )

    base_shape = u.shape
    for name, tensor in [
        ("v", v),
        ("g_u", g_u),
        ("g_v", g_v),
        ("dudt", dudt),
        ("dvdt", dvdt),
    ]:
        if tensor.shape != base_shape:
            raise ValueError(
                f"Shape mismatch: u {tuple(base_shape)} vs {name} {tuple(tensor.shape)}"
            )

    vel_seq = torch.cat([u, v], dim=2)  # [Ns, Nt, 2, H, W]
    g_seq = torch.cat([g_u, g_v], dim=2)  # [Ns, Nt, 2, H, W]
    dudt_total_seq = torch.cat([dudt, dvdt], dim=2)  # [Ns, Nt, 2, H, W]

    x = data.get("x", None)
    y = data.get("y", None)
    t = data.get("t", None)
    metadata = data.get("metadata", {})

    return (
        vel_seq.to(device),
        g_seq.to(device),
        dudt_total_seq.to(device),
        x,
        y,
        t,
        metadata,
    )


def flatten_time_triplets(vel_seq, g_seq, dudt_total_seq):
    """
    vel_seq        : [Ns, Nt, 2, H, W]
    g_seq          : [Ns, Nt, 2, H, W]
    dudt_total_seq : [Ns, Nt, 2, H, W]
    return         : [Ns*Nt, 2, H, W], [Ns*Nt, 2, H, W], [Ns*Nt, 2, H, W]
    """
    Ns, Nt, C, H, W = vel_seq.shape
    vel_flat = vel_seq.reshape(Ns * Nt, C, H, W)
    g_flat = g_seq.reshape(Ns * Nt, C, H, W)
    dudt_flat = dudt_total_seq.reshape(Ns * Nt, C, H, W)
    return vel_flat, g_flat, dudt_flat


# ============================================================
# 3. Model
# Keep same idea:
# 3x3 periodic stencil -> MLP generates local weights -> linear dot product
#
# Network input channels = 2:
#   [u, v]
# Network output channels = 2:
#   local part of [du/dt, dv/dt]
# ============================================================


class VectorStencilLGNO(nn.Module):
    def __init__(self, hidden=16, in_channels=2, out_channels=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stencil_points = 9
        self.local_dim = in_channels * self.stencil_points

        self.net = nn.Sequential(
            nn.Linear(self.local_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_channels * self.local_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u):
        """
        u: [B, 2, H, W]
        return: [B, 2, H, W] = local part of [du/dt, dv/dt]
        """
        B, C, H, W = u.shape
        if C != self.in_channels:
            raise ValueError(f"Expected input channels = {self.in_channels}, got {C}")

        stencil_list = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                shifted = torch.roll(u, shifts=(dy, dx), dims=(2, 3))
                stencil_list.append(shifted)

        # [B, 2*9, H, W]
        stencil = torch.cat(stencil_list, dim=1)

        # [B*H*W, local_dim]
        local = stencil.permute(0, 2, 3, 1).reshape(-1, self.local_dim)

        # [B*H*W, out_channels * local_dim]
        weights = self.net(local)

        # [B*H*W, 2, local_dim]
        weights = weights.view(-1, self.out_channels, self.local_dim)

        # linear dot product
        out = torch.einsum("boj,bj->bo", weights, local)

        pred = out.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return pred


# ============================================================
# 4. Save helpers
# ============================================================


def save_fig_all_formats(fig, save_stem):
    fig.savefig(save_stem + ".pdf")
    fig.savefig(save_stem + ".svg")
    fig.savefig(save_stem + ".png", dpi=400)
    plt.close(fig)


def save_single_field_image(field2d, save_stem, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(
        field2d,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        extent=[-1.0, 1.0, -1.0, 1.0],
    )
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.tick_params(axis="both", direction="out", length=3, width=0.8, labelsize=30)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=30)
    fig.tight_layout()
    save_fig_all_formats(fig, save_stem)


def save_standalone_colorbar(
    save_stem, cmap, vmin, vmax, center_zero=False, label=None
):
    fig, ax = plt.subplots(figsize=(1.3, 4.8))

    if center_zero:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=vmin, vmax=vmax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=ax)
    if label is not None:
        cbar.set_label(label)

    save_fig_all_formats(fig, save_stem)


def save_loss_curve(loss_history, save_stem):
    epochs = np.arange(len(loss_history))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(epochs, loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_fig_all_formats(fig, save_stem)


# ============================================================
# 5. Training
# Network predicts local term only
# Total derivative used for loss:
#   dudt_pred_total = local_pred + g
# ============================================================


def train(model, train_loader, lr, epochs, patience):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=300, threshold=1e-6, min_lr=1e-6
    )

    loss_fn = nn.MSELoss()

    best_loss = 1e20
    wait = 0
    loss_history = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_vel, batch_g, batch_dudt_total in train_loader:
            optimizer.zero_grad()

            local_pred = model(batch_vel)
            dudt_pred_total = local_pred + batch_g

            loss = loss_fn(dudt_pred_total, batch_dudt_total)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        loss_val = running_loss / max(num_batches, 1)
        scheduler.step(loss_val)
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            wait = 0
            torch.save(model.state_dict(), os.path.join(base_dir, "best_model.pth"))
        else:
            wait += 1

        if wait > patience:
            log_print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            log_print(f"[{epoch:6d}] loss = {loss_val:.6e} | lr = {current_lr:.3e}")

    total_time = time.time() - start_time
    log_print(f"Training time: {total_time:.2f} sec")

    loss_history = np.array(loss_history, dtype=np.float64)

    np.save(os.path.join(base_dir, "loss_history.npy"), loss_history)
    np.savetxt(os.path.join(base_dir, "loss_total.txt"), loss_history, fmt="%.12e")
    save_loss_curve(loss_history, os.path.join(loss_dir, "training_loss"))

    return total_time, loss_history


# ============================================================
# 6. Evaluation helpers
# One-step test still saves TOTAL dudt prediction,
# so file names / downstream format remain unchanged.
# ============================================================


def evaluate_one_step(model, test_loader):
    model.eval()
    pred_list = []
    true_list = []

    with torch.no_grad():
        for batch_vel, batch_g, batch_dudt_total in test_loader:
            local_pred = model(batch_vel)
            dudt_pred_total = local_pred + batch_g

            pred_list.append(dudt_pred_total.cpu())
            true_list.append(batch_dudt_total.cpu())

    pred = torch.cat(pred_list, dim=0)
    true = torch.cat(true_list, dim=0)

    rel_err = torch.norm(pred - true) / (torch.norm(true) + 1e-20)
    mse_err = torch.mean((pred - true) ** 2)

    return pred, true, rel_err.item(), mse_err.item()


def rollout_euler(model, init_state, g_seq, nt, dt):
    """
    init_state: [2, H, W]
    g_seq     : [Nt, 2, H, W]
    return    : [Nt, 2, H, W]
    """
    traj = [init_state.detach().clone()]
    cur = init_state.detach().clone()

    model.eval()
    with torch.no_grad():
        for n in range(nt - 1):
            g_cur = g_seq[n].detach().clone()  # [2, H, W]
            local_pred = model(cur.unsqueeze(0)).squeeze(0)
            dudt_pred_total = local_pred + g_cur
            cur = cur + dt * dudt_pred_total
            traj.append(cur.detach().clone())

    return torch.stack(traj, dim=0)


def rollout_all_samples(model, vel_test_seq, g_test_seq, dt):
    """
    vel_test_seq: [Ns, Nt, 2, H, W]
    g_test_seq  : [Ns, Nt, 2, H, W]
    """
    Ns, Nt, C, H, W = vel_test_seq.shape
    rollout_list = []

    for s in range(Ns):
        init_state = vel_test_seq[s, 0]
        g_seq = g_test_seq[s]
        rollout = rollout_euler(model, init_state, g_seq, Nt, dt)
        rollout_list.append(rollout.cpu())

        log_print(f"Rollout sample {s:03d} done.")

    rollout_all = torch.stack(rollout_list, dim=0)  # [Ns, Nt, 2, H, W]
    return rollout_all


# ============================================================
# 7. Visualization
# Save every 50 frames:
#   truth / rollout predict / error
# separate for u and v
# ============================================================


def save_rollout_images(rollout_pred, rollout_truth, step_stride=50):
    """
    rollout_pred : [Ns, Nt, 2, H, W] on cpu
    rollout_truth: [Ns, Nt, 2, H, W] on cpu
    """
    pred_np = rollout_pred.numpy()
    true_np = rollout_truth.numpy()

    Ns, Nt, C, H, W = pred_np.shape
    comp_names = ["u", "v"]

    frame_ids = list(range(0, Nt, step_stride))
    if (Nt - 1) not in frame_ids:
        frame_ids.append(Nt - 1)

    for s in range(Ns):
        for n in frame_ids:
            for c in range(C):
                name = comp_names[c]

                pred_img = pred_np[s, n, c]
                true_img = true_np[s, n, c]
                err_img = pred_img - true_img

                vmax = np.max(np.abs(true_img))
                if vmax == 0:
                    vmax = 1.0
                vmin = -vmax

                err_abs = np.max(np.abs(err_img))
                if err_abs == 0:
                    err_abs = 1.0

                pred_name = os.path.join(
                    pred_dir, f"sample{s:03d}_step{n:04d}_{name}_predict"
                )
                true_name = os.path.join(
                    truth_dir, f"sample{s:03d}_step{n:04d}_{name}_truth"
                )
                err_name = os.path.join(
                    error_dir, f"sample{s:03d}_step{n:04d}_{name}_error"
                )

                save_single_field_image(
                    field2d=pred_img,
                    save_stem=pred_name,
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                )

                save_single_field_image(
                    field2d=true_img,
                    save_stem=true_name,
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                )

                save_single_field_image(
                    field2d=err_img,
                    save_stem=err_name,
                    cmap="RdBu_r",
                    vmin=-err_abs,
                    vmax=err_abs,
                )

                save_standalone_colorbar(
                    save_stem=os.path.join(
                        colorbar_dir, f"sample{s:03d}_step{n:04d}_{name}_field_colorbar"
                    ),
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                    center_zero=True,
                    label=rf"${name}$",
                )

                save_standalone_colorbar(
                    save_stem=os.path.join(
                        colorbar_dir, f"sample{s:03d}_step{n:04d}_{name}_error_colorbar"
                    ),
                    cmap="seismic",
                    vmin=-err_abs,
                    vmax=err_abs,
                    center_zero=True,
                    label=f"{name} error",
                )

    log_print("Rollout figures saved every 50 frames.")


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=hidden)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_dir", type=str, default="dataset/train_data")
    parser.add_argument("--test_dir", type=str, default="dataset/test_data")
    parser.add_argument("--save_stride", type=int, default=50)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--run_dir", type=str, default=base_dir)

    args = parser.parse_args()

    if args.run_dir != base_dir:
        raise ValueError(
            "run_dir must be provided before startup. Use --run_dir so output "
            "directories are configured before logging begins."
        )

    if not args.eval_only:
        with open(os.path.join(base_dir, "config.txt"), "w", encoding="utf-8") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")
    else:
        log_print("Eval-only mode: preserving existing config.txt and skipping training.")

    # --------------------------------------------------------
    # Load sequence dataset
    # --------------------------------------------------------
    (
        vel_train_seq,
        g_train_seq,
        dudt_train_seq,
        x_train,
        y_train,
        t_train,
        meta_train,
    ) = load_sequence_dataset(args.train_dir, "train")

    vel_test_seq, g_test_seq, dudt_test_seq, x_test, y_test, t_test, meta_test = (
        load_sequence_dataset(args.test_dir, "test")
    )

    log_print(f"Train velocity shape       : {tuple(vel_train_seq.shape)}")
    log_print(f"Train g shape              : {tuple(g_train_seq.shape)}")
    log_print(f"Train total dudt shape     : {tuple(dudt_train_seq.shape)}")
    log_print(f"Test  velocity shape       : {tuple(vel_test_seq.shape)}")
    log_print(f"Test  g shape              : {tuple(g_test_seq.shape)}")
    log_print(f"Test  total dudt shape     : {tuple(dudt_test_seq.shape)}")
    log_print(f"Train metadata: {meta_train}")
    log_print(f"Test  metadata: {meta_test}")

    # time step
    if t_train is not None and len(t_train) > 1:
        dt = float((t_train[1] - t_train[0]).item())
    else:
        raise ValueError("Dataset must contain time vector t with at least 2 entries.")

    log_print(f"Detected dt = {dt:.6e}")

    # --------------------------------------------------------
    # Flatten all time pairs for one-step training
    # --------------------------------------------------------
    vel_train_flat, g_train_flat, dudt_train_flat = flatten_time_triplets(
        vel_train_seq, g_train_seq, dudt_train_seq
    )
    vel_test_flat, g_test_flat, dudt_test_flat = flatten_time_triplets(
        vel_test_seq, g_test_seq, dudt_test_seq
    )

    log_print(f"Flattened train velocity shape : {tuple(vel_train_flat.shape)}")
    log_print(f"Flattened train g shape        : {tuple(g_train_flat.shape)}")
    log_print(f"Flattened train target shape   : {tuple(dudt_train_flat.shape)}")
    log_print(f"Flattened test velocity shape  : {tuple(vel_test_flat.shape)}")
    log_print(f"Flattened test g shape         : {tuple(g_test_flat.shape)}")
    log_print(f"Flattened test target shape    : {tuple(dudt_test_flat.shape)}")

    train_dataset = TensorDataset(vel_train_flat, g_train_flat, dudt_train_flat)
    test_dataset = TensorDataset(vel_test_flat, g_test_flat, dudt_test_flat)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = VectorStencilLGNO(hidden=args.hidden, in_channels=2, out_channels=2).to(
        device
    )
    log_print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    if not args.eval_only:
        train_time, loss_history = train(
            model=model,
            train_loader=train_loader,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
        )

    # --------------------------------------------------------
    # One-step test
    # --------------------------------------------------------
    model.load_state_dict(
        torch.load(os.path.join(base_dir, "best_model.pth"), map_location=device)
    )
    model.eval()

    dudt_pred_test, dudt_true_test, rel_err_1step, mse_err_1step = evaluate_one_step(
        model, test_loader
    )

    log_print(f"One-step dudt Relative L2 error: {rel_err_1step:.6e}")
    log_print(f"One-step dudt MSE error        : {mse_err_1step:.6e}")

    torch.save(dudt_pred_test, os.path.join(base_dir, "test_dudt_pred.pt"))
    torch.save(dudt_true_test, os.path.join(base_dir, "test_dudt_true.pt"))

    # --------------------------------------------------------
    # Rollout test
    # --------------------------------------------------------
    rollout_pred = rollout_all_samples(model, vel_test_seq, g_test_seq, dt=dt)  # cpu
    rollout_true = vel_test_seq.detach().cpu()

    rollout_rel = torch.norm(rollout_pred - rollout_true) / (
        torch.norm(rollout_true) + 1e-20
    )
    rollout_mse = torch.mean((rollout_pred - rollout_true) ** 2)

    log_print(f"Rollout Relative L2 error: {rollout_rel.item():.6e}")
    log_print(f"Rollout MSE error        : {rollout_mse.item():.6e}")

    torch.save(rollout_pred, os.path.join(base_dir, "rollout_pred.pt"))
    torch.save(rollout_true, os.path.join(base_dir, "rollout_true.pt"))

    # --------------------------------------------------------
    # Save every 50 frames
    # --------------------------------------------------------
    save_rollout_images(
        rollout_pred=rollout_pred,
        rollout_truth=rollout_true,
        step_stride=args.save_stride,
    )

    log_print("All results saved successfully.")
