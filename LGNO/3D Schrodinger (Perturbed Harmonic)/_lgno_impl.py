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

# ============================================================
# 3D Schrodinger / operator learning
# Learn: u -> f
# Periodic-learning full version
#
# Input:
#   u = [Re(psi), Im(psi), V]              -> [B,3,Nx,Ny,Nz] or [1,3,Nx,Ny,Nz,Nt]
# Output:
#   f = [Re(Hpsi), Im(Hpsi)]               -> [B,2,Nx,Ny,Nz] or [1,2,Nx,Ny,Nz,Nt]
#
# Keep the same local-gradient learning style:
#   periodic local stencil + MLP-generated weights + local dot product
#
# NOTE:
#   The dataset generator stores one single trajectory with a time axis:
#       u: [1, 3, Nx, Ny, Nz, Nt]
#       f: [1, 2, Nx, Ny, Nz, Nt]
#   Here we only reshape time into the sample axis for training/testing.
#   All result folders / logs / saved outputs keep the original style.
# ============================================================

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
fontsize = 45
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = fontsize
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["legend.fontsize"] = fontsize
plt.rcParams["figure.titlesize"] = fontsize
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.02

# ============================================================
# 1. Create output directories
# ============================================================

def get_cli_hidden(default=16):
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--hidden" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if arg.startswith("--hidden="):
            return int(arg.split("=", 1)[1])
    return default


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = os.path.join(f"LGNO_{get_cli_hidden()}hidden_results", f"run_{timestamp}")
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
# ============================================================


def _find_dataset_file(folder, mode):
    """
    Prefer:
        train_dataset.pt / test_dataset.pt
    Fallback:
        train.pt / test.pt
    """
    candidates = [
        os.path.join(folder, f"{mode}_dataset.pt"),
        os.path.join(folder, f"{mode}.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Cannot find dataset file for mode='{mode}' in folder='{folder}'. "
        f"Tried: {candidates}"
    )


def _reshape_to_3d_samples(t, expected_channels, name):
    """
    Convert dataset tensors to [B_samples, C, Nx, Ny, Nz].

    Supported inputs:
      - [B, C, Nx, Ny, Nz, Nt]   -> [B*Nt, C, Nx, Ny, Nz]
      - [B, C, Nx, Ny, Nz]       -> unchanged
      - [C, Nx, Ny, Nz]          -> [1, C, Nx, Ny, Nz]
    """
    if not torch.is_tensor(t):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t)}")

    if t.dim() == 6:
        if t.shape[1] != expected_channels:
            raise ValueError(
                f"{name} expected channel dimension = {expected_channels}, got shape = {tuple(t.shape)}"
            )
        # [B,C,Nx,Ny,Nz,Nt] -> [B,Nt,C,Nx,Ny,Nz] -> [B*Nt,C,Nx,Ny,Nz]
        B, C, Nx, Ny, Nz, Nt = t.shape
        t = t.permute(0, 5, 1, 2, 3, 4).contiguous().view(B * Nt, C, Nx, Ny, Nz)
        return t

    if t.dim() == 5:
        if t.shape[1] == expected_channels:
            return t.contiguous()
        if t.shape[0] == expected_channels:
            return t.unsqueeze(0).contiguous()
        raise ValueError(
            f"{name} expected [B,C,Nx,Ny,Nz] or [C,Nx,Ny,Nz] with C={expected_channels}, got {tuple(t.shape)}"
        )

    if t.dim() == 4:
        if t.shape[0] != expected_channels:
            raise ValueError(
                f"{name} expected [C,Nx,Ny,Nz] with C={expected_channels}, got shape = {tuple(t.shape)}"
            )
        return t.unsqueeze(0).contiguous()

    raise ValueError(f"Unsupported tensor shape for {name}: {tuple(t.shape)}")


def load_dataset(folder, mode="train"):
    """
    3D Schrodinger dataset version.

    Expected saved tensors from the generator:
        u: [1,3,Nx,Ny,Nz,Nt]
        f: [1,2,Nx,Ny,Nz,Nt]

    Returned tensors:
        u: [Ns,3,Nx,Ny,Nz]
        f: [Ns,2,Nx,Ny,Nz]
    where Ns = B * Nt.
    """
    file_path = _find_dataset_file(folder, mode)
    data = torch.load(file_path, map_location="cpu")

    u = _reshape_to_3d_samples(data["u"], expected_channels=3, name="u")
    f = _reshape_to_3d_samples(data["f"], expected_channels=2, name="f")

    if u.shape[0] != f.shape[0]:
        raise ValueError(
            f"Sample count mismatch: u {tuple(u.shape)} vs f {tuple(f.shape)}"
        )

    extras = {
        "x": data.get("x", None),
        "y": data.get("y", None),
        "z": data.get("z", None),
        "t": data.get("t", None),
        "metadata": data.get("metadata", None),
        "raw_u_shape": tuple(data["u"].shape),
        "raw_f_shape": tuple(data["f"].shape),
        "file_path": file_path,
    }

    # Keep dataset tensors on CPU.
    # Move mini-batches to GPU inside the training/evaluation loops.
    return u.contiguous(), f.contiguous(), extras


# ============================================================
# 3. Model
# ============================================================


class U2GPEOperator(nn.Module):
    """
    3D periodic local LGNO-style operator.

    - periodic 7-point stencil for Re(psi), Im(psi)
      (center, +/-x, +/-y, +/-z)
    - V*psi is added explicitly:
          Re(V psi) = V * Re(psi)
          Im(V psi) = V * Im(psi)
    - network only learns the remaining local operator part,
      which is expected to approximate the discrete Laplacian part
    - input to network:
          7 * Re(psi) + 7 * Im(psi) = 14
    - output channels = 2 -> [Re(part), Im(part)]
    """

    def __init__(self, out_channels=2, hidden=16):
        super().__init__()

        self.out_channels = out_channels
        assert (
            out_channels == 2
        ), "For Schrodinger operator learning, out_channels should be 2."

        self.local_dim = 14

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

    def _make_7point_patch(self, x):
        """
        x: [B,1,Nx,Ny,Nz]
        return: [B,7,Nx,Ny,Nz]
        """
        offsets = [
            (0, 0, 0),
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        return torch.cat(
            [
                torch.roll(x, shifts=(dx, dy, dz), dims=(2, 3, 4))
                for dx, dy, dz in offsets
            ],
            dim=1,
        )

    def forward(self, u):
        """
        u: [B,3,Nx,Ny,Nz]
           channel 0 = Re(psi)
           channel 1 = Im(psi)
           channel 2 = V

        return: [B,2,Nx,Ny,Nz]
                channel 0 = Re(Hpsi)
                channel 1 = Im(Hpsi)
        """
        B, C, Nx, Ny, Nz = u.shape
        assert C == 3, f"Expected input channels = 3, got {C}"

        psi_re = u[:, 0:1]
        psi_im = u[:, 1:2]
        Vc = u[:, 2:3]

        re_patch = self._make_7point_patch(psi_re)  # [B,7,Nx,Ny,Nz]
        im_patch = self._make_7point_patch(psi_im)  # [B,7,Nx,Ny,Nz]

        local = torch.cat([re_patch, im_patch], dim=1)  # [B,14,Nx,Ny,Nz]

        local_flat = local.permute(0, 2, 3, 4, 1).reshape(
            -1, self.local_dim
        )  # [B*Nx*Ny*Nz,14]

        weights = self.net(local_flat)
        weights = weights.view(
            -1, self.out_channels, self.local_dim
        )  # [B*Nx*Ny*Nz,2,14]

        out = torch.sum(weights * local_flat.unsqueeze(1), dim=2)  # [B*Nx*Ny*Nz,2]

        pred_local = (
            out.view(B, Nx, Ny, Nz, self.out_channels)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

        Vpsi = torch.cat([Vc * psi_re, Vc * psi_im], dim=1)
        pred = pred_local + Vpsi

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
    """
    Save one field image only, no ticks, no colorbar.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(
        field2d,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_fig_all_formats(fig, save_stem)


def save_standalone_colorbar(
    save_stem, cmap, vmin, vmax, center_zero=False, label=None, orientation="vertical"
):
    """
    Save one standalone vertical colorbar.
    Only change:
      - for small/large values, show exponent at the top-right
        instead of long decimals on ticks.
    Everything else stays the same.
    """
    if orientation == "vertical":
        fig, ax = plt.subplots(figsize=(0.35, 4.8))
    else:
        fig, ax = plt.subplots(figsize=(4.8, 0.4))

    if center_zero:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=vmin, vmax=vmax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    orientation = "horizontal"

    if orientation == "vertical":
        cbar = plt.colorbar(sm, cax=ax)
    else:
        cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")

    # ===== only this display detail is changed =====
    from matplotlib.ticker import FuncFormatter

    max_abs = max(abs(vmin), abs(vmax))
    use_sci = (max_abs > 0) and (max_abs < 1e-1 or max_abs >= 1e2)

    if use_sci:
        exponent = int(np.floor(np.log10(max_abs)))
        scale = 10.0**exponent

        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x / scale:g}")
        )
        cbar.ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x / scale:g}")
        )

        # if orientation == "vertical":
        #     cbar.ax.text(1.05, 1.02, rf"$\times 10^{{{exponent}}}$",
        #                  transform=cbar.ax.transAxes,
        #                  ha="left", va="bottom", fontsize=fontsize)
        # else:
        #     cbar.ax.text(0.2, -1.5, rf"$\times 10^{{{exponent}}}$",
        #                  transform=cbar.ax.transAxes,
        #                  ha="left", va="top", fontsize=fontsize)

    if label is not None:
        cbar.set_label(label)

    fig.savefig(save_stem + ".pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(save_stem + ".svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(save_stem + ".png", dpi=400, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


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
# 5. DataLoader helpers
# ============================================================


def make_loader(u, f, batch_size, shuffle):
    ds = TensorDataset(u, f)
    use_pin_memory = (
        torch.cuda.is_available() and u.device.type == "cpu" and f.device.type == "cpu"
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_pin_memory,
        drop_last=False,
    )


# ============================================================
# 6. Training
# ============================================================


def train(model, u_train, f_train, lr, epochs, patience, batch_size):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=500, threshold=1e-6, min_lr=1e-6
    )

    loss_fn = nn.MSELoss()
    train_loader = make_loader(u_train, f_train, batch_size=batch_size, shuffle=True)

    best_loss = 1e20
    wait = 0
    loss_history = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for u_batch, f_batch in train_loader:
            u_batch = u_batch.to(device, non_blocking=True)
            f_batch = f_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(u_batch)
            loss = loss_fn(pred, f_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = u_batch.shape[0]
            running_loss += loss.item() * bs
            n_seen += bs

        loss_val = running_loss / max(n_seen, 1)
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

        if epoch % 200 == 0:
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
# 7. Batched evaluation
# ============================================================


def predict_in_batches(model, u, batch_size):
    use_pin_memory = torch.cuda.is_available() and u.device.type == "cpu"
    loader = DataLoader(
        u,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    preds = []
    with torch.no_grad():
        for u_batch in loader:
            u_batch = u_batch.to(device, non_blocking=True)
            pred_batch = model(u_batch)
            preds.append(pred_batch.cpu())
    return torch.cat(preds, dim=0)


def compute_metrics(f_pred, f_true):
    diff = f_pred - f_true

    rel_err = torch.norm(diff) / (torch.norm(f_true) + 1e-20)
    mse_err = torch.mean(diff**2)

    re_rel_err = torch.norm(diff[:, 0:1]) / (torch.norm(f_true[:, 0:1]) + 1e-20)
    im_rel_err = torch.norm(diff[:, 1:2]) / (torch.norm(f_true[:, 1:2]) + 1e-20)

    re_mse_err = torch.mean(diff[:, 0:1] ** 2)
    im_mse_err = torch.mean(diff[:, 1:2] ** 2)

    return {
        "rel_err": rel_err,
        "mse_err": mse_err,
        "re_rel_err": re_rel_err,
        "im_rel_err": im_rel_err,
        "re_mse_err": re_mse_err,
        "im_mse_err": im_mse_err,
    }


# ============================================================
# 8. Test visualization
# ============================================================


def save_all_test_images(f_pred_test, f_test, z_mid=None):
    """
    For each sample and each output channel:
    - use the z-mid slice of the 3D field
    - normalize by truth max of that channel
    - predict / truth share one color scale
    - error uses its own symmetric scale

    f_pred_test, f_test: [B,2,Nx,Ny,Nz]
    """
    f_pred_np = f_pred_test.detach().cpu().numpy()
    f_true_np = f_test.detach().cpu().numpy()

    B, C, Nx, Ny, Nz = f_pred_np.shape
    assert C == 2, f"Expected 2 output channels, got {C}"

    if z_mid is None:
        z_mid = Nz // 2

    channel_names = ["reHpsi", "imHpsi"]
    channel_labels = [r"$\Re(H\psi)$", r"$\Im(H\psi)$"]
    channel_cmaps = ["Reds", "Blues"]

    amp_dir = os.path.join(fig_dir, "amplitude")
    os.makedirs(amp_dir, exist_ok=True)

    for b in range(B):
        for c in range(C):
            if b % 500 == 0 or b == 4999:
                pred_img = f_pred_np[b, c, :, :, z_mid]
                true_img = f_true_np[b, c, :, :, z_mid]
                err_img = pred_img - true_img

                vmax = np.max(np.abs(true_img))
                if vmax == 0:
                    vmax = 1.0
                vmin = -vmax

                err_abs = np.max(np.abs(err_img))
                if err_abs == 0:
                    err_abs = 1.0

                ch_name = channel_names[c]
                ch_label = channel_labels[c]

                pred_name = os.path.join(pred_dir, f"sample{b:03d}_{ch_name}_predict")
                true_name = os.path.join(truth_dir, f"sample{b:03d}_{ch_name}_truth")
                err_name = os.path.join(error_dir, f"sample{b:03d}_{ch_name}_error")

                save_single_field_image(
                    field2d=pred_img,
                    save_stem=pred_name,
                    cmap=channel_cmaps[c],
                    vmin=vmin,
                    vmax=vmax,
                )

                save_single_field_image(
                    field2d=true_img,
                    save_stem=true_name,
                    cmap=channel_cmaps[c],
                    vmin=vmin,
                    vmax=vmax,
                )

                save_single_field_image(
                    field2d=err_img,
                    save_stem=err_name,
                    cmap=channel_cmaps[c],
                    vmin=-err_abs,
                    vmax=err_abs,
                )

                save_standalone_colorbar(
                    save_stem=os.path.join(
                        colorbar_dir, f"sample{b:03d}_{ch_name}_field_colorbar"
                    ),
                    cmap=channel_cmaps[c],
                    vmin=vmin,
                    vmax=vmax,
                    center_zero=True,
                    # label=ch_label
                )

                save_standalone_colorbar(
                    save_stem=os.path.join(
                        colorbar_dir, f"sample{b:03d}_{ch_name}_error_colorbar"
                    ),
                    cmap="seismic",
                    vmin=-err_abs,
                    vmax=err_abs,
                    center_zero=True,
                )
        # ===============================
        # Amplitude |Hpsi|
        # ===============================
        if b % 500 == 0 or b == 4999:
            pred_amp = np.sqrt(
                f_pred_np[b, 0, :, :, z_mid] ** 2 + f_pred_np[b, 1, :, :, z_mid] ** 2
            )

            true_amp = np.sqrt(
                f_true_np[b, 0, :, :, z_mid] ** 2 + f_true_np[b, 1, :, :, z_mid] ** 2
            )

            err_amp = pred_amp - true_amp

            vmax = np.max(true_amp)
            vmax = 4

            if vmax == 0:
                vmax = 1.0
            vmin = 0.0  # amplitude is usually non-negative

            err_abs = np.max(np.abs(err_amp))
            if err_abs == 0:
                err_abs = 1.0

            # ---------- save ----------
            save_single_field_image(
                field2d=pred_amp,
                save_stem=os.path.join(amp_dir, f"sample{b:03d}_amp_predict"),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            save_single_field_image(
                field2d=true_amp,
                save_stem=os.path.join(amp_dir, f"sample{b:03d}_amp_truth"),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            save_single_field_image(
                field2d=err_amp,
                save_stem=os.path.join(amp_dir, f"sample{b:03d}_amp_error"),
                cmap="seismic",
                vmin=-err_abs,
                vmax=err_abs,
            )

            # ---------- colorbar ----------
            save_standalone_colorbar(
                save_stem=os.path.join(colorbar_dir, f"sample{b:03d}_amp_colorbar"),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                center_zero=False,
            )

            save_standalone_colorbar(
                save_stem=os.path.join(
                    colorbar_dir, f"sample{b:03d}_amp_error_colorbar"
                ),
                cmap="seismic",
                vmin=-err_abs,
                vmax=err_abs,
                center_zero=True,
                orientation="horizontal",
            )

    log_print(f"Per-sample normalized figures saved from z-mid slice = {z_mid}.")


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--train_dir", type=str, default="dataset/train_data")
    parser.add_argument("--test_dir", type=str, default="dataset/test_data")

    args = parser.parse_args()

    with open(os.path.join(base_dir, "config.txt"), "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    u_train, f_train, train_extras = load_dataset(args.train_dir, "train")
    u_test, f_test, test_extras = load_dataset(args.test_dir, "test")

    log_print(f"Train raw u shape: {train_extras['raw_u_shape']}")
    log_print(f"Train raw f shape: {train_extras['raw_f_shape']}")
    log_print(f"Test  raw u shape: {test_extras['raw_u_shape']}")
    log_print(f"Test  raw f shape: {test_extras['raw_f_shape']}")

    log_print(f"Train u shape: {tuple(u_train.shape)}")
    log_print(f"Train f shape: {tuple(f_train.shape)}")
    log_print(f"Test  u shape: {tuple(u_test.shape)}")
    log_print(f"Test  f shape: {tuple(f_test.shape)}")

    if test_extras["t"] is not None:
        log_print(f"Test time steps detected: {len(test_extras['t'])}")
    if test_extras["z"] is not None:
        log_print(f"Test z grid size: {len(test_extras['z'])}")

    model = U2GPEOperator(out_channels=2, hidden=args.hidden).to(device)

    log_print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    train_time, loss_history = train(
        model, u_train, f_train, args.lr, args.epochs, args.patience, args.batch_size
    )

    # ============================================================
    # Test
    # ============================================================

    model.load_state_dict(
        torch.load(os.path.join(base_dir, "best_model.pth"), map_location=device)
    )
    model.eval()

    with torch.no_grad():
        f_pred_test = predict_in_batches(model, u_test, batch_size=args.test_batch_size)
        f_true_cpu = f_test.detach().cpu()

        metrics = compute_metrics(f_pred_test, f_true_cpu)

        log_print(f"Test Relative L2 error: {metrics['rel_err'].item():.6e}")
        log_print(f"Test MSE error        : {metrics['mse_err'].item():.6e}")
        log_print(f"Re(Hpsi) Relative L2  : {metrics['re_rel_err'].item():.6e}")
        log_print(f"Im(Hpsi) Relative L2  : {metrics['im_rel_err'].item():.6e}")
        log_print(f"Re(Hpsi) MSE          : {metrics['re_mse_err'].item():.6e}")
        log_print(f"Im(Hpsi) MSE          : {metrics['im_mse_err'].item():.6e}")

        torch.save(f_pred_test.cpu(), os.path.join(base_dir, "test_f_pred.pt"))
        torch.save(f_true_cpu.cpu(), os.path.join(base_dir, "test_f_true.pt"))

        z_mid = None
        if test_extras["z"] is not None:
            z_mid = len(test_extras["z"]) // 2
        save_all_test_images(f_pred_test, f_true_cpu, z_mid=z_mid)

    log_print("All results saved successfully.")
