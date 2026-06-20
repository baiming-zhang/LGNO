# ============================================================
# 2D Poisson / static operator learning
# Learn: u -> f
# Periodic-learning full version
# ============================================================

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import TwoSlopeNorm

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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = os.path.join("LGNO_8hidden_results", f"run_{timestamp}")
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

def load_dataset(folder, mode="train"):
    """
    Static periodic dataset version.

    Expected file:
        folder/train.pt or folder/test.pt

    Expected tensor shapes:
        u: [B,1,H,W] or [B,H,W] or [H,W]
        f: [B,1,H,W] or [B,H,W] or [H,W]
    """
    file_path = os.path.join(folder, f"{mode}.pt")
    data = torch.load(file_path, map_location="cpu")

    u = data["u"]
    f = data["f"]

    # -> [B,1,H,W]
    if u.dim() == 4:
        # already [B,1,H,W] or [B,C,H,W]
        if u.shape[1] != 1:
            raise ValueError(f"Expected channel dimension = 1, got u.shape = {u.shape}")
    elif u.dim() == 3:
        # [B,H,W] -> [B,1,H,W]
        u = u.unsqueeze(1)
        f = f.unsqueeze(1)
    elif u.dim() == 2:
        # [H,W] -> [1,1,H,W]
        u = u.unsqueeze(0).unsqueeze(0)
        f = f.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported tensor shape: u.shape = {u.shape}")

    return u.to(device), f.to(device)

# ============================================================
# 3. Model
# ============================================================

class U2LapU(nn.Module):
    """
    LGNO modified: 3x3 periodic stencil
    input local 3x3 patch of u -> predict target operator f
    """
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 9)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u):
        """
        u: [B,1,H,W]
        return: [B,1,H,W]

        Periodic learning is implemented by torch.roll on spatial dims.
        """
        B, _, H, W = u.shape

        stencil_list = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                shifted = torch.roll(u, shifts=(dy, dx), dims=(2, 3))
                stencil_list.append(shifted)

        stencil = torch.cat(stencil_list, dim=1)              # [B,9,H,W]
        local = stencil.permute(0, 2, 3, 1).reshape(-1, 9)    # [B*H*W, 9]
        weights = self.net(local)                              # [B*H*W, 9]
        out = torch.sum(weights * local, dim=1, keepdim=True) # [B*H*W, 1]
        pred = out.view(B, 1, H, W)

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
        aspect="equal"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_fig_all_formats(fig, save_stem)

def save_standalone_colorbar(save_stem, cmap, vmin, vmax, center_zero=False, label=None):
    """
    Save one standalone vertical colorbar.
    """
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
# ============================================================

def train(model, u_train, f_train, lr, epochs, patience):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=500,
        threshold=1e-6,
        min_lr=1e-6
    )

    loss_fn = nn.MSELoss()

    best_loss = 1e20
    wait = 0
    loss_history = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(u_train)
        loss = loss_fn(pred, f_train)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
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

    # save original numpy
    np.save(os.path.join(base_dir, "loss_history.npy"), loss_history)

    # save txt
    np.savetxt(
        os.path.join(base_dir, "loss_total.txt"),
        loss_history,
        fmt="%.12e"
    )

    # save loss figure
    save_loss_curve(loss_history, os.path.join(loss_dir, "training_loss"))

    return total_time, loss_history

# ============================================================
# 6. Test visualization
# ============================================================

def save_all_test_images(f_pred_test, f_test):
    """
    每个 sample 单独归一：
    - 用 truth 的 max 做归一
    - predict / truth 共用同一色卡
    - error 单独对称归一
    """

    f_pred_np = f_pred_test.detach().cpu().numpy()   # [B,1,H,W]
    f_true_np = f_test.detach().cpu().numpy()

    B, _, H, W = f_pred_np.shape

    for b in range(B):
        pred_img = f_pred_np[b, 0]
        true_img = f_true_np[b, 0]
        err_img  = pred_img - true_img

        # ====================================================
        # ⭐ 用 truth 做归一（关键）
        # ====================================================
        vmax = np.max(np.abs(true_img))
        if vmax == 0:
            vmax = 1.0
        vmax = 100
        vmin = -vmax

        # error 单独归一
        err_abs = np.max(np.abs(err_img))
        if err_abs == 0:
            err_abs = 1.0

        # ====================================================
        # 保存图
        # ====================================================
        pred_name = os.path.join(pred_dir,  f"sample{b:03d}_predict")
        true_name = os.path.join(truth_dir, f"sample{b:03d}_truth")
        err_name  = os.path.join(error_dir, f"sample{b:03d}_error")

        save_single_field_image(
            field2d=pred_img,
            save_stem=pred_name,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax
        )

        save_single_field_image(
            field2d=true_img,
            save_stem=true_name,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax
        )

        save_single_field_image(
            field2d=err_img,
            save_stem=err_name,
            cmap="RdBu_r",
            vmin=-err_abs,
            vmax=err_abs
        )

        # ====================================================
        # ⭐ 每个 sample 单独 colorbar
        # ====================================================
        save_standalone_colorbar(
            save_stem=os.path.join(colorbar_dir, f"sample{b:03d}_field_colorbar"),
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            center_zero=True,
            label=r"$f$"
        )

        save_standalone_colorbar(
            save_stem=os.path.join(colorbar_dir, f"sample{b:03d}_error_colorbar"),
            cmap="seismic",
            vmin=-err_abs,
            vmax=err_abs,
            center_zero=True,
            label="Error"
        )

    log_print("Per-sample normalized figures saved.")

# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--train_dir", type=str, default="dataset/train_data")
    parser.add_argument("--test_dir", type=str, default="dataset/test_data")

    args = parser.parse_args()

    with open(os.path.join(base_dir, "config.txt"), "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    u_train, f_train = load_dataset(args.train_dir, "train")
    u_test,  f_test  = load_dataset(args.test_dir,  "test")

    log_print(f"Train u shape: {tuple(u_train.shape)}")
    log_print(f"Train f shape: {tuple(f_train.shape)}")
    log_print(f"Test  u shape: {tuple(u_test.shape)}")
    log_print(f"Test  f shape: {tuple(f_test.shape)}")

    model = U2LapU(hidden=args.hidden).to(device)

    log_print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    train_time, loss_history = train(
        model, u_train, f_train,
        args.lr, args.epochs, args.patience
    )

    # ============================================================
    # Test
    # ============================================================

    model.load_state_dict(torch.load(os.path.join(base_dir, "best_model.pth"), map_location=device))
    model.eval()

    with torch.no_grad():
        f_pred_test = model(u_test)

        rel_err = torch.norm(f_pred_test - f_test) / (torch.norm(f_test) + 1e-20)
        mse_err = torch.mean((f_pred_test - f_test) ** 2)

        log_print(f"Test Relative L2 error: {rel_err.item():.6e}")
        log_print(f"Test MSE error        : {mse_err.item():.6e}")

        # save tensors
        torch.save(f_pred_test.cpu(), os.path.join(base_dir, "test_f_pred.pt"))
        torch.save(f_test.cpu(),      os.path.join(base_dir, "test_f_true.pt"))

        # save all figures
        save_all_test_images(f_pred_test, f_test)

    log_print("All results saved successfully.")
