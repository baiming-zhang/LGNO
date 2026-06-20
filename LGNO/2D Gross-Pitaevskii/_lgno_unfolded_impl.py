# ============================================================
# 2D GPE-like / fully learned local operator learning
# LGNO without explicit V prior; V is used only as an MLP input feature
#
# Task:
#   Learn u -> f from data without adding or folding any exact V*psi prior.
#
# Input from dataset:
#   u = [Re(psi), Im(psi), V]          -> [B,3,H,W]
# Output from dataset:
#   f = [Re(Hpsi), Im(Hpsi)]           -> [B,2,H,W]
#
# LGNO setting used here:
#   1. No explicit V*psi term is added in forward().
#   2. V is provided directly to the MLP as a normalized local conditioning feature.
#   3. The MLP learns the full normalized stencil-weight vector.
#   4. The dot-product variables are std-scaled instead of raw physical fields,
#      so the MLP no longer has to learn huge finite-difference weights.
#   5. |psi|^2 is allowed as an auxiliary local feature.
#   6. The model follows:
#          local stencil features including V -> MLP-generated weights
#          -> local coefficient-field dot product
#
# Weight-generation feature channels:
#   Re branch: [stencil(Re(psi)_norm), V_norm_center, |psi|^2_norm_center]
#   Im branch: [stencil(Im(psi)_norm), V_norm_center, |psi|^2_norm_center]
#
# Dot-product variables:
#   Re branch: stencil(Re(psi)_raw / Re_std)
#   Im branch: stencil(Im(psi)_raw / Im_std)
#
# Prediction is made in normalized output space:
#   f_norm = learned local dot product - f_mean / f_std.
# Final reported errors are computed after inverse-normalization.
# ============================================================

import os
import time
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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
plt.rcParams["svg.fonttype"] = "none"


# ============================================================
# 1. Dataset loading
# ============================================================

def _find_dataset_file(folder, mode):
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


def _ensure_4d_tensor(t, expected_channels, name):
    if t.dim() == 4:
        if t.shape[1] != expected_channels:
            raise ValueError(
                f"{name} expected channel dimension = {expected_channels}, "
                f"got shape = {tuple(t.shape)}"
            )
        return t.float()

    if t.dim() == 3:
        if t.shape[0] != expected_channels:
            raise ValueError(
                f"{name} expected [C,H,W] with C={expected_channels}, "
                f"got shape = {tuple(t.shape)}"
            )
        return t.unsqueeze(0).float()

    raise ValueError(f"Unsupported tensor shape for {name}: {tuple(t.shape)}")


def load_dataset(folder, mode="train"):
    file_path = _find_dataset_file(folder, mode)
    data = torch.load(file_path, map_location="cpu")

    u = _ensure_4d_tensor(data["u"], expected_channels=3, name="u")
    f = _ensure_4d_tensor(data["f"], expected_channels=2, name="f")

    labels = data.get("labels", None)
    metadata = data.get("metadata", {})

    return u.to(device), f.to(device), labels, metadata, file_path


# ============================================================
# 2. Normalization utilities
# ============================================================

def channel_mean_std(x, eps=1e-8):
    """
    x: [B,C,H,W]
    Return mean/std with shape [1,C,1,1].
    """
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(eps)
    return mean, std


def compute_train_stats(u_train, f_train, eps=1e-8):
    """
    Statistics are computed from training data only.
    No governing-equation information is used.
    """
    u_mean, u_std = channel_mean_std(u_train, eps=eps)
    f_mean, f_std = channel_mean_std(f_train, eps=eps)

    psi_re = u_train[:, 0:1]
    psi_im = u_train[:, 1:2]
    amp2 = psi_re ** 2 + psi_im ** 2
    amp2_mean, amp2_std = channel_mean_std(amp2, eps=eps)

    return {
        "u_mean": u_mean,
        "u_std": u_std,
        "f_mean": f_mean,
        "f_std": f_std,
        "amp2_mean": amp2_mean,
        "amp2_std": amp2_std,
    }


# ============================================================
# 3. Fully learned LGNO model
# ============================================================

class NormalizedResidualLGNOAmp2(nn.Module):
    """
    LGNO variant for the generated GPE-like data without any explicit V prior.

    The model does not add V*psi separately and does not fold V into the center
    stencil coefficient by hand. Instead, V is normalized and passed to the MLP
    as a local conditioning feature. The MLP must learn the full local stencil
    coefficient vector from data.

    For q = Re(psi) or Im(psi):

        feature_q = [stencil(q_norm), V_norm_center, |psi|^2_norm_center]
        alpha_q = shared_base + SharedMLP(feature_q)
        f_q_norm = sum_k alpha_q,k * (q_raw,k / q_std) - f_mean_q / f_std_q

    After inverse normalization, the physical prediction is

        f_q = f_std_q * sum_k alpha_q,k * (q_raw,k / q_std),

    so the output remains in local coefficient-times-field form and contains no
    additive physical bias. Any V-dependent coefficient must be learned by the
    MLP through the V input feature.
    """

    def __init__(
        self,
        stats,
        out_channels=2,
        hidden=64,
        depth=3,
        stencil="5pt",
        dropout=0.0,
        last_scale=1e-3,
        variant="folded",
    ):
        super().__init__()

        assert out_channels == 2, "For this dataset, out_channels should be 2."
        assert stencil in ["5pt", "9pt"], "stencil must be '5pt' or '9pt'."
        assert depth >= 2, "depth should be at least 2."
        assert variant == "unfolded", "LGNO_unfolded.py only runs the LGNO(unfolded) baseline."

        self.out_channels = out_channels
        self.stencil = stencil
        self.variant = variant

        if stencil == "5pt":
            self.offsets = [
                (0, 0),
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ]
        else:
            self.offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),  (0, 0),  (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]

        self.num_points = len(self.offsets)
        self.center_index = self.offsets.index((0, 0))

        # ------------------------------------------------------------
        # Dimensions
        # ------------------------------------------------------------
        # Dot-product part for each branch:
        #   one scalar field stencil only, either Re or Im.
        #   5pt: 5
        #   9pt: 9
        self.dot_dim = self.num_points

        # Folded MLP input:
        #   q_norm stencil + V_norm center + |psi|^2_norm center
        # Unfolded MLP input:
        #   Re_norm stencil + Im_norm stencil + V_norm center + |psi|^2_norm center
        self.weight_input_dim = self.num_points + 2 if self.variant == "folded" else 2 * self.num_points + 2

        # Keep local_dim name for compatibility with the original interface.
        self.local_dim = self.weight_input_dim

        # Register normalization statistics as buffers, so they are saved in state_dict.
        self.register_buffer("u_mean", stats["u_mean"].detach().clone())
        self.register_buffer("u_std", stats["u_std"].detach().clone())
        self.register_buffer("f_mean", stats["f_mean"].detach().clone())
        self.register_buffer("f_std", stats["f_std"].detach().clone())
        self.register_buffer("amp2_mean", stats["amp2_mean"].detach().clone())
        self.register_buffer("amp2_std", stats["amp2_std"].detach().clone())

        def make_weight_net(out_dim):
            layers = []
            in_dim = self.weight_input_dim

            for _ in range(depth - 1):
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.SiLU())
                layers.append(nn.LayerNorm(hidden))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden

            layers.append(nn.Linear(in_dim, out_dim))
            return nn.Sequential(*layers)

        if self.variant == "folded":
            # Folded: Re and Im branches share one weight MLP and one base stencil.
            self.net = make_weight_net(self.num_points)
            self.weight_base = nn.Parameter(torch.zeros(1, self.num_points, 1, 1))
        else:
            # Unfolded: one joint MLP sees Re and Im stencils together and
            # predicts independent Re/Im stencil weights.
            self.net = make_weight_net(2 * self.num_points)
            self.weight_base = nn.Parameter(torch.zeros(1, 2 * self.num_points, 1, 1))
        self.reset_parameters(last_scale=last_scale)

    def reset_parameters(self, last_scale=1e-3):
        def reset_net(net):
            linears = [m for m in net.modules() if isinstance(m, nn.Linear)]

            for m in linears[:-1]:
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

            # Small final-layer initialization keeps the initial learned weights small.
            # No exact V contribution is added or folded into the coefficients.
            last = linears[-1]
            nn.init.normal_(last.weight, mean=0.0, std=last_scale)
            nn.init.zeros_(last.bias)

        reset_net(self.net)
        nn.init.zeros_(self.weight_base)

    def normalize_u(self, u):
        return (u - self.u_mean) / self.u_std

    def normalize_amp2(self, amp2):
        return (amp2 - self.amp2_mean) / self.amp2_std

    def normalize_f(self, f):
        return (f - self.f_mean) / self.f_std

    def denormalize_f(self, f_norm):
        return f_norm * self.f_std + self.f_mean

    def scale_q(self, u, branch="re"):
        """
        Std-scale q without subtracting its mean.

        Using q_raw / q_std preserves constant components while still shrinking
        the effective learned stencil coefficients by q_std / f_std.
        """
        if branch == "re":
            return u[:, 0:1] / self.u_std[:, 0:1]
        if branch == "im":
            return u[:, 1:2] / self.u_std[:, 1:2]
        raise ValueError(f"branch must be 're' or 'im', got {branch}")

    def periodic_patch(self, x):
        """
        x: [B,1,H,W]
        return: [B,num_points,H,W]
        """
        return torch.cat(
            [torch.roll(x, shifts=(dy, dx), dims=(2, 3)) for dy, dx in self.offsets],
            dim=1,
        )

    def build_branch_feature(self, u, branch="re"):
        """
        Build the shared-MLP input for one scalar branch.

        branch='re': feature = [stencil(Re_norm), V_norm_center, |psi|^2_norm_center]
        branch='im': feature = [stencil(Im_norm), V_norm_center, |psi|^2_norm_center]

        V is used only as a normalized MLP input feature. No explicit V*psi
        contribution is added in forward().
        """
        if u.dim() != 4 or u.shape[1] != 3:
            raise ValueError(f"Expected u shape [B,3,H,W], got {tuple(u.shape)}")
        if branch not in ["re", "im"]:
            raise ValueError(f"branch must be 're' or 'im', got {branch}")

        psi_re_raw = u[:, 0:1]
        psi_im_raw = u[:, 1:2]
        amp2_raw = psi_re_raw ** 2 + psi_im_raw ** 2

        u_norm = self.normalize_u(u)
        amp2_norm = self.normalize_amp2(amp2_raw)

        q_norm = u_norm[:, 0:1] if branch == "re" else u_norm[:, 1:2]
        v_norm = u_norm[:, 2:3]
        q_patch_norm = self.periodic_patch(q_norm)

        local_feature = torch.cat(
            [q_patch_norm, v_norm, amp2_norm],
            dim=1,
        )
        return local_feature

    def build_local_feature(self, u):
        """
        Compatibility function used by diagnostics.
        Returns the active MLP feature layout.
        """
        if self.variant == "unfolded":
            return self.build_unfolded_feature(u)
        return self.build_branch_feature(u, branch="re")

    def build_unfolded_feature(self, u):
        """
        Build the unfolded MLP input with real and imaginary stencils together:
            [stencil(Re_norm), stencil(Im_norm), V_norm_center, |psi|^2_norm_center]
        """
        if u.dim() != 4 or u.shape[1] != 3:
            raise ValueError(f"Expected u shape [B,3,H,W], got {tuple(u.shape)}")

        psi_re_raw = u[:, 0:1]
        psi_im_raw = u[:, 1:2]
        amp2_raw = psi_re_raw ** 2 + psi_im_raw ** 2

        u_norm = self.normalize_u(u)
        re_patch_norm = self.periodic_patch(u_norm[:, 0:1])
        im_patch_norm = self.periodic_patch(u_norm[:, 1:2])
        v_norm = u_norm[:, 2:3]
        amp2_norm = self.normalize_amp2(amp2_raw)

        return torch.cat([re_patch_norm, im_patch_norm, v_norm, amp2_norm], dim=1)

    def build_psi_local(self, u):
        """
        Compatibility function.
        Return normalized Re and Im stencil variables concatenated:
            [Re_norm stencil, Im_norm stencil]
        This is not used in the new forward pass, but keeps old diagnostics safer.
        """
        if u.dim() != 4 or u.shape[1] != 3:
            raise ValueError(f"Expected u shape [B,3,H,W], got {tuple(u.shape)}")

        u_norm = self.normalize_u(u)
        re_patch = self.periodic_patch(u_norm[:, 0:1])
        im_patch = self.periodic_patch(u_norm[:, 1:2])
        return torch.cat([re_patch, im_patch], dim=1)

    def predict_branch_weights(self, u, branch="re"):
        """
        Predict full normalized stencil weights for one branch.

        Return:
            weights: [B,num_points,H,W]
        """
        B, C, H, W = u.shape
        assert C == 3, f"Expected input channels = 3, got {C}"

        if self.variant == "folded":
            feature = self.build_branch_feature(u, branch=branch)
            feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, self.weight_input_dim)
            learned_weights = self.net(feature_flat)
            learned_weights = learned_weights.view(B, H, W, self.num_points)
            learned_weights = learned_weights.permute(0, 3, 1, 2).contiguous()
            return learned_weights + self.weight_base

        all_weights = self.predict_unfolded_weights(u)
        if branch == "re":
            return all_weights[:, :self.num_points]
        if branch == "im":
            return all_weights[:, self.num_points:]
        raise ValueError(f"branch must be 're' or 'im', got {branch}")

    def predict_unfolded_weights(self, u):
        """
        Predict unfolded Re/Im stencil weights from the joint Re+Im feature.

        Return:
            weights: [B,2*num_points,H,W]
        """
        B, C, H, W = u.shape
        assert C == 3, f"Expected input channels = 3, got {C}"
        if self.variant != "unfolded":
            raise RuntimeError("predict_unfolded_weights is only available for variant='unfolded'.")

        feature = self.build_unfolded_feature(u)
        feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, self.weight_input_dim)
        learned_weights = self.net(feature_flat)
        learned_weights = learned_weights.view(B, H, W, 2 * self.num_points)
        learned_weights = learned_weights.permute(0, 3, 1, 2).contiguous()
        return learned_weights + self.weight_base

    def forward(self, u):
        """
        Return normalized prediction of f.

        Input:
            u: [B,3,H,W]
               channel 0 = Re(psi)
               channel 1 = Im(psi)
               channel 2 = V

        Output:
            out_norm: [B,2,H,W]
                      normalized [Re(Hpsi), Im(Hpsi)]

        Structure:
            Re branch:
                [Re_norm stencil, V_norm center, |psi|^2_norm center]
                -> shared MLP -> full stencil weights
                output_norm = weights dot (Re_raw / Re_std) stencil - f_mean_Re / f_std_Re

            Im branch:
                same construction with Im.

        No exact V*psi term is added, and no V contribution is manually folded
        into the center coefficient. V can affect the output only through the
        MLP-generated weights.
        """
        B, C, H, W = u.shape
        assert C == 3, f"Expected input channels = 3, got {C}"

        re_patch_scaled = self.periodic_patch(self.scale_q(u, branch="re"))
        im_patch_scaled = self.periodic_patch(self.scale_q(u, branch="im"))

        # MLP-generated full normalized stencil weights.
        # V is included in build_branch_feature(), not added explicitly here.
        w_re = self.predict_branch_weights(u, branch="re")
        w_im = self.predict_branch_weights(u, branch="im")

        pred_re_norm = torch.sum(w_re * re_patch_scaled, dim=1, keepdim=True)
        pred_im_norm = torch.sum(w_im * im_patch_scaled, dim=1, keepdim=True)

        # Output centering for normalized-space training. In physical space this
        # cancels exactly after denormalize_f(), so no physical bias is added.
        pred_re_norm = pred_re_norm - self.f_mean[:, 0:1] / self.f_std[:, 0:1]
        pred_im_norm = pred_im_norm - self.f_mean[:, 1:2] / self.f_std[:, 1:2]

        return torch.cat([pred_re_norm, pred_im_norm], dim=1)

    @torch.no_grad()
    def predict_physical(self, u):
        f_norm = self.forward(u)
        return self.denormalize_f(f_norm)



# ============================================================
# 4. Output directories and logging
# ============================================================

def create_output_dirs(root="LGNO_no_V_prior_result"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(root, f"run_{timestamp}")

    dirs = {
        "base": base_dir,
        "fig": os.path.join(base_dir, "figures"),
        "pred": os.path.join(base_dir, "figures", "predict"),
        "truth": os.path.join(base_dir, "figures", "truth"),
        "error": os.path.join(base_dir, "figures", "error"),
        "colorbar": os.path.join(base_dir, "figures", "colorbar"),
        "loss": os.path.join(base_dir, "figures", "loss"),
        "weights": os.path.join(base_dir, "figures", "weights"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def __call__(self, msg):
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ============================================================
# 5. Figure helpers
# ============================================================

def save_fig_all_formats(fig, save_stem):
    fig.savefig(save_stem + ".pdf")
    fig.savefig(save_stem + ".svg")
    fig.savefig(save_stem + ".png", dpi=400)
    plt.close(fig)


def save_single_field_image(field2d, save_stem, cmap, vmin, vmax):
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


def save_standalone_colorbar(save_stem, cmap, vmin, vmax, center_zero=False, label=None):
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


def save_all_test_images(f_pred_test, f_test, dirs, labels=None):
    f_pred_np = f_pred_test.detach().cpu().numpy()
    f_true_np = f_test.detach().cpu().numpy()

    B, C, H, W = f_pred_np.shape
    assert C == 2, f"Expected 2 output channels, got {C}"

    channel_names = ["reHpsi", "imHpsi"]
    channel_labels = [r"$\Re(H\psi)$", r"$\Im(H\psi)$"]
    channel_cmaps = ["Reds", "Blues"]

    for b in range(B):
        label_suffix = ""
        if labels is not None and b < len(labels):
            label_suffix = f"_{labels[b]}"

        for c in range(C):
            pred_img = f_pred_np[b, c]
            true_img = f_true_np[b, c]
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

            pred_name = os.path.join(dirs["pred"],  f"sample{b:03d}{label_suffix}_{ch_name}_predict")
            true_name = os.path.join(dirs["truth"], f"sample{b:03d}{label_suffix}_{ch_name}_truth")
            err_name = os.path.join(dirs["error"], f"sample{b:03d}{label_suffix}_{ch_name}_error")

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
                cmap="seismic",
                vmin=-err_abs,
                vmax=err_abs,
            )

            save_standalone_colorbar(
                save_stem=os.path.join(dirs["colorbar"], f"sample{b:03d}{label_suffix}_{ch_name}_field_colorbar"),
                cmap=channel_cmaps[c],
                vmin=vmin,
                vmax=vmax,
                center_zero=True,
                label=ch_label,
            )

            save_standalone_colorbar(
                save_stem=os.path.join(dirs["colorbar"], f"sample{b:03d}{label_suffix}_{ch_name}_error_colorbar"),
                cmap="seismic",
                vmin=-err_abs,
                vmax=err_abs,
                center_zero=True,
                label=f"{ch_label} Error",
            )


def save_weight_maps(model, u, dirs, max_samples=1):
    """
    Optional diagnostic: save mean absolute local weight maps for Re and Im branches.
    This does not affect training.

    Current network convention:
        - Re branch: [Re_norm stencil, V_norm center, |psi|^2_norm center] -> shared MLP -> full weights
        - Im branch: [Im_norm stencil, V_norm center, |psi|^2_norm center] -> shared MLP -> full weights
        - No exact V contribution is added or folded in forward().
    """
    model.eval()
    with torch.no_grad():
        u = u[:max_samples]

        w_re = model.predict_branch_weights(u, branch="re")  # [B,num_points,H,W]
        w_im = model.predict_branch_weights(u, branch="im")  # [B,num_points,H,W]

        # Mean absolute weight over stencil dimension.
        weight_abs_mean = torch.stack(
            [w_re.abs().mean(dim=1), w_im.abs().mean(dim=1)],
            dim=1,
        )  # [B,2,H,W]

        names = ["re_branch", "im_branch"]
        for b in range(weight_abs_mean.shape[0]):
            for c in range(2):
                img = weight_abs_mean[b, c].detach().cpu().numpy()
                vmax = np.max(img)
                if vmax == 0:
                    vmax = 1.0
                save_single_field_image(
                    field2d=img,
                    save_stem=os.path.join(dirs["weights"], f"sample{b:03d}_{names[c]}_mean_abs_weight"),
                    cmap="viridis",
                    vmin=0.0,
                    vmax=vmax,
                )

        # Additional useful diagnostics: learned center weights.
        center_names = ["re_branch_center_learned_weight", "im_branch_center_learned_weight"]
        center_maps = torch.stack(
            [w_re[:, model.center_index], w_im[:, model.center_index]],
            dim=1,
        )  # [B,2,H,W]

        for b in range(center_maps.shape[0]):
            for c in range(2):
                img = center_maps[b, c].detach().cpu().numpy()
                vmax = np.max(np.abs(img))
                if vmax == 0:
                    vmax = 1.0
                save_single_field_image(
                    field2d=img,
                    save_stem=os.path.join(dirs["weights"], f"sample{b:03d}_{center_names[c]}"),
                    cmap="seismic",
                    vmin=-vmax,
                    vmax=vmax,
                )



# ============================================================
# 6. Training and evaluation
# ============================================================

def evaluate_physical(model, u, f):
    model.eval()
    with torch.no_grad():
        f_pred = model.predict_physical(u)

        rel_err = torch.norm(f_pred - f) / (torch.norm(f) + 1e-20)
        mse_err = torch.mean((f_pred - f) ** 2)

        re_rel_err = torch.norm(f_pred[:, 0:1] - f[:, 0:1]) / (torch.norm(f[:, 0:1]) + 1e-20)
        im_rel_err = torch.norm(f_pred[:, 1:2] - f[:, 1:2]) / (torch.norm(f[:, 1:2]) + 1e-20)

        re_mse_err = torch.mean((f_pred[:, 0:1] - f[:, 0:1]) ** 2)
        im_mse_err = torch.mean((f_pred[:, 1:2] - f[:, 1:2]) ** 2)

    return {
        "pred": f_pred,
        "rel_l2": rel_err.item(),
        "mse": mse_err.item(),
        "re_rel_l2": re_rel_err.item(),
        "im_rel_l2": im_rel_err.item(),
        "re_mse": re_mse_err.item(),
        "im_mse": im_mse_err.item(),
    }


def train(model, u_train, f_train, args, dirs, log_print):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        threshold=1e-7,
        min_lr=args.min_lr,
    )

    loss_fn = nn.MSELoss()

    f_train_norm = model.normalize_f(f_train).detach()

    best_loss = float("inf")
    wait = 0
    loss_history = []

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pred_norm = model(u_train)
        loss = loss_fn(pred_norm, f_train_norm)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_val = float(loss.item())
        scheduler.step(loss_val)
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            wait = 0
            torch.save(model.state_dict(), os.path.join(dirs["base"], "best_model.pth"))
        else:
            wait += 1

        if epoch % args.log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            log_print(
                f"[{epoch:6d}] "
                f"loss = {loss_val:.6e} | "
                f"lr = {current_lr:.3e}"
            )

        if wait > args.patience:
            log_print(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    log_print(f"Training time: {total_time:.2f} sec")
    log_print(f"Best normalized training loss: {best_loss:.6e}")

    loss_history = np.array(loss_history, dtype=np.float64)

    np.save(os.path.join(dirs["base"], "loss_history.npy"), loss_history)
    np.savetxt(os.path.join(dirs["base"], "loss_total.txt"), loss_history, fmt="%.12e")
    save_loss_curve(loss_history, os.path.join(dirs["loss"], "training_loss"))

    return total_time, loss_history


# ============================================================
# 7. Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="dataset/train_data")
    parser.add_argument("--test_dir", type=str, default="dataset/test_data")
    parser.add_argument("--out_dir", type=str, default="LGNO_unfolded_12hidden_results")

    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--stencil", type=str, default="5pt", choices=["5pt", "9pt"])
    parser.add_argument("--variant", type=str, default="unfolded", choices=["unfolded"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--last_scale", type=float, default=1e-3)

    parser.add_argument("--epochs", type=int, default=2000000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10000)
    parser.add_argument("--scheduler_patience", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=200)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_weights", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dirs = create_output_dirs(args.out_dir)
    log_file = os.path.join(dirs["base"], "train_log.txt")
    log_print = Logger(log_file)

    log_print(f"Device: {device}")
    log_print(f"Result directory: {dirs['base']}")

    with open(os.path.join(dirs["base"], "config.txt"), "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    u_train, f_train, train_labels, train_metadata, train_path = load_dataset(args.train_dir, "train")
    u_test, f_test, test_labels, test_metadata, test_path = load_dataset(args.test_dir, "test")

    log_print(f"Train file: {train_path}")
    log_print(f"Test  file: {test_path}")
    log_print(f"Train u shape: {tuple(u_train.shape)}")
    log_print(f"Train f shape: {tuple(f_train.shape)}")
    log_print(f"Test  u shape: {tuple(u_test.shape)}")
    log_print(f"Test  f shape: {tuple(f_test.shape)}")

    stats = compute_train_stats(u_train, f_train)

    log_print("\n[Normalization statistics from training data]")
    log_print(f"u_mean  = {stats['u_mean'].detach().cpu().view(-1).numpy()}")
    log_print(f"u_std   = {stats['u_std'].detach().cpu().view(-1).numpy()}")
    log_print(f"f_mean  = {stats['f_mean'].detach().cpu().view(-1).numpy()}")
    log_print(f"f_std   = {stats['f_std'].detach().cpu().view(-1).numpy()}")
    log_print(f"amp2_mean = {stats['amp2_mean'].item():.6e}")
    log_print(f"amp2_std  = {stats['amp2_std'].item():.6e}")

    model = NormalizedResidualLGNOAmp2(
        stats=stats,
        out_channels=2,
        hidden=args.hidden,
        depth=args.depth,
        stencil=args.stencil,
        dropout=args.dropout,
        last_scale=args.last_scale,
        variant=args.variant,
    ).to(device)

    log_print("\n[Model]")
    log_print(f"Class: NormalizedResidualLGNOAmp2  # no explicit V prior")
    log_print(f"Variant: LGNO({args.variant})")
    log_print(f"Stencil: {args.stencil}")
    log_print(f"Condition features: Re/Im normalized stencil + V normalized center + |psi|^2 center")
    log_print(f"Weight-input dimension: {model.weight_input_dim}")
    log_print(f"Dot-product dimension: {model.dot_dim}")
    log_print(f"Compatibility local_dim: {model.local_dim}")
    log_print("No explicit V*psi prior is added or folded into the center coefficient.")
    log_print("V is used only as an MLP input feature.")
    log_print("MLP learns the full normalized stencil weights.")
    log_print("Dot-product variables use q_raw / q_std to shrink large coefficients.")
    log_print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    train_time, loss_history = train(model, u_train, f_train, args, dirs, log_print)

    model.load_state_dict(torch.load(os.path.join(dirs["base"], "best_model.pth"), map_location=device))
    model.eval()

    train_eval = evaluate_physical(model, u_train, f_train)
    test_eval = evaluate_physical(model, u_test, f_test)

    log_print("\n[Final train metrics in physical scale]")
    log_print(f"Train Relative L2 error: {train_eval['rel_l2']:.6e}")
    log_print(f"Train MSE error        : {train_eval['mse']:.6e}")
    log_print(f"Train Re Relative L2   : {train_eval['re_rel_l2']:.6e}")
    log_print(f"Train Im Relative L2   : {train_eval['im_rel_l2']:.6e}")
    log_print(f"Train Re MSE           : {train_eval['re_mse']:.6e}")
    log_print(f"Train Im MSE           : {train_eval['im_mse']:.6e}")

    log_print("\n[Final test metrics in physical scale]")
    log_print(f"Test Relative L2 error : {test_eval['rel_l2']:.6e}")
    log_print(f"Test MSE error         : {test_eval['mse']:.6e}")
    log_print(f"Test Re Relative L2    : {test_eval['re_rel_l2']:.6e}")
    log_print(f"Test Im Relative L2    : {test_eval['im_rel_l2']:.6e}")
    log_print(f"Test Re MSE            : {test_eval['re_mse']:.6e}")
    log_print(f"Test Im MSE            : {test_eval['im_mse']:.6e}")

    # Save tensors in physical scale.
    torch.save(train_eval["pred"].detach().cpu(), os.path.join(dirs["base"], "train_f_pred.pt"))
    torch.save(f_train.detach().cpu(), os.path.join(dirs["base"], "train_f_true.pt"))
    torch.save(test_eval["pred"].detach().cpu(), os.path.join(dirs["base"], "test_f_pred.pt"))
    torch.save(f_test.detach().cpu(), os.path.join(dirs["base"], "test_f_true.pt"))

    # Save prediction/truth/error images for test set.
    save_all_test_images(test_eval["pred"], f_test, dirs, labels=test_labels)
    log_print("Test prediction/truth/error figures saved.")

    if args.save_weights:
        save_weight_maps(model, u_test, dirs, max_samples=min(1, u_test.shape[0]))
        log_print("Diagnostic local-weight figures saved.")

    log_print("All results saved successfully.")


if __name__ == "__main__":
    main()
