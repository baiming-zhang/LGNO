import os
import re
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader, Dataset

# ============================================================
# 3D Smoke Velocity Component-Aux-LGNO
#
# Learn:
#   vel_{t+1} = vel_t + local_part_pred - dt_eff * grad(p_t)
#
# where the network only learns the local remaining part:
#   local_part_pred ~ dt_eff * [-(v . grad)v + nu * laplacian(v) + other local numerical effects]
#
# IMPORTANT:
#   - The pressure gradient is NOT an input to the network.
#   - The pressure gradient is subtracted explicitly outside the network.
#   - The velocity stencil uses open/replicate boundary padding, not periodic roll.
#   - The model is component-shared with cross-component auxiliary input:
#         u, v, w are treated as three scalar samples.
#         each sample uses one scalar 7-point stencil as the main input,
#         while the full center velocity (u,v,w) and speed are auxiliary inputs.
#         sample count = 3 * TRAIN_TRANSITIONS.
#   - dt_eff is inferred from saved frame times or saved step stride.
#
# Data source:
#   existing saved fields/frame_*.npz
#
# Train:
#   first TRAIN_TRANSITIONS transitions.
#
# Rollout:
#   start from frame 0 and use known grad(p_t) from saved frames.
# ============================================================


# ============================================================
# 0. Direct config (edit here)
# ============================================================
SAVE_DIR = r"bunny_teapot_collision_128x128x128_open"  # set this to your path
TRAIN_TRANSITIONS = 100
ROLLOUT_STEPS = 100
HIDDEN = 16
EPOCHS = 800
LR = 2e-3
PATIENCE = 120
BATCH_SIZE = 1
WEIGHT_DECAY = 1e-6

# Adaptive LR schedule. The old scheduler patience was 400, so LR never dropped
# before early stopping. Now LR drops first; early stopping happens much later.
LR_FACTOR = 0.5
LR_PATIENCE = 8
LR_THRESHOLD = 1e-5
MIN_LR = 1e-6

# Numerical safety for rollout. This does not change training loss; it prevents
# one unstable component from turning all metrics into NaN.
ROLLOUT_CLAMP_NORM = 20.0
AUX_USE_SPEED = True
SEED = 42
CHUNK_VOXELS = 32768
# SAVE_FRAME_LIST = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SAVE_FRAME_LIST = list(range(1, 21))  # generates [1, ..., 20]


# ============================================================
# 1. Global settings
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 18
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.02

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = os.path.join(
    "LGNO_16hidden_results", f"smoke_component_aux_lgno_open_pgrad_run_{timestamp}"
)
os.makedirs(base_dir, exist_ok=True)

fig_dir = os.path.join(base_dir, "figures")
pred_dir = os.path.join(fig_dir, "predict")
truth_dir = os.path.join(fig_dir, "truth")
error_dir = os.path.join(fig_dir, "error")
loss_dir = os.path.join(fig_dir, "loss")

for d in [fig_dir, pred_dir, truth_dir, error_dir, loss_dir]:
    os.makedirs(d, exist_ok=True)

log_file = os.path.join(base_dir, "train_log.txt")


def log_print(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


log_print(f"Device: {device}")
log_print(f"Result directory: {base_dir}")
log_print(f"Data save_dir: {SAVE_DIR}")


# ============================================================
# 2. IO helpers
# ============================================================
def natural_frame_id(path):
    m = re.search(r"frame_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_metadata(save_dir):
    meta_path = os.path.join(save_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_field_files(save_dir):
    fields_dir = os.path.join(save_dir, "fields")
    if not os.path.isdir(fields_dir):
        raise FileNotFoundError(f"fields folder not found: {fields_dir}")

    files = [
        os.path.join(fields_dir, fn)
        for fn in os.listdir(fields_dir)
        if fn.endswith(".npz") and fn.startswith("frame_")
    ]
    files = sorted(files, key=natural_frame_id)

    if len(files) == 0:
        raise RuntimeError(f"No frame_*.npz found in {fields_dir}")
    return files


def central_diff_3d(f, dx, dy, dz):
    """
    f: [Nx,Ny,Nz]
    edge-padded central difference
    """
    fp = np.pad(f, ((1, 1), (1, 1), (1, 1)), mode="edge")

    dfdx = (fp[2:, 1:-1, 1:-1] - fp[:-2, 1:-1, 1:-1]) / (2.0 * dx)
    dfdy = (fp[1:-1, 2:, 1:-1] - fp[1:-1, :-2, 1:-1]) / (2.0 * dy)
    dfdz = (fp[1:-1, 1:-1, 2:] - fp[1:-1, 1:-1, :-2]) / (2.0 * dz)

    return dfdx.astype(np.float32), dfdy.astype(np.float32), dfdz.astype(np.float32)


def load_velocity_and_pgrad_trajectory(save_dir, rollout_steps):
    """
    Read:
        fields/frame_*.npz

    Return:
        vel   : [Nt,3,Nx,Ny,Nz]
        g_p   : [Nt,3,Nx,Ny,Nz]   where g_p = [p_x, p_y, p_z]
        times : [Nt]
        steps : [Nt]
        frame_ids : [Nt]
        meta  : metadata dict
    """
    meta = load_metadata(save_dir)
    files = list_field_files(save_dir)

    max_needed = rollout_steps + 1
    files = files[:max_needed]

    Lx = float(meta["Lx"])
    Ly = float(meta["Ly"])
    Lz = float(meta["Lz"])
    Nx = int(meta["Nx"])
    Ny = int(meta["Ny"])
    Nz = int(meta["Nz"])
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    vel_list = []
    pgrad_list = []
    time_list = []
    step_list = []
    frame_ids = []

    for fp in files:
        data = np.load(fp)

        u = data["u"].astype(np.float32)
        v = data["v"].astype(np.float32)
        w = data["w"].astype(np.float32)
        p = data["p"].astype(np.float32)

        px, py, pz = central_diff_3d(p, dx, dy, dz)

        vel = np.stack([u, v, w], axis=0)  # [3,Nx,Ny,Nz]
        g_p = np.stack([px, py, pz], axis=0)  # [3,Nx,Ny,Nz]

        vel_list.append(vel)
        pgrad_list.append(g_p)

        if "time" in data:
            time_list.append(float(data["time"]))
        else:
            time_list.append(np.nan)

        if "step" in data:
            step_list.append(int(data["step"]))
        else:
            step_list.append(len(step_list))

        frame_ids.append(natural_frame_id(fp))

    vel = np.stack(vel_list, axis=0)  # [Nt,3,Nx,Ny,Nz]
    pgrad = np.stack(pgrad_list, axis=0)  # [Nt,3,Nx,Ny,Nz]
    times = np.array(time_list, dtype=np.float32)
    steps = np.array(step_list, dtype=np.int32)
    frame_ids = np.array(frame_ids, dtype=np.int32)

    return (
        torch.from_numpy(vel).contiguous(),
        torch.from_numpy(pgrad).contiguous(),
        torch.from_numpy(times),
        torch.from_numpy(steps),
        torch.from_numpy(frame_ids),
        meta,
    )


def infer_effective_frame_dt(meta, times, steps):
    """
    Infer the effective time interval between two saved frames.

    Priority:
      1) use saved physical times if they are finite and increasing;
      2) otherwise use metadata solver dt times saved step stride;
      3) fallback to metadata["dt"].

    Return:
      dt_eff: float, the frame-to-frame time step used in training/rollout
      info  : dict, diagnostics for logging
    """
    meta_dt = float(meta["dt"])

    times_np = times.detach().cpu().numpy().astype(np.float64)
    steps_np = steps.detach().cpu().numpy().astype(np.int64)

    info = {"meta_dt": meta_dt, "source": "metadata_dt"}

    if len(times_np) >= 2 and np.all(np.isfinite(times_np[:2])):
        dt_seq = np.diff(times_np)
        valid = dt_seq[np.isfinite(dt_seq) & (dt_seq > 0)]
        if valid.size > 0:
            dt_eff = float(np.median(valid))
            info.update(
                {
                    "source": "saved_time_difference",
                    "dt_min": float(valid.min()),
                    "dt_max": float(valid.max()),
                    "dt_mean": float(valid.mean()),
                    "dt_median": dt_eff,
                    "dt_std": float(valid.std()),
                }
            )
            return dt_eff, info

    if len(steps_np) >= 2:
        stride_seq = np.diff(steps_np)
        valid_stride = stride_seq[stride_seq > 0]
        if valid_stride.size > 0:
            step_stride = int(np.median(valid_stride))
            dt_eff = float(meta_dt * step_stride)
            info.update(
                {
                    "source": "metadata_dt_times_saved_step_stride",
                    "step_stride_min": int(valid_stride.min()),
                    "step_stride_max": int(valid_stride.max()),
                    "step_stride_median": step_stride,
                    "dt_eff": dt_eff,
                }
            )
            return dt_eff, info

    return meta_dt, info


# ============================================================
# 3. Normalization
# ============================================================
def compute_channel_stats(x):
    """
    x: [Nt,3,Nx,Ny,Nz]
    return mean/std: [1,3,1,1,1]
    """
    mean = x.mean(dim=(0, 2, 3, 4), keepdim=True)
    std = x.std(dim=(0, 2, 3, 4), keepdim=True)
    std = torch.clamp(std, min=1e-6)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(xn, mean, std):
    return xn * std + mean


# ============================================================
# 4. Dataset construction
# ============================================================
def build_train_pairs(vel, pgrad, train_transitions):
    """
    vel  : [Nt,3,Nx,Ny,Nz]
    pgrad: [Nt,3,Nx,Ny,Nz]

    Use first train_transitions one-step pairs:
      x_t     = vel[0:train_transitions]
      y_t     = vel[1:train_transitions+1]
      gradp_t = pgrad[0:train_transitions]
    """
    if vel.shape[0] < train_transitions + 1:
        raise ValueError(
            f"Need at least {train_transitions+1} frames, got {vel.shape[0]}"
        )

    x = vel[:train_transitions].contiguous()
    y = vel[1 : train_transitions + 1].contiguous()
    gp = pgrad[:train_transitions].contiguous()
    return x, y, gp


class ComponentAuxDataset(Dataset):
    """
    Component-wise scalar samples with cross-component auxiliary velocity.

    For transition n and component c:
      main input  = x[n, c:c+1]       [1,Nx,Ny,Nz]
      auxiliary   = x[n, 0:3]         [3,Nx,Ny,Nz]
      target      = target[n,c:c+1]   [1,Nx,Ny,Nz]

    Length = Ns * 3, so u/v/w are still treated as three samples, but each
    sample can see the full advecting velocity vector at the same time step.
    """

    def __init__(self, x, target):
        super().__init__()
        assert x.ndim == 5 and target.ndim == 5
        assert x.shape == target.shape
        assert x.shape[1] == 3
        self.x = x
        self.target = target
        self.Ns = x.shape[0]
        self.C = x.shape[1]

    def __len__(self):
        return self.Ns * self.C

    def __getitem__(self, idx):
        n = idx // self.C
        c = idx % self.C
        x_comp = self.x[n, c : c + 1]
        x_aux = self.x[n]
        target_comp = self.target[n, c : c + 1]
        return x_comp, x_aux, target_comp


def make_component_aux_loader(x, target_local, batch_size=1, shuffle=True):
    ds = ComponentAuxDataset(x, target_local)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ============================================================
# 5. Model
# ============================================================
class ComponentAuxLGNO3D(nn.Module):
    """
    Component-shared 3D local LGNO with cross-component auxiliary input.

    The pressure-gradient term is still handled explicitly outside the model.
    The network learns the remaining local update.

    Main LGNO symmetry:
      The same scalar local operator is shared by u, v, and w. During training,
      the three velocity components are treated as three scalar samples.

    Cross-component auxiliary input:
      For each scalar component c in {u,v,w}, the model sees
        1) the 7-point stencil of c, and
        2) the full center velocity vector (u,v,w), optionally with speed.

      This is designed for the pressure-free momentum residual because each
      component approximately follows a scalar advection-diffusion form:

          dc/dt ~= -(v . grad)c + nu Deltac + local numerical effects.

      Therefore the same scalar operator can be shared across components, while
      the auxiliary velocity vector supplies the cross-component coupling needed
      for the advective term.

    Open-boundary stencil:
      Neighbor values are constructed with replicate padding, not torch.roll.
    """

    def __init__(
        self,
        hidden=16,
        chunk_voxels=32768,
        dt=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        nu=0.0,
        use_speed=True,
        init_small_final_weight=True,
    ):
        super().__init__()
        self.stencil_dim = 7
        self.aux_dim = 4 if use_speed else 3
        self.net_in_dim = self.stencil_dim + self.aux_dim
        self.chunk_voxels = int(chunk_voxels)
        self.use_speed = bool(use_speed)

        self.dt = float(dt)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)
        self.nu = float(nu)

        # The MLP predicts an input-dependent 7-point scalar stencil.
        # Input = scalar stencil + full center velocity auxiliary features.
        self.net = nn.Sequential(
            nn.Linear(self.net_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.stencil_dim),
        )

        self._init_as_viscous_stencil(init_small_final_weight=init_small_final_weight)

    def _build_viscous_bias(self):
        """
        Build constant bias corresponding to:
            dt * nu * Laplacian
        with scalar 7-point stencil order:
            [center, -x, +x, -y, +y, -z, +z]
        """
        cx = 1.0 / (self.dx * self.dx)
        cy = 1.0 / (self.dy * self.dy)
        cz = 1.0 / (self.dz * self.dz)

        lap = (
            self.dt
            * self.nu
            * np.array(
                [
                    -2.0 * (cx + cy + cz),  # center
                    cx,
                    cx,  # -x, +x
                    cy,
                    cy,  # -y, +y
                    cz,
                    cz,  # -z, +z
                ],
                dtype=np.float32,
            )
        )

        return torch.from_numpy(lap)

    def _init_as_viscous_stencil(self, init_small_final_weight=True):
        """
        Make the scalar LGNO initially output approximately:
            dt * nu * Laplacian(component)

        Since the final-layer bias directly represents the stencil weights,
        this initialization makes the model initially ignore the auxiliary
        velocity and behave like a fixed viscous stencil.
        """
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)

        nn.init.xavier_normal_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

        if init_small_final_weight:
            nn.init.xavier_normal_(self.net[4].weight, gain=1.0e-3)
        else:
            nn.init.zeros_(self.net[4].weight)

        with torch.no_grad():
            self.net[4].bias.copy_(self._build_viscous_bias())

    def _make_7point_patch_open(self, x):
        """
        x: [B,1,Nx,Ny,Nz]
        return: [B,7,Nx,Ny,Nz]

        F.pad uses the order:
          (z_left, z_right, y_left, y_right, x_left, x_right)
        for a 5D tensor [B,C,Nx,Ny,Nz].
        """
        xp = F.pad(x, pad=(1, 1, 1, 1, 1, 1), mode="replicate")

        center = xp[:, :, 1:-1, 1:-1, 1:-1]
        xm = xp[:, :, 0:-2, 1:-1, 1:-1]
        xp1 = xp[:, :, 2:, 1:-1, 1:-1]
        ym = xp[:, :, 1:-1, 0:-2, 1:-1]
        yp1 = xp[:, :, 1:-1, 2:, 1:-1]
        zm = xp[:, :, 1:-1, 1:-1, 0:-2]
        zp1 = xp[:, :, 1:-1, 1:-1, 2:]

        return torch.cat([center, xm, xp1, ym, yp1, zm, zp1], dim=1)

    def _make_aux_features(self, vel_aux):
        """
        vel_aux: [B,3,Nx,Ny,Nz]
        return : [B,aux_dim,Nx,Ny,Nz]

        Auxiliary features are intentionally local-center features, not a full
        21-dimensional vector stencil. This keeps the component-shared model
        compact while supplying the advecting velocity vector.
        """
        assert vel_aux.ndim == 5 and vel_aux.shape[1] == 3
        if self.use_speed:
            speed = torch.sqrt(torch.sum(vel_aux**2, dim=1, keepdim=True) + 1.0e-12)
            return torch.cat([vel_aux, speed], dim=1)
        return vel_aux

    def _forward_scalar(self, x_comp, vel_aux):
        """
        x_comp : [B,1,Nx,Ny,Nz], selected scalar component
        vel_aux: [B,3,Nx,Ny,Nz], full velocity at the same time step
        return local_part: [B,1,Nx,Ny,Nz]
        """
        B, C, Nx, Ny, Nz = x_comp.shape
        assert C == 1, f"Expected scalar C=1, got C={C}"
        assert vel_aux.shape == (
            B,
            3,
            Nx,
            Ny,
            Nz,
        ), f"vel_aux must have shape {(B, 3, Nx, Ny, Nz)}, got {tuple(vel_aux.shape)}"

        local = self._make_7point_patch_open(x_comp)  # [B,7,Nx,Ny,Nz]
        aux = self._make_aux_features(vel_aux)  # [B,aux_dim,Nx,Ny,Nz]

        local = local.permute(0, 2, 3, 4, 1).contiguous()  # [B,Nx,Ny,Nz,7]
        aux = aux.permute(0, 2, 3, 4, 1).contiguous()  # [B,Nx,Ny,Nz,aux_dim]

        nvox = Nx * Ny * Nz
        local = local.view(B, nvox, self.stencil_dim)
        aux = aux.view(B, nvox, self.aux_dim)

        out_chunks = []
        for s in range(0, nvox, self.chunk_voxels):
            e = min(s + self.chunk_voxels, nvox)

            lf = local[:, s:e, :]  # [B,chunk,7]
            af = aux[:, s:e, :]  # [B,chunk,aux_dim]

            lf2 = lf.reshape(-1, self.stencil_dim)  # [B*chunk,7]
            af2 = af.reshape(-1, self.aux_dim)  # [B*chunk,aux_dim]
            net_in = torch.cat([lf2, af2], dim=1)  # [B*chunk,7+aux_dim]

            weights = self.net(net_in)  # [B*chunk,7]
            out = torch.sum(weights * lf2, dim=1, keepdim=True)  # [B*chunk,1]
            out = out.view(B, e - s, 1)
            out_chunks.append(out)

        out = torch.cat(out_chunks, dim=1)  # [B,Nvox,1]
        out = out.view(B, Nx, Ny, Nz, 1).permute(0, 4, 1, 2, 3).contiguous()
        return out

    def forward(self, x_comp, vel_aux=None):
        """
        Two modes:

        Training mode:
          x_comp: [B,1,Nx,Ny,Nz]
          vel_aux: [B,3,Nx,Ny,Nz]
          return: [B,1,Nx,Ny,Nz]

        Rollout mode:
          x_comp: [B,3,Nx,Ny,Nz]
          vel_aux: None
          return: [B,3,Nx,Ny,Nz]
        """
        B, C, Nx, Ny, Nz = x_comp.shape

        if C == 1:
            if vel_aux is None:
                raise ValueError("vel_aux is required when x_comp has one component.")
            return self._forward_scalar(x_comp, vel_aux)

        if C == 3:
            if vel_aux is not None:
                raise ValueError(
                    "For vector rollout mode, pass only x_comp with C=3 and keep vel_aux=None."
                )
            full_vel = x_comp
            outs = [
                self._forward_scalar(x_comp[:, i : i + 1], full_vel) for i in range(3)
            ]
            return torch.cat(outs, dim=1).contiguous()

        raise ValueError(f"Expected C=1 or C=3, got C={C}")


# ============================================================
# 6. Save helpers
# ============================================================
def save_fig_all_formats(fig, save_stem):
    fig.savefig(save_stem + ".pdf")
    fig.savefig(save_stem + ".svg")
    fig.savefig(save_stem + ".png", dpi=300)
    plt.close(fig)


def save_single_field_image(field2d, save_stem, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
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
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_fig_all_formats(fig, save_stem)


def save_loss_curve(loss_history, save_stem):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(np.arange(len(loss_history)), loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, save_stem)


# ============================================================
# 7. Training
# ============================================================
def make_loader(x, target_local, batch_size=1, shuffle=True):
    ds = TensorDataset(x, target_local)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def train(
    model,
    x_train_n,
    target_local_n,
    lr,
    epochs,
    patience,
    batch_size,
    weight_decay,
    lr_factor=0.5,
    lr_patience=8,
    lr_threshold=1e-5,
    min_lr=1e-6,
):
    """
    x_train_n        : normalized input vector states    [Ns,3,Nx,Ny,Nz]
    target_local_n   : normalized local target residual  [Ns,3,Nx,Ny,Nz]

    target_local_n =
        (y_train_n - x_train_n) + dt * gradp / std
    because:
        y = x + local_part - dt * gradp
      => local_part = (y - x) + dt * gradp
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        threshold=lr_threshold,
        threshold_mode="rel",
        cooldown=2,
        min_lr=min_lr,
    )
    loss_fn = nn.MSELoss()
    loader = make_component_aux_loader(
        x_train_n, target_local_n, batch_size=batch_size, shuffle=True
    )

    best_loss = 1e30
    wait = 0
    loss_history = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for x_comp_batch, x_aux_batch, target_local_batch in loader:
            x_comp_batch = x_comp_batch.to(device, non_blocking=True)
            x_aux_batch = x_aux_batch.to(device, non_blocking=True)
            target_local_batch = target_local_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_local = model(x_comp_batch, x_aux_batch)
            loss = loss_fn(pred_local, target_local_batch)

            if not torch.isfinite(loss):
                log_print(
                    "Non-finite training loss detected; stopping training and reloading best model."
                )
                total_time = time.time() - start_time
                loss_history = np.array(loss_history, dtype=np.float64)
                np.save(os.path.join(base_dir, "loss_history.npy"), loss_history)
                np.savetxt(
                    os.path.join(base_dir, "loss_total.txt"), loss_history, fmt="%.12e"
                )
                if len(loss_history) > 0:
                    save_loss_curve(
                        loss_history, os.path.join(loss_dir, "training_loss")
                    )
                return total_time, loss_history

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x_comp_batch.shape[0]
            running_loss += loss.item() * bs
            n_seen += bs

        epoch_loss = running_loss / max(n_seen, 1)

        if not np.isfinite(epoch_loss):
            log_print(f"Non-finite epoch loss at epoch {epoch}; stopping training.")
            break

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            log_print(f"LR reduced: {old_lr:.3e} -> {new_lr:.3e} at epoch {epoch}")
            # After an LR drop, give the optimizer more time before early stopping.
            wait = 0

        loss_history.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(base_dir, "best_model.pth"))
        else:
            wait += 1

        if epoch % 1 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            log_print(f"[{epoch:6d}] loss = {epoch_loss:.6e} | lr = {current_lr:.3e}")

        if wait > patience:
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr <= min_lr * 1.01:
                log_print(f"Early stopping at epoch {epoch} after LR reached min_lr")
                break
            else:
                log_print(
                    f"No improvement for {wait} epochs, but LR is still above min_lr; continue for scheduler."
                )
                wait = 0

    total_time = time.time() - start_time
    log_print(f"Training time: {total_time:.2f} sec")

    loss_history = np.array(loss_history, dtype=np.float64)
    np.save(os.path.join(base_dir, "loss_history.npy"), loss_history)
    np.savetxt(os.path.join(base_dir, "loss_total.txt"), loss_history, fmt="%.12e")
    save_loss_curve(loss_history, os.path.join(loss_dir, "training_loss"))

    return total_time, loss_history


# ============================================================
# 8. Rollout
# ============================================================
@torch.no_grad()
def rollout(model, x0_n, gp_corr_n_seq, rollout_steps, clamp_norm=None):
    """
    x0_n        : [3,Nx,Ny,Nz] normalized initial velocity
    gp_corr_n_seq:
        [rollout_steps,3,Nx,Ny,Nz]
        = dt * gradp / std   in normalized velocity units

    update:
        x_{t+1} = x_t + local_pred - gp_corr_n_seq[t]
    """
    model.eval()

    x = x0_n.unsqueeze(0).to(device)  # [1,3,Nx,Ny,Nz]
    preds = [x.squeeze(0).cpu()]

    for t in range(rollout_steps):
        local_pred = model(x)  # [1,3,Nx,Ny,Nz]
        gp_corr_t = gp_corr_n_seq[t : t + 1].to(device)

        x = x + local_pred - gp_corr_t

        if clamp_norm is not None:
            x = torch.nan_to_num(x, nan=0.0, posinf=clamp_norm, neginf=-clamp_norm)
            x = torch.clamp(x, min=-clamp_norm, max=clamp_norm)
        elif not torch.isfinite(x).all():
            log_print(f"Non-finite rollout state detected at step {t + 1}.")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

        preds.append(x.squeeze(0).cpu())

    return torch.stack(preds, dim=0).contiguous()


# ============================================================
# 9. Metrics
# ============================================================
def relative_l2(pred, true):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    true = torch.nan_to_num(true, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    return torch.norm(pred - true) / (torch.norm(true) + 1e-20)


def compute_rollout_metrics(pred_vel, true_vel):
    """
    pred_vel, true_vel: [T,3,Nx,Ny,Nz]
    """
    pred_vel = torch.nan_to_num(pred_vel, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    true_vel = torch.nan_to_num(true_vel, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    diff = pred_vel - true_vel

    metrics = {}
    metrics["rel_l2_all"] = relative_l2(pred_vel[1:], true_vel[1:])
    metrics["mse_all"] = torch.mean(diff[1:] ** 2)

    names = ["u", "v", "w"]
    for i, name in enumerate(names):
        metrics[f"rel_l2_{name}"] = relative_l2(
            pred_vel[1:, i : i + 1], true_vel[1:, i : i + 1]
        )
        metrics[f"mse_{name}"] = torch.mean(diff[1:, i : i + 1] ** 2)

    pred_speed = torch.sqrt(torch.sum(pred_vel**2, dim=1))
    true_speed = torch.sqrt(torch.sum(true_vel**2, dim=1))
    metrics["rel_l2_speed"] = relative_l2(pred_speed[1:], true_speed[1:])
    metrics["mse_speed"] = torch.mean((pred_speed[1:] - true_speed[1:]) ** 2)

    return metrics


# ============================================================
# 10. Visualization
# ============================================================
def save_rollout_images(pred_vel, true_vel, frame_ids_to_save):
    """
    Save z-mid slice images for:
      u, v, w, speed
    """
    pred_np = pred_vel.numpy()
    true_np = true_vel.numpy()

    T, C, Nx, Ny, Nz = pred_np.shape
    z_mid = Nz // 2

    names = ["u", "v", "w"]
    cmaps = ["seismic", "seismic", "seismic"]

    valid_frames = [t for t in frame_ids_to_save if 0 <= t < T]

    for t in valid_frames:
        for c in range(3):
            pred_img = pred_np[t, c, :, :, z_mid]
            true_img = true_np[t, c, :, :, z_mid]
            err_img = pred_img - true_img

            vmax = np.max(np.abs(true_img))
            if vmax == 0:
                vmax = 1.0
            err_abs = np.max(np.abs(err_img))
            if err_abs == 0:
                err_abs = 1.0

            save_single_field_image(
                pred_img,
                os.path.join(pred_dir, f"frame{t:03d}_{names[c]}_predict"),
                cmap=cmaps[c],
                vmin=-vmax,
                vmax=vmax,
            )
            save_single_field_image(
                true_img,
                os.path.join(truth_dir, f"frame{t:03d}_{names[c]}_truth"),
                cmap=cmaps[c],
                vmin=-vmax,
                vmax=vmax,
            )
            save_single_field_image(
                err_img,
                os.path.join(error_dir, f"frame{t:03d}_{names[c]}_error"),
                cmap="seismic",
                vmin=-err_abs,
                vmax=err_abs,
            )

        pred_speed = np.sqrt(
            pred_np[t, 0, :, :, z_mid] ** 2
            + pred_np[t, 1, :, :, z_mid] ** 2
            + pred_np[t, 2, :, :, z_mid] ** 2
        )
        true_speed = np.sqrt(
            true_np[t, 0, :, :, z_mid] ** 2
            + true_np[t, 1, :, :, z_mid] ** 2
            + true_np[t, 2, :, :, z_mid] ** 2
        )
        err_speed = pred_speed - true_speed

        vmax = np.max(true_speed)
        if vmax == 0:
            vmax = 1.0
        err_abs = np.max(np.abs(err_speed))
        if err_abs == 0:
            err_abs = 1.0

        save_single_field_image(
            pred_speed,
            os.path.join(pred_dir, f"frame{t:03d}_speed_predict"),
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        save_single_field_image(
            true_speed,
            os.path.join(truth_dir, f"frame{t:03d}_speed_truth"),
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        save_single_field_image(
            err_speed,
            os.path.join(error_dir, f"frame{t:03d}_speed_error"),
            cmap="seismic",
            vmin=-err_abs,
            vmax=err_abs,
        )

    log_print(f"Saved rollout images for frames: {valid_frames}")


# ============================================================
# 11. Main
# ============================================================
def main():
    cfg = {
        "SAVE_DIR": SAVE_DIR,
        "TRAIN_TRANSITIONS": TRAIN_TRANSITIONS,
        "ROLLOUT_STEPS": ROLLOUT_STEPS,
        "HIDDEN": HIDDEN,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "PATIENCE": PATIENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LR_FACTOR": LR_FACTOR,
        "LR_PATIENCE": LR_PATIENCE,
        "LR_THRESHOLD": LR_THRESHOLD,
        "MIN_LR": MIN_LR,
        "ROLLOUT_CLAMP_NORM": ROLLOUT_CLAMP_NORM,
        "AUX_USE_SPEED": AUX_USE_SPEED,
        "SEED": SEED,
        "CHUNK_VOXELS": CHUNK_VOXELS,
        "SAVE_FRAME_LIST": SAVE_FRAME_LIST,
        "MODEL_TYPE": "ComponentAuxLGNO3D_shared_scalar_with_cross_component_aux_open_boundary",
    }
    with open(os.path.join(base_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # ----------------------------
    # load trajectory
    # ----------------------------
    vel, pgrad, times, steps, frame_ids, meta = load_velocity_and_pgrad_trajectory(
        SAVE_DIR, ROLLOUT_STEPS
    )

    Nt, C, Nx, Ny, Nz = vel.shape
    dt_solver = float(meta["dt"])
    dt, dt_info = infer_effective_frame_dt(meta, times, steps)
    nu = float(meta["nu"])
    Lx = float(meta["Lx"])
    Ly = float(meta["Ly"])
    Lz = float(meta["Lz"])
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    log_print(f"Loaded velocity trajectory shape: {tuple(vel.shape)}")
    log_print(f"Loaded pressure-gradient shape: {tuple(pgrad.shape)}")
    log_print(f"Total available frames used: {Nt}")
    log_print(f"Grid = {Nx} x {Ny} x {Nz}")
    log_print(f"solver dt from metadata = {dt_solver:.6e}")
    log_print(f"effective frame dt used by learning = {dt:.6e}")
    log_print(f"effective dt source = {dt_info.get('source')}")
    for k, v in dt_info.items():
        if k not in ["source", "meta_dt"]:
            log_print(f"  {k} = {v}")
    log_print(f"nu = {nu:.6e}")
    log_print(f"dx = {dx:.6e}, dy = {dy:.6e}, dz = {dz:.6e}")

    with open(os.path.join(base_dir, "effective_dt.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"dt_solver": dt_solver, "dt_eff": dt, "dt_info": dt_info}, f, indent=2
        )

    if Nt < TRAIN_TRANSITIONS + 1:
        raise RuntimeError(f"Need at least {TRAIN_TRANSITIONS + 1} frames, got {Nt}")

    rollout_steps = min(ROLLOUT_STEPS, Nt - 1)
    log_print(f"Actual rollout_steps = {rollout_steps}")

    # ----------------------------
    # train split
    # ----------------------------
    x_train, y_train, gp_train = build_train_pairs(vel, pgrad, TRAIN_TRANSITIONS)
    log_print(f"x_train shape: {tuple(x_train.shape)}")
    log_print(f"y_train shape: {tuple(y_train.shape)}")
    log_print(f"gp_train shape: {tuple(gp_train.shape)}")

    # normalize by train velocity window only
    train_states_for_stats = vel[: TRAIN_TRANSITIONS + 1]
    mean, std = compute_channel_stats(train_states_for_stats)

    torch.save(mean, os.path.join(base_dir, "state_mean.pt"))
    torch.save(std, os.path.join(base_dir, "state_std.pt"))

    x_train_n = normalize(x_train, mean, std).float()
    y_train_n = normalize(y_train, mean, std).float()

    # pressure gradient correction in normalized velocity units
    gp_corr_train_n = (dt * gp_train / std).float()

    # target local part:
    #   local = (y - x) + dt * gradp
    # in normalized units:
    #   local_n = (y_n - x_n) + dt * gradp / std
    target_local_train_n = (y_train_n - x_train_n + gp_corr_train_n).float()

    # Component-wise scalar training is handled lazily by ComponentAuxDataset.
    # It exposes Ns*3 samples without physically repeating the full auxiliary
    # velocity field in memory.
    log_print(f"component-wise auxiliary samples: {x_train_n.shape[0] * 3}")
    log_print(
        "each sample: scalar component [1,Nx,Ny,Nz] + auxiliary velocity [3,Nx,Ny,Nz]"
    )

    true_rollout_vel = vel[: rollout_steps + 1].float()
    true_rollout_vel_n = normalize(true_rollout_vel, mean, std).float()

    # rollout needs gradp_t for t = 0..rollout_steps-1
    gp_rollout = pgrad[:rollout_steps].float()
    gp_corr_rollout_n = (dt * gp_rollout / std).float()

    # ----------------------------
    # model
    # ----------------------------
    model = ComponentAuxLGNO3D(
        hidden=HIDDEN,
        chunk_voxels=CHUNK_VOXELS,
        dt=dt,
        dx=dx,
        dy=dy,
        dz=dz,
        nu=nu,
        use_speed=AUX_USE_SPEED,
        init_small_final_weight=True,  # keep a small amount of trainability
    ).to(device)
    # print initialization information for easy confirmation
    lap_center = dt * nu * (-2.0 / dx**2 - 2.0 / dy**2 - 2.0 / dz**2)
    lap_x = dt * nu * (1.0 / dx**2)
    lap_y = dt * nu * (1.0 / dy**2)
    lap_z = dt * nu * (1.0 / dz**2)

    log_print("Initialized MLP close to viscous stencil:")
    log_print(f"  center = {lap_center:.6e}")
    log_print(f"  x-neigh = {lap_x:.6e}")
    log_print(f"  y-neigh = {lap_y:.6e}")
    log_print(f"  z-neigh = {lap_z:.6e}")

    nparams = sum(p.numel() for p in model.parameters())
    log_print(
        f"Auxiliary features: center velocity (u,v,w)"
        + (" + speed" if AUX_USE_SPEED else "")
    )
    log_print(
        f"Network input dimension: {model.net_in_dim} = 7 stencil + {model.aux_dim} auxiliary"
    )
    log_print(f"Total parameters: {nparams}")

    # ----------------------------
    # train
    # ----------------------------
    train_time, loss_history = train(
        model=model,
        x_train_n=x_train_n,
        target_local_n=target_local_train_n,
        lr=LR,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        lr_factor=LR_FACTOR,
        lr_patience=LR_PATIENCE,
        lr_threshold=LR_THRESHOLD,
        min_lr=MIN_LR,
    )

    # ----------------------------
    # reload best
    # ----------------------------
    model.load_state_dict(
        torch.load(os.path.join(base_dir, "best_model.pth"), map_location=device)
    )
    model.eval()

    # ----------------------------
    # rollout
    # ----------------------------
    pred_rollout_n = rollout(
        model=model,
        x0_n=true_rollout_vel_n[0],
        gp_corr_n_seq=gp_corr_rollout_n,
        rollout_steps=rollout_steps,
        clamp_norm=ROLLOUT_CLAMP_NORM,
    )
    pred_rollout_vel = denormalize(pred_rollout_n, mean.cpu(), std.cpu()).float()

    torch.save(pred_rollout_vel, os.path.join(base_dir, "pred_rollout_vel.pt"))
    torch.save(true_rollout_vel, os.path.join(base_dir, "true_rollout_vel.pt"))
    torch.save(times[: rollout_steps + 1], os.path.join(base_dir, "times.pt"))
    torch.save(steps[: rollout_steps + 1], os.path.join(base_dir, "steps.pt"))
    torch.save(frame_ids[: rollout_steps + 1], os.path.join(base_dir, "frame_ids.pt"))
    torch.save(gp_rollout, os.path.join(base_dir, "rollout_pgrad.pt"))

    # ----------------------------
    # metrics
    # ----------------------------
    metrics = compute_rollout_metrics(pred_rollout_vel, true_rollout_vel)

    for k, v in metrics.items():
        log_print(f"{k}: {v.item():.6e}")

    with open(os.path.join(base_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v.item():.12e}\n")

    # ----------------------------
    # figures
    # ----------------------------
    save_rollout_images(
        pred_vel=pred_rollout_vel.cpu(),
        true_vel=true_rollout_vel.cpu(),
        frame_ids_to_save=SAVE_FRAME_LIST,
    )

    log_print("All results saved successfully.")


if __name__ == "__main__":
    main()
