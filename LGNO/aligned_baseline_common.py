import argparse
import itertools
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prod(values):
    out = 1
    for value in values:
        out *= int(value)
    return out


def init_module_weights(module):
    for layer in module.modules():
        if isinstance(
            layer,
            (
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
            ),
        ):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def conv_nd(ndim):
    if ndim == 1:
        return nn.Conv1d
    if ndim == 2:
        return nn.Conv2d
    if ndim == 3:
        return nn.Conv3d
    raise ValueError(f"Unsupported ndim={ndim}")


def adaptive_avg_pool_nd(ndim, output_size):
    if ndim == 1:
        return nn.AdaptiveAvgPool1d(output_size)
    if ndim == 2:
        return nn.AdaptiveAvgPool2d((output_size, output_size))
    if ndim == 3:
        return nn.AdaptiveAvgPool3d((output_size, output_size, output_size))
    raise ValueError(f"Unsupported ndim={ndim}")


def circular_pad(x, radius=1):
    ndim = x.dim() - 2
    pad = tuple([radius, radius] * ndim)
    return F.pad(x, pad, mode="circular")


def make_grid(batch_size, spatial_shape, device):
    axes = [
        torch.linspace(0.0, 1.0, int(n), device=device)
        for n in spatial_shape
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack(mesh, dim=0).unsqueeze(0)
    return grid.repeat(batch_size, 1, *([1] * len(spatial_shape)))


def make_coords(spatial_shape, device):
    axes = [
        torch.linspace(0.0, 1.0, int(n), device=device)
        for n in spatial_shape
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack([m.reshape(-1) for m in mesh], dim=1)
    return coords


class GlobalMLP(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_shape, hidden, depth=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        in_dim = in_channels * prod(self.spatial_shape)
        out_dim = out_channels * prod(self.spatial_shape)
        layers = []
        dim = in_dim
        for _ in range(max(depth - 1, 1)):
            layers += [nn.Linear(dim, hidden), nn.SiLU()]
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)
        init_module_weights(self)

    def forward(self, x):
        if tuple(x.shape[2:]) != self.spatial_shape:
            raise ValueError(
                f"GlobalMLP expects spatial shape {self.spatial_shape}, "
                f"got {tuple(x.shape[2:])}"
            )
        b = x.shape[0]
        out = self.net(x.reshape(b, -1))
        return out.view(b, self.out_channels, *self.spatial_shape)


class LocalPatchMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, ndim, depth=3, radius=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ndim = ndim
        self.radius = radius
        values = list(range(-radius, radius + 1))
        self.offsets = list(itertools.product(values, repeat=ndim))
        self.local_dim = in_channels * len(self.offsets)

        layers = [nn.Linear(self.local_dim, hidden), nn.SiLU()]
        for _ in range(max(depth - 2, 0)):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, out_channels))
        self.net = nn.Sequential(*layers)
        init_module_weights(self)

    def forward(self, x):
        dims = tuple(range(2, 2 + self.ndim))
        patches = [
            torch.roll(x, shifts=offset, dims=dims)
            for offset in self.offsets
        ]
        patch = torch.cat(patches, dim=1)
        b, _, *shape = patch.shape
        local = patch.permute(0, *range(2, 2 + self.ndim), 1).reshape(
            -1, self.local_dim
        )
        out = self.net(local)
        out = out.view(b, *shape, self.out_channels)
        return out.permute(0, self.ndim + 1, *range(1, self.ndim + 1)).contiguous()


class ConvNetND(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, ndim, depth=4):
        super().__init__()
        Conv = conv_nd(ndim)
        self.ndim = ndim
        layers = []
        c_in = in_channels
        for _ in range(max(depth - 1, 1)):
            layers += [Conv(c_in, hidden, kernel_size=3, padding=0), nn.SiLU()]
            c_in = hidden
        layers.append(Conv(c_in, out_channels, kernel_size=1))
        self.layers = nn.ModuleList(layers)
        init_module_weights(self)

    def forward(self, x):
        out = x
        for layer in self.layers:
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and layer.kernel_size != (1,) * self.ndim:
                out = circular_pad(out, 1)
            out = layer(out)
        return out


class CNONetND(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, ndim, depth=4, use_grid=True):
        super().__init__()
        self.ndim = ndim
        self.use_grid = use_grid
        Conv = conv_nd(ndim)
        c_in = in_channels + (ndim if use_grid else 0)
        self.lift = Conv(c_in, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(hidden, hidden, kernel_size=3, padding=0),
                    nn.SiLU(),
                    Conv(hidden, hidden, kernel_size=3, padding=0),
                )
                for _ in range(depth)
            ]
        )
        self.proj = Conv(hidden, out_channels, kernel_size=1)
        init_module_weights(self)

    def forward(self, x):
        if self.use_grid:
            grid = make_grid(x.shape[0], x.shape[2:], x.device)
            x = torch.cat([x, grid], dim=1)
        h = self.lift(x)
        for block in self.blocks:
            y = circular_pad(h, 1)
            y = block[0](y)
            y = block[1](y)
            y = circular_pad(y, 1)
            y = block[2](y)
            h = F.silu(h + y)
        return self.proj(h)


class SpectralConvND(nn.Module):
    def __init__(self, in_channels, out_channels, modes, ndim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ndim = ndim
        if isinstance(modes, int):
            modes = (modes,) * ndim
        self.modes = tuple(int(m) for m in modes)
        scale = 1.0 / max(1, in_channels * out_channels)
        weight_shape = (in_channels, out_channels, *self.modes)
        self.weight = nn.Parameter(
            scale * torch.randn(*weight_shape, dtype=torch.cfloat)
        )

    def forward(self, x):
        spatial_shape = tuple(x.shape[2:])
        x_ft = torch.fft.rfftn(x, dim=tuple(range(2, 2 + self.ndim)))
        out_shape = (x.shape[0], self.out_channels, *x_ft.shape[2:])
        out_ft = torch.zeros(out_shape, dtype=torch.cfloat, device=x.device)
        slices = tuple(
            slice(0, min(self.modes[i], x_ft.shape[2 + i]))
            for i in range(self.ndim)
        )
        x_low = x_ft[(slice(None), slice(None), *slices)]
        w_low = self.weight[
            (slice(None), slice(None), *tuple(slice(0, s.stop) for s in slices))
        ]
        out_ft[(slice(None), slice(None), *slices)] = torch.einsum(
            "bc...,co...->bo...", x_low, w_low
        )
        return torch.fft.irfftn(
            out_ft,
            s=spatial_shape,
            dim=tuple(range(2, 2 + self.ndim)),
        )


class FNONetND(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden,
        ndim,
        modes=8,
        depth=4,
        use_grid=True,
    ):
        super().__init__()
        self.ndim = ndim
        self.use_grid = use_grid
        Conv = conv_nd(ndim)
        c_in = in_channels + (ndim if use_grid else 0)
        self.lift = Conv(c_in, hidden, kernel_size=1)
        self.spectral = nn.ModuleList(
            [SpectralConvND(hidden, hidden, modes=modes, ndim=ndim) for _ in range(depth)]
        )
        self.pointwise = nn.ModuleList(
            [Conv(hidden, hidden, kernel_size=1) for _ in range(depth)]
        )
        self.proj = nn.Sequential(
            Conv(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            Conv(hidden, out_channels, kernel_size=1),
        )
        init_module_weights(self)

    def forward(self, x):
        if self.use_grid:
            x = torch.cat([x, make_grid(x.shape[0], x.shape[2:], x.device)], dim=1)
        h = self.lift(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            h = F.silu(spec(h) + pw(h))
        return self.proj(h)


class GraphMessageLayerND(nn.Module):
    def __init__(self, channels, ndim, message_hidden):
        super().__init__()
        self.ndim = ndim
        self.offsets = []
        for axis in range(ndim):
            for sign in (-1, 1):
                offset = [0] * ndim
                offset[axis] = sign
                self.offsets.append(tuple(offset))
        Conv = conv_nd(ndim)
        self.msg = nn.Sequential(
            Conv(2 * channels + ndim, message_hidden, kernel_size=1),
            nn.SiLU(),
            Conv(message_hidden, channels, kernel_size=1),
        )
        self.update = nn.Sequential(
            Conv(2 * channels, message_hidden, kernel_size=1),
            nn.SiLU(),
            Conv(message_hidden, channels, kernel_size=1),
        )
        init_module_weights(self)

    def forward(self, h):
        dims = tuple(range(2, 2 + self.ndim))
        messages = []
        for offset in self.offsets:
            neigh = torch.roll(h, shifts=offset, dims=dims)
            direction = torch.tensor(offset, device=h.device, dtype=h.dtype).view(
                1, self.ndim, *([1] * self.ndim)
            )
            direction = direction.repeat(h.shape[0], 1, *h.shape[2:])
            messages.append(self.msg(torch.cat([h, neigh, direction], dim=1)))
        agg = torch.stack(messages, dim=0).mean(dim=0)
        return F.silu(h + self.update(torch.cat([h, agg], dim=1)))


class GNONetND(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, ndim, layers=3):
        super().__init__()
        Conv = conv_nd(ndim)
        self.lift = Conv(in_channels, hidden, kernel_size=1)
        self.layers = nn.ModuleList(
            [GraphMessageLayerND(hidden, ndim, hidden) for _ in range(layers)]
        )
        self.proj = Conv(hidden, out_channels, kernel_size=1)
        init_module_weights(self)

    def forward(self, x):
        h = self.lift(x)
        for layer in self.layers:
            h = layer(h)
        return self.proj(h)


class DeepONetND(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_shape,
        hidden,
        ndim,
        p=None,
        trunk_depth=4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.ndim = ndim
        self.p = int(p or hidden)
        branch_in = in_channels * prod(self.spatial_shape)
        self.branch = nn.Sequential(
            nn.Linear(branch_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_channels * self.p),
        )
        trunk_layers = [nn.Linear(ndim, hidden), nn.SiLU()]
        for _ in range(max(trunk_depth - 1, 0)):
            trunk_layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        trunk_layers.append(nn.Linear(hidden, self.p))
        self.trunk = nn.Sequential(*trunk_layers)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        init_module_weights(self)

    def forward(self, x):
        if tuple(x.shape[2:]) != self.spatial_shape:
            raise ValueError(
                f"DeepONet expects spatial shape {self.spatial_shape}, "
                f"got {tuple(x.shape[2:])}"
            )
        b = x.shape[0]
        branch = self.branch(x.reshape(b, -1)).view(b, self.out_channels, self.p)
        coords = make_coords(self.spatial_shape, x.device)
        trunk = self.trunk(coords)
        out = torch.einsum("bcp,np->bcn", branch, trunk)
        out = out + self.bias.view(1, self.out_channels, 1)
        return out.view(b, self.out_channels, *self.spatial_shape)


def build_model(
    method,
    in_channels,
    out_channels,
    spatial_shape,
    hidden,
    ndim,
    modes=8,
    depth=4,
):
    key = method.lower()
    if key == "mlp":
        return GlobalMLP(in_channels, out_channels, spatial_shape, hidden, depth=3)
    if key == "mlpconv":
        return LocalPatchMLP(in_channels, out_channels, hidden, ndim, depth=3)
    if key == "cnn":
        return ConvNetND(in_channels, out_channels, hidden, ndim, depth=depth)
    if key in {"oneshot", "oneshotpde"}:
        return LocalPatchMLP(in_channels, out_channels, hidden, ndim, depth=4)
    if key == "fno":
        return FNONetND(
            in_channels,
            out_channels,
            hidden,
            ndim,
            modes=modes,
            depth=depth,
            use_grid=True,
        )
    if key == "cno":
        return CNONetND(
            in_channels,
            out_channels,
            hidden,
            ndim,
            depth=depth,
            use_grid=True,
        )
    if key == "gno":
        return GNONetND(in_channels, out_channels, hidden, ndim, layers=3)
    if key == "deeponet":
        return DeepONetND(
            in_channels,
            out_channels,
            spatial_shape,
            hidden,
            ndim,
            p=hidden,
            trunk_depth=4,
        )
    raise ValueError(f"Unknown method: {method}")


def method_label(method):
    key = method.lower()
    if key in {"oneshot", "oneshotpde"}:
        return "OneShotPDE"
    if key == "deeponet":
        return "DeepONet"
    return method.upper() if key in {"fno", "cno", "gno", "mlp", "cnn"} else method


class RunContext:
    def __init__(self, result_root):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(result_root) / f"run_{timestamp}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.base_dir / "train_log.txt"

    def log(self, text):
        print(text, flush=True)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def find_dataset_file(folder, mode):
    folder = Path(folder)
    for name in (f"{mode}_dataset.pt", f"{mode}.pt"):
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find {mode} dataset in {folder}")


def reshape_static_tensor(tensor, channels, ndim, name):
    if tensor.dim() == ndim + 2 and tensor.shape[1] == channels:
        return tensor.float().contiguous()
    if tensor.dim() == ndim + 1 and tensor.shape[0] == channels:
        return tensor.unsqueeze(0).float().contiguous()
    if ndim == 3 and tensor.dim() == 6 and tensor.shape[1] == channels:
        b, c, nx, ny, nz, nt = tensor.shape
        return (
            tensor.permute(0, 5, 1, 2, 3, 4)
            .contiguous()
            .view(b * nt, c, nx, ny, nz)
            .float()
        )
    raise ValueError(f"Unsupported {name} shape {tuple(tensor.shape)}")


def load_static_dataset(folder, mode, in_channels, out_channels, ndim):
    path = find_dataset_file(folder, mode)
    data = torch.load(path, map_location="cpu")
    u = reshape_static_tensor(data["u"], in_channels, ndim, "u")
    f = reshape_static_tensor(data["f"], out_channels, ndim, "f")
    labels = data.get("labels", None)
    return u, f, labels, str(path)


def make_loader(u, f, batch_size, shuffle):
    return DataLoader(
        TensorDataset(u, f),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def predict_batches(model, u, batch_size):
    loader = DataLoader(
        u,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=True)
            preds.append(model(batch).cpu())
    return torch.cat(preds, dim=0)


def rel_l2(pred, true):
    return torch.norm(pred - true) / (torch.norm(true) + 1e-20)


def train_static_model(ctx, model, u_train, f_train, args):
    model = model.to(DEVICE)
    opt_class = optim.AdamW if args.weight_decay > 0 else optim.Adam
    optimizer = opt_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        threshold=1e-6,
        min_lr=args.min_lr,
    )
    loss_fn = nn.MSELoss()
    loader = make_loader(u_train, f_train, args.batch_size, shuffle=True)
    best_loss = float("inf")
    wait = 0
    losses = []
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n_seen = 0
        for u_batch, f_batch in loader:
            u_batch = u_batch.to(DEVICE, non_blocking=True)
            f_batch = f_batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(u_batch)
            loss = loss_fn(pred, f_batch)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.item()) * u_batch.shape[0]
            n_seen += u_batch.shape[0]

        loss_val = total / max(n_seen, 1)
        losses.append(loss_val)
        scheduler.step(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            wait = 0
            torch.save(model.state_dict(), ctx.base_dir / "best_model.pth")
        else:
            wait += 1

        if epoch % args.print_every == 0:
            ctx.log(
                f"[{epoch:6d}] loss = {loss_val:.6e} | "
                f"lr = {optimizer.param_groups[0]['lr']:.3e}"
            )
        if wait > args.patience:
            ctx.log(f"Early stopping at epoch {epoch}")
            break

    duration = time.time() - start
    ctx.log(f"Training time: {duration:.2f} sec")
    ctx.log(f"Best training loss: {best_loss:.6e}")
    np.savetxt(ctx.base_dir / "loss_total.txt", np.asarray(losses), fmt="%.12e")
    np.save(ctx.base_dir / "loss_history.npy", np.asarray(losses, dtype=np.float64))


def run_static_cli(
    method,
    out_dir,
    hidden,
    epochs,
    lr,
    patience,
    in_channels,
    out_channels,
    ndim,
    train_dir="dataset/train_data",
    test_dir="dataset/test_data",
    batch_size=1,
    test_batch_size=None,
    scheduler_patience=500,
    min_lr=1e-6,
    weight_decay=0.0,
    grad_clip=1.0,
    print_every=200,
    modes=8,
    depth=4,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=hidden)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--patience", type=int, default=patience)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--test_batch_size", type=int, default=test_batch_size or batch_size)
    parser.add_argument("--scheduler_patience", type=int, default=scheduler_patience)
    parser.add_argument("--min_lr", type=float, default=min_lr)
    parser.add_argument("--weight_decay", type=float, default=weight_decay)
    parser.add_argument("--grad_clip", type=float, default=grad_clip)
    parser.add_argument("--print_every", type=int, default=print_every)
    parser.add_argument("--modes", type=int, default=modes)
    parser.add_argument("--depth", type=int, default=depth)
    parser.add_argument("--train_dir", type=str, default=train_dir)
    parser.add_argument("--test_dir", type=str, default=test_dir)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    label = method_label(method)
    ctx = RunContext(out_dir.format(hidden=args.hidden, method=label))
    ctx.log(f"Device: {DEVICE}")
    ctx.log(f"Result directory: {ctx.base_dir}")

    with (ctx.base_dir / "config.txt").open("w", encoding="utf-8") as f:
        f.write(f"method: {label}\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    u_train, f_train, train_labels, train_path = load_static_dataset(
        args.train_dir, "train", in_channels, out_channels, ndim
    )
    u_test, f_test, test_labels, test_path = load_static_dataset(
        args.test_dir, "test", in_channels, out_channels, ndim
    )
    spatial_shape = tuple(u_train.shape[2:])
    ctx.log(f"Train file: {train_path}")
    ctx.log(f"Test  file: {test_path}")
    ctx.log(f"Train u shape: {tuple(u_train.shape)}")
    ctx.log(f"Train f shape: {tuple(f_train.shape)}")
    ctx.log(f"Test  u shape: {tuple(u_test.shape)}")
    ctx.log(f"Test  f shape: {tuple(f_test.shape)}")

    model = build_model(
        method,
        in_channels,
        out_channels,
        spatial_shape,
        args.hidden,
        ndim,
        modes=args.modes,
        depth=args.depth,
    )
    ctx.log(f"Model: {model.__class__.__name__}")
    ctx.log(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    if not args.eval_only:
        train_static_model(ctx, model, u_train, f_train, args)

    model.load_state_dict(torch.load(ctx.base_dir / "best_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    pred = predict_batches(model, u_test, args.test_batch_size)
    true = f_test.cpu()
    rel = rel_l2(pred, true)
    mse = torch.mean((pred - true) ** 2)
    ctx.log(f"Test Relative L2 error: {rel.item():.6e}")
    ctx.log(f"Test MSE error        : {mse.item():.6e}")
    torch.save(pred, ctx.base_dir / "test_f_pred.pt")
    torch.save(true, ctx.base_dir / "test_f_true.pt")
    ctx.log("All results saved successfully.")


def load_diffusion_train(folder):
    u_list = []
    ut_list = []
    dt_ref = None
    for file in sorted(Path(folder).glob("*.pt")):
        data = torch.load(file, map_location="cpu")
        u = data["u_train"].float()
        if "ut_train" in data:
            ut = data["ut_train"].float()
        elif "target_operator_train" in data:
            ut = data["target_operator_train"].float()
        else:
            ut = data["f_train"].float()
        t = data["t"]
        dt = float(t[1] - t[0])
        dt_ref = dt if dt_ref is None else dt_ref
        u_list.append(u)
        ut_list.append(ut)
    if not u_list:
        raise FileNotFoundError(f"No .pt train files in {folder}")
    min_t = min(u.shape[-1] for u in u_list)
    u_train = torch.cat([u[..., :min_t] for u in u_list], dim=0)
    ut_train = torch.cat([u[..., :min_t] for u in ut_list], dim=0)
    return u_train, ut_train, dt_ref


def load_diffusion_test(path):
    data = torch.load(path, map_location="cpu")
    u = data["u"].float()
    if "ut" in data:
        ut = data["ut"].float()
    elif "target_operator" in data:
        ut = data["target_operator"].float()
    else:
        ut = data["f"].float()
    t = data["t"]
    dt = float(t[1] - t[0])
    return u, ut, dt


def rollout_from_state(model, u0, dt, steps):
    states = [u0]
    cur = u0
    for _ in range(steps):
        cur = cur + dt * apply_diffusion_model(model, cur)
        states.append(cur)
    return torch.cat(states, dim=-1)


def apply_diffusion_model(model, x):
    if not isinstance(model, (GlobalMLP, DeepONetND)):
        return model(x)
    if x.ndim != 4:
        return model(x)

    batch, channels, nx, nt = x.shape
    frames = x.permute(0, 3, 1, 2).reshape(batch * nt, channels, nx)
    if isinstance(model, GlobalMLP):
        frames = frames.unsqueeze(-1)
    pred = model(frames)
    if pred.ndim == 4:
        pred = pred.squeeze(-1)
    return (
        pred.reshape(batch, nt, pred.shape[1], nx)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def rollout_time_marching(model, u_true, dt):
    model.eval()
    out = torch.zeros_like(u_true)
    out[..., 0] = u_true[..., 0]
    with torch.no_grad():
        for i in range(u_true.shape[-1] - 1):
            cur = out[..., i : i + 1]
            out[..., i + 1 : i + 2] = cur + dt * apply_diffusion_model(
                model, cur
            )
    return out


def run_diffusion_cli(method, out_dir, hidden=4):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=hidden)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--data_dir", type=str, default="train_data")
    parser.add_argument("--test_path", type=str, default="test_data/test.pt")
    parser.add_argument("--lambda_deriv", type=float, default=1.0)
    parser.add_argument("--lambda_roll", type=float, default=1.0)
    parser.add_argument("--max_rollout_steps", type=int, default=10)
    parser.add_argument("--num_rollout_starts", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.7)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--modes", type=int, default=8)
    args = parser.parse_args()

    ctx = RunContext(out_dir.format(hidden=args.hidden, method=method_label(method)))
    ctx.log(f"Device: {DEVICE}")
    ctx.log(f"Result directory: {ctx.base_dir}")

    u_train, ut_train, dt_train = load_diffusion_train(args.data_dir)
    u_test, ut_test, dt_test = load_diffusion_test(args.test_path)
    spatial_shape = tuple(u_train.shape[2:])
    label = method_label(method)
    if label == "DeepONet":
        model = DeepONetND(
            in_channels=1,
            out_channels=1,
            spatial_shape=(spatial_shape[0],),
            hidden=args.hidden,
            ndim=1,
            p=args.hidden,
            trunk_depth=2,
        ).to(DEVICE)
    else:
        model_spatial_shape = (
            (spatial_shape[0], 1) if label == "MLP" else spatial_shape
        )
        model = build_model(
            method,
            in_channels=1,
            out_channels=1,
            spatial_shape=model_spatial_shape,
            hidden=args.hidden,
            ndim=2,
            modes=args.modes,
            depth=4,
        ).to(DEVICE)
    ctx.log(f"Train u shape: {tuple(u_train.shape)}")
    ctx.log(f"Train ut shape: {tuple(ut_train.shape)}")
    ctx.log(f"Test u shape: {tuple(u_test.shape)}")
    ctx.log(f"Test ut shape: {tuple(ut_test.shape)}")
    ctx.log(f"Model: {model.__class__.__name__}")
    ctx.log(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    u_train = u_train.to(DEVICE)
    ut_train = ut_train.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=1,
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    loss_fn = nn.MSELoss()
    best = float("inf")
    wait = 0
    losses = []
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = apply_diffusion_model(model, u_train)
        deriv_loss = loss_fn(pred, ut_train)
        max_steps = min(args.max_rollout_steps, u_train.shape[-1] - 1)
        warm = max(1, int(args.epochs * args.warmup_ratio))
        steps = 1 + int(min(1.0, epoch / warm) * max(0, max_steps - 1))
        starts_all = list(range(max(1, u_train.shape[-1] - steps)))
        if args.num_rollout_starts < len(starts_all):
            starts = random.sample(starts_all, args.num_rollout_starts)
        else:
            starts = starts_all
        rollout_loss = 0.0
        for s in starts:
            u0 = u_train[..., s : s + 1]
            true_seg = u_train[..., s : s + steps + 1]
            pred_seg = rollout_from_state(model, u0, dt_train, steps)
            rollout_loss = rollout_loss + loss_fn(pred_seg[..., 1:], true_seg[..., 1:])
        rollout_loss = rollout_loss / max(1, len(starts))
        loss = args.lambda_deriv * deriv_loss + args.lambda_roll * rollout_loss
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        value = float(loss.item())
        losses.append(value)
        if value < best:
            best = value
            wait = 0
            torch.save(model.state_dict(), ctx.base_dir / "best_model.pth")
        else:
            wait += 1
        if epoch % args.print_every == 0:
            ctx.log(
                f"[{epoch:6d}] total = {value:.6e}, deriv = {deriv_loss.item():.6e}, "
                f"roll = {rollout_loss.item():.6e}, lr = {optimizer.param_groups[0]['lr']:.3e}"
            )
        if wait > args.patience:
            ctx.log(f"Early stopping at epoch {epoch}")
            break

    ctx.log(f"Training time: {time.time() - start:.2f} sec")
    np.savetxt(ctx.base_dir / "loss_total.txt", np.asarray(losses), fmt="%.12e")
    np.save(ctx.base_dir / "loss_history.npy", np.asarray(losses, dtype=np.float64))

    model.load_state_dict(torch.load(ctx.base_dir / "best_model.pth", map_location=DEVICE))
    u_test = u_test.to(DEVICE)
    ut_test = ut_test.to(DEVICE)
    with torch.no_grad():
        dudt_pred = apply_diffusion_model(model, u_test)
        rollout_pred = rollout_time_marching(model, u_test, dt_test)
    ctx.log(f"Test dudt Relative L2 error: {rel_l2(dudt_pred, ut_test).item():.6e}")
    ctx.log(f"Rollout Relative L2 error: {rel_l2(rollout_pred, u_test).item():.6e}")
    torch.save(dudt_pred.cpu(), ctx.base_dir / "test_dudt_pred.pt")
    torch.save(ut_test.cpu(), ctx.base_dir / "test_dudt_true.pt")
    torch.save(rollout_pred.cpu(), ctx.base_dir / "rollout_pred.pt")
    torch.save(u_test.cpu(), ctx.base_dir / "rollout_true.pt")
    ctx.log("All results saved successfully.")


def load_sturm_train(train_dir):
    u_list = []
    f_list = []
    for path in sorted(Path(train_dir).glob("*.pt")):
        data = torch.load(path, map_location="cpu")
        u_list.append(data["u_train"].float())
        f_list.append(data["f_train"].float())
    if not u_list:
        raise FileNotFoundError(f"No .pt train files in {train_dir}")
    return torch.cat(u_list, dim=0), torch.cat(f_list, dim=0)


def load_sturm_test(path):
    data = torch.load(path, map_location="cpu")
    return data["u"].float().unsqueeze(0).unsqueeze(0), data["f"].float().unsqueeze(0).unsqueeze(0)


def run_sturm_cli(method, out_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20000)
    parser.add_argument("--scheduler_patience", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_dir", type=str, default="train_data")
    parser.add_argument("--test_path", type=str, default="test_data/test.pt")
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--modes", type=int, default=16)
    args = parser.parse_args()

    ctx = RunContext(out_dir.format(hidden=args.hidden, method=method_label(method)))
    ctx.log(f"Device: {DEVICE}")
    ctx.log(f"Result directory: {ctx.base_dir}")
    u_train, f_train = load_sturm_train(args.train_dir)
    u_test, f_test = load_sturm_test(args.test_path)
    model = build_model(
        method,
        1,
        1,
        tuple(u_train.shape[2:]),
        args.hidden,
        ndim=1,
        modes=args.modes,
        depth=4,
    ).to(DEVICE)
    ctx.log(f"Train u shape: {tuple(u_train.shape)}")
    ctx.log(f"Train f shape: {tuple(f_train.shape)}")
    ctx.log(f"Test u shape: {tuple(u_test.shape)}")
    ctx.log(f"Test f shape: {tuple(f_test.shape)}")
    ctx.log(f"Model: {model.__class__.__name__}")
    ctx.log(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    u_train = u_train.to(DEVICE)
    f_train = f_train.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.scheduler_patience, factor=0.5
    )
    loss_fn = nn.MSELoss()
    best = float("inf")
    wait = 0
    losses = []
    interior = slice(1, -1)
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(u_train)
        loss = loss_fn(pred[:, :, interior], f_train[:, :, interior])
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step(loss.item())
        value = float(loss.item())
        losses.append(value)
        if value < best - 1e-12:
            best = value
            wait = 0
            torch.save(model.state_dict(), ctx.base_dir / "best_model.pth")
        else:
            wait += 1
        if epoch % args.print_every == 0:
            ctx.log(
                f"[{epoch:6d}] loss = {value:.6e} | "
                f"lr = {optimizer.param_groups[0]['lr']:.3e}"
            )
        if wait > args.patience:
            ctx.log(f"Early stopping at epoch {epoch}")
            break

    ctx.log(f"Training time: {time.time() - start:.2f} sec")
    np.savetxt(ctx.base_dir / "loss_total.txt", np.asarray(losses), fmt="%.12e")
    np.save(ctx.base_dir / "loss_history.npy", np.asarray(losses, dtype=np.float64))
    model.load_state_dict(torch.load(ctx.base_dir / "best_model.pth", map_location=DEVICE))
    model.eval()
    u_test = u_test.to(DEVICE)
    f_test = f_test.to(DEVICE)
    with torch.no_grad():
        pred = model(u_test)
    rel = rel_l2(pred[:, :, interior], f_test[:, :, interior])
    mse = torch.mean((pred[:, :, interior] - f_test[:, :, interior]) ** 2)
    ctx.log(f"Test Relative L2 error: {rel.item():.6e}")
    ctx.log(f"Test MSE error        : {mse.item():.6e}")
    torch.save(pred.cpu(), ctx.base_dir / "test_f_pred.pt")
    torch.save(f_test.cpu(), ctx.base_dir / "test_f_true.pt")
    ctx.log("All results saved successfully.")


def load_navier_dataset(folder, mode):
    data = torch.load(Path(folder) / f"{mode}.pt", map_location="cpu")
    u = data["u"].float()
    v = data["v"].float()
    gu = data["g_u"].float()
    gv = data["g_v"].float()
    dudt = data["dudt"].float()
    dvdt = data["dvdt"].float()
    vel = torch.cat([u, v], dim=2)
    g = torch.cat([gu, gv], dim=2)
    target = torch.cat([dudt, dvdt], dim=2)
    return vel, g, target, data.get("t", None)


def flatten_navier(vel, g, target):
    ns, nt, c, h, w = vel.shape
    return (
        vel.reshape(ns * nt, c, h, w),
        g.reshape(ns * nt, c, h, w),
        target.reshape(ns * nt, c, h, w),
    )


def navier_rollout(model, init_state, g_seq, dt):
    traj = [init_state.detach().clone()]
    cur = init_state.detach().clone()
    model.eval()
    with torch.no_grad():
        for i in range(g_seq.shape[0] - 1):
            local = model(cur.unsqueeze(0)).squeeze(0)
            cur = cur + dt * (local + g_seq[i])
            traj.append(cur.detach().clone())
    return torch.stack(traj, dim=0)


def run_navier_cli(method, out_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--scheduler_patience", type=int, default=300)
    parser.add_argument("--train_dir", type=str, default="dataset/train_data")
    parser.add_argument("--test_dir", type=str, default="dataset/test_data")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--modes", type=int, default=8)
    args = parser.parse_args()

    label = method_label(method)
    ctx = RunContext(out_dir.format(hidden=args.hidden, method=label))
    ctx.log(f"Device: {DEVICE}")
    ctx.log(f"Result directory: {ctx.base_dir}")

    with (ctx.base_dir / "config.txt").open("w", encoding="utf-8") as f:
        f.write(f"method: {label}\n")
        if label == "DeepONet":
            f.write(f"p: {args.hidden}\n")
            f.write(f"branch_width: {args.hidden}\n")
            f.write(f"trunk_hidden: {args.hidden}\n")
            f.write("trunk_depth: 4\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    vel_train, g_train, target_train, t_train = load_navier_dataset(args.train_dir, "train")
    vel_test, g_test, target_test, t_test = load_navier_dataset(args.test_dir, "test")
    v_train, g_train_flat, y_train = flatten_navier(vel_train, g_train, target_train)
    v_test, g_test_flat, y_test = flatten_navier(vel_test, g_test, target_test)
    model = build_model(
        method,
        2,
        2,
        tuple(v_train.shape[2:]),
        args.hidden,
        ndim=2,
        modes=args.modes,
        depth=4,
    ).to(DEVICE)
    ctx.log(f"Train velocity shape: {tuple(vel_train.shape)}")
    ctx.log(f"Test velocity shape: {tuple(vel_test.shape)}")
    ctx.log(f"Flattened train shape: {tuple(v_train.shape)}")
    ctx.log(f"Model: {model.__class__.__name__}")
    ctx.log(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    loader = DataLoader(
        TensorDataset(v_train, g_train_flat, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        threshold=1e-6,
        min_lr=1e-6,
    )
    loss_fn = nn.MSELoss()
    best = float("inf")
    wait = 0
    losses = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n_seen = 0
        for vb, gb, yb in loader:
            vb = vb.to(DEVICE, non_blocking=True)
            gb = gb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred_total = model(vb) + gb
            loss = loss_fn(pred_total, yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.item()) * vb.shape[0]
            n_seen += vb.shape[0]
        value = total / max(n_seen, 1)
        scheduler.step(value)
        losses.append(value)
        if value < best:
            best = value
            wait = 0
            torch.save(model.state_dict(), ctx.base_dir / "best_model.pth")
        else:
            wait += 1
        if epoch % args.print_every == 0:
            ctx.log(
                f"[{epoch:6d}] loss = {value:.6e} | "
                f"lr = {optimizer.param_groups[0]['lr']:.3e}"
            )
        if wait > args.patience:
            ctx.log(f"Early stopping at epoch {epoch}")
            break

    ctx.log(f"Training time: {time.time() - start:.2f} sec")
    np.savetxt(ctx.base_dir / "loss_total.txt", np.asarray(losses), fmt="%.12e")
    np.save(ctx.base_dir / "loss_history.npy", np.asarray(losses, dtype=np.float64))
    model.load_state_dict(torch.load(ctx.base_dir / "best_model.pth", map_location=DEVICE))
    model.eval()
    pred_list = []
    true_list = []
    test_loader = DataLoader(
        TensorDataset(v_test, g_test_flat, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    with torch.no_grad():
        for vb, gb, yb in test_loader:
            vb = vb.to(DEVICE)
            gb = gb.to(DEVICE)
            pred_list.append((model(vb) + gb).cpu())
            true_list.append(yb.cpu())
    dudt_pred = torch.cat(pred_list, dim=0)
    dudt_true = torch.cat(true_list, dim=0)
    ctx.log(f"One-step dudt Relative L2 error: {rel_l2(dudt_pred, dudt_true).item():.6e}")
    ctx.log(f"One-step dudt MSE error        : {torch.mean((dudt_pred - dudt_true) ** 2).item():.6e}")
    torch.save(dudt_pred, ctx.base_dir / "test_dudt_pred.pt")
    torch.save(dudt_true, ctx.base_dir / "test_dudt_true.pt")

    if t_test is None or len(t_test) < 2:
        raise ValueError("Navier dataset must contain t with at least two steps")
    dt = float((t_test[1] - t_test[0]).item())
    vel_test_gpu = vel_test.to(DEVICE)
    g_test_gpu = g_test.to(DEVICE)
    rollouts = []
    for sample in range(vel_test_gpu.shape[0]):
        rollout = navier_rollout(model, vel_test_gpu[sample, 0], g_test_gpu[sample], dt)
        rollouts.append(rollout.cpu())
        ctx.log(f"Rollout sample {sample:03d} done.")
    rollout_pred = torch.stack(rollouts, dim=0)
    rollout_true = vel_test.cpu()
    ctx.log(f"Rollout Relative L2 error: {rel_l2(rollout_pred, rollout_true).item():.6e}")
    ctx.log(f"Rollout MSE error        : {torch.mean((rollout_pred - rollout_true) ** 2).item():.6e}")
    torch.save(rollout_pred, ctx.base_dir / "rollout_pred.pt")
    torch.save(rollout_true, ctx.base_dir / "rollout_true.pt")
    ctx.log("All results saved successfully.")
