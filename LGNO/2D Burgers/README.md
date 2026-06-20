# 2D Burgers-type static operator

## 1. Problem

This nonlinear static benchmark learns:

$$
f = u\,u_x + u\,u_y - \nu \Delta u,
\qquad \nu=0.01,
$$

on a 512×512 grid over `[-1,1)²`. Derivatives in the generator and the LGNO
stencil are periodic.

The saved metadata historically labels the field boundary as
`soft_zero_boundary`, while the generator and submitted model use
`torch.roll` periodic finite differences. For reproducibility, the actual code
is authoritative.

## 2. Files

| File | Purpose |
|---|---|
| `generate_data.py` | Generates five nonlinear scalar fields |
| `LGNO.py`, `_lgno_impl.py` | launcher and preserved LGNO |
| `MLP.py`, `MLPConv.py`, `CNN.py` | dense/local/convolution baselines |
| `DeepONet.py`, `oneshot.py` | DeepONet and OneShotPDE |
| `FNO.py`, `CNO.py`, `GNO.py` | neural-operator baselines |

The baseline launchers use the top-level `aligned_baseline_common.py`.

## 3. Generate data

```bash
python generate_data.py
```

Defaults:

| Quantity | Value |
|---|---:|
| Grid | 512×512 |
| Domain | `[-1,1) × [-1,1)` |
| Domain length | 2 |
| `dx=dy` | 2/512 |
| Viscosity | 0.01 |
| Total samples | 5 |
| Train/test split | 1 / 4 |
| Field seeds | 1234–1238 |
| Global seed | 42 |

The random field is a sum of signed periodic Gaussian structures. The target
is computed from full-domain periodic first and second differences.

Expected files:

```text
dataset/train_data/train.pt
dataset/test_data/test.pt
dataset/preview/*.png
```

### Tensor format

| Split | `u` | `f` | `x`,`y` |
|---|---|---|---|
| train | `[1,1,512,512]` float32 | `[1,1,512,512]` | `[512]` |
| test | `[4,1,512,512]` float32 | `[4,1,512,512]` | `[512]` |

Each file also contains a metadata dictionary describing the operator, domain,
and split.

## 4. LGNO

The model extracts the periodic 3×3 stencil, predicts nine local coefficients
with a three-layer SiLU MLP, and computes their dot product with the same local
stencil. Default:

```bash
python LGNO.py
```

Top-level equivalent:

```bash
python ../run_lgno.py burgers-2d
```

Native defaults:

```text
hidden=8
epochs=100000
lr=1e-3
patience=3000
train_dir=dataset/train_data
test_dir=dataset/test_data
eval_only=false
```

An explicit run directory must be visible before the numerical script starts:

```bash
python LGNO.py --run_dir LGNO_8hidden_results/run_manual
```

Evaluation-only mode:

```bash
python LGNO.py --eval_only --run_dir LGNO_8hidden_results/run_manual
```

## 5. Baselines

```bash
python MLP.py
python MLPConv.py
python CNN.py
python DeepONet.py
python oneshot.py
python FNO.py
python CNO.py
python GNO.py
```

All use the same train/test tensors and save a timestamped result directory.
The case defaults are hidden 8, 100,000 epochs, learning rate `1e-3`, patience
3,000, batch size 1, and `ReduceLROnPlateau`.

## 6. Output

A completed result directory normally contains:

```text
config.txt
train_log.txt
best_model.pth
loss_history.npy
loss_total.txt
test_f_pred.pt
test_f_true.pt
figures/
```

`test_f_pred.pt` and `test_f_true.pt` have shape `[4,1,512,512]`.

## 7. LGNO runs

| Hidden | Parameters | Final loss | Best loss | Test L2 | Test MSE | Training time | Final LR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 105 | 5.07614e-03 | 5.07515e-03 | 0.0495040 | 0.787441 | 114.01 s | 1.25e-04 |
| 8 | 233 | 6.44643e-03 | 5.87727e-03 | 0.0928579 | 2.770606 | 227.41 s | 3.125e-05 |
| 16 | 585 | 3.62487e-03 | 3.60654e-03 | 0.228290 | 16.74595 | 129.31 s | 1.563e-05 |

All three archived runs completed all 100,000 configured epochs.

### Per-test-sample relative L2

The four columns below follow the saved test order.

| Hidden | Sample 0 | Sample 1 | Sample 2 | Sample 3 |
|---:|---:|---:|---:|---:|
| 4 | 0.049421 | 0.053511 | 0.012914 | 0.018899 |
| 8 | 0.100740 | 0.095210 | 0.014972 | 0.029719 |
| 16 | 0.249131 | 0.233992 | 0.017404 | 0.043659 |

The global L2 is not the arithmetic mean of per-sample L2 values; it is
computed from the concatenated tensor norm.

## 8. Complete benchmark table

| Model | Hidden | Parameters | Final train loss | Test L2 | Test MSE |
|---|---:|---:|---:|---:|---:|
| LGNO | 4 | 105 | 0.00507614 | 0.049504 | 0.787441 |
| LGNO | 8 | 233 | 0.00644643 | 0.0928579 | 2.77061 |
| LGNO | 16 | 585 | 0.00362487 | 0.228290 | 16.7460 |
| MLPConv | 8 | 161 | 0.00264519 | 0.353949 | 40.2609 |
| MLPConv | 4 | 65 | 0.00375383 | 0.380223 | 46.4602 |
| OneShotPDE | 16 | 721 | 0.00270036 | 0.385136 | 47.6612 |
| MLPConv | 16 | 449 | 0.00224271 | 0.385750 | 47.8134 |
| OneShotPDE | 4 | 85 | 0.0203681 | 0.463480 | 69.0420 |
| CNN | 8 | 1,257 | 0.00865082 | 0.480102 | 74.0741 |
| OneShotPDE | 8 | 233 | 0.00800632 | 0.611445 | 120.138 |
| GNO | 4 | 373 | 0.00657382 | 0.628043 | 126.746 |
| FNO | 8 | 74,209 | 6.55579e-06 | 0.895329 | 257.574 |
| FNO | 4 | 9,337 | 8.47169e-06 | 0.917616 | 270.545 |
| CNN | 4 | 341 | 0.0267480 | 0.925499 | 275.259 |
| DeepONet | 8 | 175,585 | 0.0987899 | 1.00256 | 322.964 |
| CNO | 8 | 4,865 | 0.00760952 | 1.35457 | 589.576 |
| CNO | 4 | 1,205 | 0.0310016 | 1.42208 | 649.809 |
| GNO | 8 | 1,473 | 0.00769397 | 1.63996 | 864.182 |
| DeepONet | 4 | 1,048,713 | 10.3643 | 1.93056 | 1,197.52 |
| MLP | 8 | 4,456,528 | 5.47211e-12 | 3.19835 | 3,286.68 |
| MLP | 4 | 2,359,320 | 2.91360e-15 | 96.1917 | 2,973,307 |

## 9. Expected behavior

- Generated train/test shapes must be exactly those listed above.
- Training loss and test metrics must remain finite.
- A completed run prints `All results saved successfully.`
- Lower training loss does not necessarily imply better held-out error; the
  dense MLP rows illustrate severe overfitting.
- The archived best LGNO result is hidden 4, not the largest model.
- Exact wall-clock time varies; use the saved metrics and tensors for accuracy
  validation.
