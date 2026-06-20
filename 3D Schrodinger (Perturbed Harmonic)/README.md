# 3D perturbed-harmonic Schrödinger operator

## 1. Problem

The generator evolves:

$$
i\frac{\partial\psi}{\partial t}
= -\frac{1}{2}\Delta\psi + V\psi
$$

with:

$$
V(x,y,z)=
\frac{1}{2}
(\omega_x^2x^2+\omega_y^2y^2+\omega_z^2z^2)
+\alpha\sin(2x)\cos(3y)\sin(z).
$$

Parameters:

```text
omega_x=1.0
omega_y=1.5
omega_z=0.8
alpha=0.3
```

The initial state is a Gaussian wave packet with `sigma=0.8` and carrier
wave number `k0=2.0`.

## 2. Files

| File | Purpose |
|---|---|
| `generate_data.py` | full 3D complex trajectory generation |
| `LGNO.py`, `_lgno_impl.py` | launcher and preserved 3D LGNO |
| `MLP.py`, `MLPConv.py`, `CNN.py` | baseline launchers |
| `DeepONet.py`, `oneshot.py` | DeepONet and OneShotPDE |
| `FNO.py`, `CNO.py`, `GNO.py` | neural-operator baselines |

## 3. Generate data

```bash
python generate_data.py
```

Defaults:

| Quantity | Value |
|---|---:|
| Grid | 16×16×16 |
| Domain | `[-2,2]³` |
| Time frames | 5,000 |
| Time step | 1e-4 |
| Training frames | first 1,000 |
| Test frames | full 5,000 |
| Boundary | periodic finite difference |
| Integrator | explicit Euler |
| Global seed | 42 |

Expected files:

```text
dataset/train_data/train.pt
dataset/test_data/test.pt
dataset/preview/
```

The train and test files are approximately 0.41 GB each.

### Tensor format

| Key | Train | Test | Meaning |
|---|---|---|---|
| `u` | `[1,3,16,16,16,1000]` | `[1,3,16,16,16,5000]` | `[Re ψ, Im ψ, V]` |
| `f` | `[1,2,16,16,16,1000]` | `[1,2,16,16,16,5000]` | `[Re Hψ, Im Hψ]` |
| `x`,`y`,`z` | `[16]` | `[16]` | coordinates |
| `t` | `[1000]` | `[5000]` | time |
| `metadata` | dictionary | dictionary | equation, split, `dt`, boundary |

The trainer reshapes the time axis into the sample axis, giving 1,000 train
samples and 5,000 test samples of shape `[C,16,16,16]`.

## 4. LGNO

`U2GPEOperator` uses a periodic seven-point three-dimensional stencil. Local
features generate output coefficients for the two `Hpsi` channels.

```bash
python LGNO.py
```

Top-level:

```bash
python "../run_lgno.py" perturbed-harmonic-3d
```

Defaults:

```text
hidden=16
epochs=100000
lr=1e-2
patience=3000
batch_size=8
test_batch_size=8
train_dir=dataset/train_data
test_dir=dataset/test_data
```

The implementation automatically creates:

```text
LGNO_<hidden>hidden_results/run_<timestamp>/
```

It reads `--hidden` before output-directory initialization so the folder name
matches the requested width.

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

Case defaults are hidden 16, 100,000 epochs, learning rate `1e-2`, patience
3,000, batch/test batch size 8, scheduler patience 500, gradient clipping 1.0,
and three-dimensional mode count 6.

## 6. Output

A completed run contains:

- `config.txt`, `train_log.txt`;
- `best_model.pth`;
- `loss_history.npy`, `loss_total.txt`;
- `test_f_pred.pt`, `test_f_true.pt`;
- prediction/truth/error figures from the middle z slice;
- training loss figures.

Prediction and truth have shape `[5000,2,16,16,16]` after time-to-sample
reshaping. Generating and saving them requires substantial RAM and disk space.

## 7. Detailed LGNO result

| Quantity | Value |
|---|---:|
| Hidden | 16 |
| Parameters | 988 |
| Configured/completed epochs | 100,000 / 100,000 |
| Final train loss | 9.78988e-10 |
| Minimum train loss | 9.73804e-10 |
| Final learning rate | 4.883e-06 |
| Test relative L2 | 0.003130184 |
| Test MSE | 2.961123e-06 |
| Re(Hψ) relative L2 | 0.003318882 |
| Im(Hψ) relative L2 | 0.002906693 |
| Re(Hψ) MSE | 3.352740e-06 |
| Im(Hψ) MSE | 2.569506e-06 |
| Training time | 53,905.90 s = 14.97 h |

The recorded time is from the archived CUDA run and is not a universal
runtime estimate.

## 8. Complete benchmark table

| Model | Hidden | Parameters | Final train loss | Test L2 | Test MSE |
|---|---:|---:|---:|---:|---:|
| LGNO | 16 | 988 | 9.78988e-10 | 0.00313018 | 2.96112e-06 |
| MLPConv | 16 | 1,618 | 9.56337e-07 | 0.0193730 | 1.13696e-04 |
| OneShotPDE | 16 | 1,890 | 7.97541e-07 | 0.0273235 | 2.26199e-04 |
| GNO | 16 | 5,042 | 1.97334e-07 | 0.0601483 | 0.00109628 |
| CNN | 16 | 15,202 | 4.03096e-07 | 0.100256 | 0.00304114 |
| FNO | 16 | 222,690 | 1.82331e-08 | 0.181809 | 0.0100000 |
| CNO | 16 | 55,570 | 0.304049 | 0.999735 | 0.304151 |
| DeepONet | 16 | 198,594 | 0.00844735 | 1.12194 | 0.381399 |
| MLP | 16 | 336,160 | 0.00843011 | 1.66744 | 0.841670 |

## 9. Expected behavior and resource checks

- Data generation prints finite wavefunction amplitude and mass diagnostics.
- Train/test time lengths must be 1,000 and 5,000.
- Input/output channel counts must be 3 and 2.
- Training may require many hours; test a reduced `--epochs` value only as a
  smoke test, not as a reproduction of the reported result.
- CUDA out-of-memory errors can be addressed by reducing
  `--batch_size`/`--test_batch_size`; this changes throughput but not model
  definition.
- A successful full run prints `All results saved successfully.`
