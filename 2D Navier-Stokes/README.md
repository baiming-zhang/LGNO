# 2D incompressible Navier–Stokes

## 1. Problem and decomposition

The generator evolves one periodic vorticity trajectory and recovers velocity
from the stream function. The learning equations are represented as:

$$
u_t + u u_x + v u_y = \nu\Delta u + g_u,
$$

$$
v_t + u v_x + v v_y = \nu\Delta v + g_v.
$$

The current generated trajectory uses zero external forcing, so the stored
`g=(g_u,g_v)` represents the pressure-gradient contribution. LGNO learns the
local derivative term and adds `g` outside the network. Prediction uses Euler
rollout.

## 2. Files

| File | Purpose |
|---|---|
| `generate_data.py` | periodic vorticity simulation and derived velocity data |
| `LGNO.py`, `_lgno_impl.py` | launcher and preserved vector LGNO |
| `MLP.py`, `MLPConv.py`, `CNN.py` | baseline launchers |
| `DeepONet.py`, `oneshot.py` | DeepONet and OneShotPDE |
| `FNO.py`, `CNO.py`, `GNO.py` | operator baselines |

## 3. Data generation

```bash
python generate_data.py
```

Defaults:

| Quantity | Value |
|---|---:|
| Grid | 32×32 |
| Domain | `[-1,1)²` |
| Total frames | 1,000 |
| Time step | 0.002 |
| Simulated duration | 1.998 |
| Viscosity | 0.002 |
| Number of trajectories | 1 |
| Training frames | first 20 |
| Test frames | full 1,000 |
| Initial-field seed | 1234 |
| Global seed | 42 |
| Time integrator | explicit Euler |

Expected files:

```text
dataset/train_data/train.pt
dataset/test_data/test.pt
dataset/preview/
```

### Tensor format

Both files contain:

```text
omega, u, v, g_u, g_v, dudt, dvdt, x, y, t, metadata
```

| Split | Field tensor shape | Coordinate/time shapes |
|---|---|---|
| train | `[1,20,1,32,32]` | `x[32]`, `y[32]`, `t[20]` |
| test | `[1,1000,1,32,32]` | `x[32]`, `y[32]`, `t[1000]` |

The trainer concatenates `u,v`, `g_u,g_v`, and `dudt,dvdt`, producing:

```text
velocity: [Ns,Nt,2,32,32]
g:        [Ns,Nt,2,32,32]
dudt:     [Ns,Nt,2,32,32]
```

It then flattens time into 20 training samples and 1,000 test samples.

## 4. LGNO model

`VectorStencilLGNO` uses a periodic 3×3 stencil for two velocity channels. A
local MLP generates output-channel-specific stencil weights. The predicted
local derivative is added to the known `g` field:

```text
(u,v) local 3×3 stencil
    -> MLP-generated local coefficients
    -> learned local derivative
    + g
    -> total derivative
    -> repeated Euler rollout
```

Default:

```bash
python LGNO.py
```

Top-level:

```bash
python ../run_lgno.py navier-stokes-2d
```

Native defaults:

```text
hidden=16
epochs=20000
lr=1e-3
patience=2000
batch_size=8
train_dir=dataset/train_data
test_dir=dataset/test_data
save_stride=50
```

Explicit output/evaluation:

```bash
python LGNO.py --run_dir LGNO_16hidden_results/run_manual
python LGNO.py --eval_only --run_dir LGNO_16hidden_results/run_manual
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

Baseline defaults are hidden 16, 20,000 epochs, learning rate `1e-3`,
patience 2,000, batch size 8, scheduler patience 300, gradient clipping 1.0,
and mode count 8.

## 6. Output

A completed LGNO run includes:

- `config.txt`, `train_log.txt`;
- `best_model.pth`;
- `loss_history.npy`, `loss_total.txt`;
- `test_dudt_pred.pt`, `test_dudt_true.pt`;
- `rollout_pred.pt`, `rollout_true.pt`;
- rollout prediction/truth/error figures every `save_stride` frames.

`rollout_pred.pt` and `rollout_true.pt` have shape `[1,1000,2,32,32]`.

## 7. Detailed LGNO results

| Hidden | Parameters | Final loss | Best loss | One-step derivative L2 | One-step derivative MSE | Rollout L2 | Rollout MSE | Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 548 | 8.16954e-07 | 7.89089e-07 | 0.0178212 | 7.48457e-06 | 0.0102719 | 5.09942e-06 | 75.64 s |
| 16 | 1,188 | 3.37057e-07 | 3.32386e-07 | 0.0138458 | 4.51784e-06 | 0.00813705 | 3.19998e-06 | 169.55 s |
| 32 | 2,852 | 2.50490e-07 | 2.36442e-07 | 0.0143441 | 4.84894e-06 | 0.00832833 | 3.35219e-06 | 78.65 s |

All runs completed 20,000 epochs. The final learning rates were:

- hidden 8: `5.0e-4`;
- hidden 16: `2.5e-4`;
- hidden 32: `2.5e-4`.

The hidden-16 run gives the lowest recorded rollout error.

## 8. Complete benchmark table

`Test L2` and `Test MSE` use the full rollout metric.

| Model | Hidden | Parameters | Final train loss | Test L2 | Test MSE |
|---|---:|---:|---:|---:|---:|
| LGNO | 16 | 1,188 | 3.37057e-07 | 0.00813705 | 3.19998e-06 |
| LGNO | 32 | 2,852 | 2.50490e-07 | 0.00832833 | 3.35219e-06 |
| LGNO | 8 | 548 | 8.16954e-07 | 0.0102719 | 5.09942e-06 |
| MLPConv | 16 | 610 | 2.88159e-06 | 0.0173972 | 1.46275e-05 |
| MLPConv | 32 | 1,730 | 4.94775e-06 | 0.0197447 | 1.88415e-05 |
| GNO | 16 | 4,978 | 1.96396e-05 | 0.0296636 | 4.25269e-05 |
| CNN | 16 | 4,978 | 6.69566e-07 | 0.0328851 | 5.22652e-05 |
| CNO | 16 | 18,674 | 3.47471e-07 | 0.0818385 | 3.23687e-04 |
| DeepONet | 8 | 178,930 | 0.00231963 | 0.627836 | 0.0190506 |
| DeepONet | 16 | 34,738 | 0.00449103 | 1.01447 | 0.0497390 |
| MLP | 16 | 67,872 | 5.18211e-08 | 1.33909 | 0.0866629 |
| FNO | 16 | 67,010 | 2.27698e-07 | 5.14054 | 1.27709 |
| OneShotPDE | 16 | 1,170 | 3.13535e-08 | inf | inf |

`inf` indicates numerical divergence during the archived rollout, not a
missing value.

## 9. Expected behavior and checks

- `Detected dt` should be `2.000000e-03`.
- Flattened train/test shapes should be `(20,2,32,32)` and
  `(1000,2,32,32)`.
- One-step and rollout tensors must remain finite.
- The rollout must use the true first frame and known `g_t` sequence.
- A successful run ends with `All results saved successfully.`
- Do not confuse a low one-step training loss with stable long-horizon rollout;
  both metrics must be checked.
