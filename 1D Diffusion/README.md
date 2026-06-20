# 1D periodic diffusion

## 1. Problem

The generator solves the periodic discrete heat equation

$$
\frac{d u_i}{dt}
= \kappa\frac{u_{i-1}-2u_i+u_{i+1}}{\Delta x^2},
\qquad \kappa=0.1,
$$

using forward Euler time stepping and a periodic three-point central
difference. The learning task is to predict `du/dt` locally and then repeatedly
apply the learned derivative in a rollout.

## 2. Files

| File | Purpose |
|---|---|
| `generate_data.py` | Generates deterministic train/test trajectories |
| `LGNO.py` | Common-interface launcher |
| `_lgno_impl.py` | Preserved LGNO numerical implementation |
| `MLP.py` | MLP baseline |
| `MLPConv.py` | local MLPConv baseline |
| `CNN.py` | convolutional baseline |
| `DeepONet_4hidden.py` | DeepONet width 4 |
| `DeepONet_16hidden.py` | DeepONet width 16 |
| `DeepONet_64hidden.py` | DeepONet width 64 |
| `oneshot.py` | OneShotPDE baseline |
| `FNO.py`, `CNO.py`, `GNO.py` | operator baselines |

Baseline files call `../aligned_baseline_common.py`.

## 3. Data generation

Run:

```bash
python generate_data.py
```

Default settings:

| Quantity | Value |
|---|---:|
| Spatial points | 32 |
| Full time points | 300 |
| Learned time intervals | 299 |
| Spatial domain | periodic `[0,1)` |
| Time interval | `linspace(0,1,300)` |
| `dx` | 0.03125 |
| `dt` | 0.0033444815780967474 |
| Diffusivity `κ` | 0.1 |
| Train trajectories | 1 |
| Test trajectories | 6 |
| Generator seed | 42 |
| Training noise | 0 |

Expected files:

```text
train_data/train.pt
test_data/test.pt
```

### Tensor format

`train.pt` contains:

| Key | Shape/type | Meaning |
|---|---|---|
| `u_train` | `[1,1,32,299]` float32 | state at the start of each Euler interval |
| `ut_train` | `[1,1,32,299]` | exact discrete derivative |
| `uxx_train` | `[1,1,32,299]` | periodic discrete Laplacian |
| `target_operator_train` | `[1,1,32,299]` | operator target |
| `residual_train` | `[1,1,32,299]` | discrete PDE residual |
| `u0_train` | `[1,1,32]` | initial condition |
| `x` | `[32]` | spatial coordinates |
| `t` | `[299]` | interval start times |
| `t_full` | `[300]` | full time grid |
| scalar/string metadata | — | `kappa`, equation, boundary and discretization |

`test.pt` uses keys `u`, `ut`, `uxx`, `target_operator`, `residual`, and `u0`
with six samples: `[6,1,32,299]`.

## 4. LGNO model and training

The LGNO uses a periodic three-point stencil:

```text
[u(i-1), u(i), u(i+1)]
        -> three-layer SiLU MLP
        -> three local coefficients
        -> coefficient-weighted stencil sum
```

The output is scaled by the original fixed factor `0.1 * 32 * 32`. Training
combines:

- supervised derivative MSE;
- differentiable rollout MSE;
- curriculum rollout length from 1 to 10;
- gradient clipping.

Default command:

```bash
python LGNO.py
```

Equivalent top-level command:

```bash
python ../run_lgno.py diffusion-1d
```

Important native options:

```text
--hidden 4
--epochs 20000
--lr 1e-3
--patience 500
--data_dir train_data
--test_path test_data/test.pt
--save_dir LGNO_4hidden_results
--lambda_deriv 1.0
--lambda_roll 1.0
--max_rollout_steps 10
--num_rollout_starts 2
--warmup_ratio 0.7
--grad_clip 1.0
```

Example:

```bash
python LGNO.py --hidden 2 --save_dir LGNO_2hidden_results
```

## 5. Baseline execution

```bash
python MLP.py
python MLPConv.py
python CNN.py
python DeepONet_4hidden.py
python DeepONet_16hidden.py
python DeepONet_64hidden.py
python oneshot.py
python FNO.py
python CNO.py
python GNO.py
```

The common baseline CLI supports the same main training/data arguments and
rollout-loss settings.

## 6. Output artifacts

The LGNO output directory contains:

- `best_model.pth`;
- `loss_total.txt`;
- `training_log.txt`;
- training, derivative, rollout, and slice figures under `figure/`.

The log reports train derivative relative L2, test derivative relative L2,
and full rollout relative L2.

## 7. Recorded LGNO results

| Hidden | Parameters | Epochs completed | Final loss | Best loss | Train derivative L2 | Test derivative L2 | Rollout L2 | Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 23 | 11,311 | 5.89139e-04 | 1.98823e-04 | 2.35299e-03 | 2.98547e-03 | 1.20931e-02 | 73.37 s |
| 4 | 51 | 9,970 archived loss points | 2.39489e-03 | 1.02519e-03 | 3.28851e-03 | 1.88952e-02 | 2.47799e-02 | 167.15 s |
| 8 | 131 | 5,500 | 1.24903e-02 | 4.82267e-03 | 8.16325e-03 | 8.88721e-02 | 5.50473e-02 | 26.84 s |

The historical hidden-4 output directory was named
`PDEINN_4hidden_results`; its logged architecture and parameter count match the
submitted local-coefficient implementation. The package does not rename or
reinterpret that archived artifact.

## 8. Complete benchmark table

`Test L2` is the primary rollout relative L2. `Test MSE` is the corresponding
benchmark MSE derived from the saved predictions.

| Model | Hidden | Parameters | Final train loss | Test L2 | Test MSE |
|---|---:|---:|---:|---:|---:|
| LGNO | 2 | 23 | 5.89139e-04 | 0.0120931 | 2.57458e-05 |
| LGNO | 4 | 51 | 2.39489e-03 | 0.0247799 | 1.08101e-04 |
| CNN | 2 | 99 | 0.184054 | 0.0467644 | 3.85001e-04 |
| MLPConv | 4 | 41 | 0.0136092 | 0.0533175 | 5.00462e-04 |
| LGNO | 8 | 131 | 0.0124903 | 0.0550472 | 5.33461e-04 |
| CNO | 2 | 315 | 0.0168351 | 0.437714 | 0.0337297 |
| GNO | 4 | 373 | 2.11535 | 0.605379 | 0.0645188 |
| OneShotPDE | 4 | 85 | 0.126350 | 0.857034 | 0.129308 |
| MLPConv | 2 | 29 | 27.3059 | 0.947030 | 0.157891 |
| MLP | 4 | 312 | 0.00298914 | 0.993794 | 0.173870 |
| GNO | 2 | 115 | 22.5375 | 0.996923 | 0.174966 |
| CNN | 4 | 341 | 95.9924 | 1.90755 | 0.640592 |
| CNO | 4 | 1,205 | 6.89358 | 2.57929 | 1.17120 |
| OneShotPDE | 2 | 35 | 74.0601 | 3.06813 | 1.65721 |
| DeepONet | 16 | 1,649 | 161.199 | 5.23946 | 4.83286 |
| DeepONet | 4 | 221 | 165.298 | 5.61466 | 5.54981 |
| MLP | 2 | 168 | 7.93998 | 10.6713 | 20.0476 |
| DeepONet | 2 | 95 | 178.280 | 23.4921 | 97.1573 |
| FNO | 2 | 1,065 | 7.74778 | 125.772 | 2,784.81 |
| FNO | 4 | 4,217 | 2.32879 | 1.582059e+06 | 4.40633e+11 |
| DeepONet | 64 | 18,881 | 155.056 | 4.531121e+06 | 3.61445e+12 |

## 9. Expected behavior and checks

- Generator residuals should be close to floating-point discretization error.
- Shapes printed by training must match the table above.
- Loss should remain finite.
- `best_model.pth` must be loadable.
- Rollout prediction should start exactly from the true initial condition.
- The submitted hidden-4 default follows the numerical implementation in
  `_lgno_impl.py`.
