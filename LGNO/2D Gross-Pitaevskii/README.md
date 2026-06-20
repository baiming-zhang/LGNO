# 2D Gross–Pitaevskii operator

## 1. Problem

The target is the nonlinear Hamiltonian action:

$$
H\psi =
-\frac{\hbar^2}{2m}\Delta\psi
+ V\psi
+ g|\psi|^2\psi.
$$

The submitted generator uses:

```text
hbar2_over_2m = 1.0
g = 0.05
strictly periodic 64×64 domain
```

Input channels:

```text
[Re(psi), Im(psi), V]
```

Output channels:

```text
[Re(Hpsi), Im(Hpsi)]
```

Training uses one optical-lattice potential. Testing uses four different
potential families: `hex`, `deformed`, `bichromatic`, and `disorder`.

## 2. Files

| File | Purpose |
|---|---|
| `generate_data.py` | periodic complex-field and potential generator |
| `LGNO.py`, `_lgno_impl.py` | folded/shared LGNO launcher and implementation |
| `LGNO_unfolded.py`, `_lgno_unfolded_impl.py` | unfolded/separate-branch LGNO |
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
| Grid | 64×64 |
| Domain | `[0,1)²` periodic |
| Train samples | 1 |
| Test samples | 4 |
| Training potential | optical |
| Test potentials | hex, deformed, bichromatic, disorder |
| `g` | 0.05 |
| kinetic coefficient | 1.0 |
| max Fourier mode | 3 in each direction |
| real/imaginary modes | 8 / 8 |
| global seed | 42 |
| train seed offset | 1000 |
| test offsets | 2000, 3000, 4000, 5000 |

Expected files:

```text
dataset/train_data/train_dataset.pt
dataset/test_data/test_dataset.pt
dataset/train_data/preview/
dataset/test_data/preview/
```

### Tensor format

| Key | Train shape | Test shape | Meaning |
|---|---|---|---|
| `u` | `[1,3,64,64]` | `[4,3,64,64]` | `[Re ψ, Im ψ, V]` |
| `f` | `[1,2,64,64]` | `[4,2,64,64]` | `[Re Hψ, Im Hψ]` |
| `psi` | `[1,2,64,64]` | `[4,2,64,64]` | complex wavefunction channels |
| `V` | `[1,64,64]` | `[4,64,64]` | potential |
| `x`,`y` | `[64]` | `[64]` | periodic coordinates |
| `labels` | `["optical"]` | four potential names | sample identity |

The saved `f` tensors were verified to match the true tensors used by both
folded and unfolded archived runs exactly.

## 4. Folded and unfolded LGNO

Both models:

- compute train-set channel statistics;
- use periodic 5-point or 9-point stencils;
- pass normalized local stencil values, center potential, and center
  `|psi|²` to an MLP;
- predict normalized stencil coefficients;
- return physical `Hpsi` after inverse normalization;
- add no explicit analytic `V*psi` term.

Folded/shared:

```bash
python LGNO.py --variant folded
```

Unfolded/separate:

```bash
python LGNO_unfolded.py --variant unfolded
```

Top-level equivalents:

```bash
python ../run_lgno.py gpe-2d-folded
python ../run_lgno.py gpe-2d-unfolded
```

Defaults:

```text
hidden=12
depth=3
stencil=5pt
dropout=0
last_scale=1e-3
epochs=2000000
lr=1e-3
min_lr=1e-6
weight_decay=1e-6
grad_clip=1
patience=10000
scheduler_patience=2000
log_every=200
seed=42
save_weights=1
```

Recommended explicit commands:

```bash
python LGNO.py --hidden 12 --out_dir LGNO_12hidden_results
python LGNO_unfolded.py --hidden 12 --out_dir LGNO_unfolded_12hidden_results
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

Baseline defaults follow the LGNO comparison budget: hidden 12, two million
epochs, learning rate `1e-3`, patience 10,000, weight decay `1e-6`, and
gradient clipping 1.0.

## 6. Output

A completed LGNO run contains:

- `config.txt`, `train_log.txt`;
- `best_model.pth`, `loss_history.npy`, `loss_total.txt`;
- `train_f_pred.pt`, `train_f_true.pt`;
- `test_f_pred.pt`, `test_f_true.pt`;
- prediction/truth/error figures;
- optional diagnostic local-weight maps.

## 7. All recorded LGNO configurations

| Variant | Hidden | Parameters | Final loss | Best loss | Test L2 | Test MSE | Re L2 | Im L2 | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| folded | 6 | 154 | 2.56971e-05 | 2.56880e-05 | 1.282421 | 2438.522 | 1.698637 | 0.285807 | 3900.65 s |
| folded | 12 | 370 | 2.51039e-05 | 2.50895e-05 | 0.116089 | 19.98254 | 0.0746866 | 0.153056 | 7589.10 s |
| folded | 24 | 1,018 | 2.39322e-04 | 2.39322e-04 | 0.282600 | 118.4158 | 0.232535 | 0.335185 | 4335.95 s |
| unfolded | 6 | 224 | 9.63442e-05 | 9.63441e-05 | 0.276925 | 113.7079 | 0.292734 | 0.255624 | 4653.46 s |
| unfolded | 12 | 500 | 2.83618e-05 | 2.64385e-05 | 0.378591 | 212.5229 | 0.446228 | 0.270293 | 6667.06 s |
| unfolded | 24 | 1,268 | 1.53788e-05 | 1.53784e-05 | 0.431434 | 275.9904 | 0.459407 | 0.393377 | 4800.70 s |

All configurations completed two million epochs.

### Legacy archived run

The consolidated archive also contains an older hidden-12 normalized-residual
LGNO run under the historical `old/` directory:

| Parameters | Final/best loss | Test L2 | Test MSE | Re L2 | Im L2 | Time |
|---:|---:|---:|---:|---:|---:|---:|
| 358 | 8.93002e-03 | 0.616378 | 682.413 | 0.459270 | 0.770550 | 11,066.07 s |

That result predates the submitted folded/unfolded implementation and is
reported for archival completeness only. It should not be presented as a run
of the current `_lgno_impl.py`.

### Per-potential relative L2

| Variant/hidden | hex | deformed | bichromatic | disorder |
|---|---:|---:|---:|---:|
| folded 6 | 0.734708 | 0.011367 | 1.767124 | 0.007581 |
| folded 12 | 0.044660 | 0.018450 | 0.162928 | 0.015019 |
| folded 24 | 0.209076 | 0.229177 | 0.342391 | 0.170258 |
| unfolded 6 | 0.038831 | 0.135817 | 0.386447 | 0.014558 |
| unfolded 12 | 0.047559 | 0.118099 | 0.536103 | 0.027508 |
| unfolded 24 | 0.166243 | 0.150904 | 0.599451 | 0.061532 |

The bichromatic potential is the hardest held-out family for all six recorded
configurations. Global L2 is weighted by tensor norm and is not the mean of
these four values.

## 8. Complete benchmark table

| Model | Hidden | Parameters | Final train loss | Test L2 | Test MSE |
|---|---:|---:|---:|---:|---:|
| LGNO folded | 12 | 370 | 2.51039e-05 | 0.116089 | 19.98254 |
| LGNO unfolded | 6 | 224 | 9.63442e-05 | 0.276925 | 113.7079 |
| LGNO folded | 24 | 1,018 | 2.39322e-04 | 0.282600 | 118.4158 |
| LGNO unfolded | 12 | 500 | 2.83618e-05 | 0.378591 | 212.5229 |
| LGNO unfolded | 24 | 1,268 | 1.53788e-05 | 0.431434 | 275.9904 |
| GNO | 12 | 2,882 | 2.27770 | 0.631984 | 592.214 |
| CNN | 12 | 2,978 | 0.182286 | 0.831388 | 1024.879 |
| CNO | 12 | 10,562 | 0.00395519 | 0.874467 | 1133.841 |
| FNO | 12 | 83,822 | 0.0126388 | 1.055089 | 1650.607 |
| DeepONet | 12 | 148,598 | 1.18178 | 1.066565 | 1686.710 |
| MLP | 12 | 254,120 | 2.96894e-10 | 1.070799 | 1700.129 |
| LGNO folded | 6 | 154 | 2.56971e-05 | 1.282421 | 2438.522 |
| OneShotPDE | 12 | 674 | 1.21718 | 3.085473 | 14115.92 |
| MLPConv | 12 | 518 | 0.591545 | 3.547656 | 18661.60 |

## 9. Expected behavior

- Generated labels must be exactly optical for training and the four stated
  test potentials.
- Train/test true tensors used by folded and unfolded runs must be identical.
- Complex-component metrics must be finite.
- The folded hidden-12 run is the best archived global test result.
- Low training loss alone is not evidence of cross-potential generalization;
  compare the per-potential table.
- A successful run ends with `All results saved successfully.`
