# Robustness to multiplicative input noise

## 1. Experiment

The clean 2D Burgers-type field is perturbed as:

$$
u_{\mathrm{noisy}} = u_{\mathrm{clean}}(1+\alpha\eta),
$$

where `η` is sampled uniformly in `[-1,1]` and smoothed with a 21×21 average
pool. Noise levels are:

```text
α = 0.00, 0.01, ..., 0.10
```

For every perturbed input, the target is recomputed:

$$
f = u u_x + u u_y - 0.01\Delta u.
$$

This is therefore an operator-learning robustness test on matched noisy
input/target pairs, not merely corruption of the test input while retaining a
clean target.

## 2. Files

| File | Purpose |
|---|---|
| `generate_data.py` | generates 11 train/test datasets and seeds 447–451 |
| `LGNO.py`, `_lgno_impl.py` | launcher and preserved LGNO |
| `MLP.py`, `MLPConv.py`, `CNN.py` | baseline launchers |
| `DeepONet.py`, `oneshot.py` | DeepONet and OneShotPDE |
| `FNO.py`, `CNO.py`, `GNO.py` | operator baselines |
| `run_all_noise_lgno.py` | generate data and train LGNO at all 11 levels |
| `evaluate_seeded_lgno.py` | LGNO-only evaluation for seeds 447–451 |
| `run_all_noise.py` | archived all-model batch runner |
| `evaluate_seeded_l2_lgno_mlpconv.py` | archived LGNO/MLPConv seeded comparison |

The baseline launchers use the top-level `aligned_baseline_common.py`.

## 3. Generate data

```bash
python generate_data.py
```

Defaults:

| Quantity | Value |
|---|---:|
| Grid | 512×512 |
| Domain | `[-1,1)²` |
| Viscosity | 0.01 |
| Samples per noise level | 2 |
| Train/test split | 1 / 1 |
| Base field seeds | 1234, 1235 |
| Training noise seeds | 9000, 9001 |
| Evaluation seeds | 447–451 |
| Noise levels | 11 |
| Smoothing kernel | 21×21 |
| Global seed | 42 |

Expected structure:

```text
dataset/
├── noise_0.00/
│   ├── train_data/train.pt
│   ├── test_data/test.pt
│   ├── eval_seed_447/test_data/test.pt
│   ├── ...
│   └── eval_seed_451/test_data/test.pt
├── ...
└── noise_0.10/
```

Each train/test file contains:

```text
u: [1,1,512,512] float32
f: [1,1,512,512] float32
x: [512]
y: [512]
metadata: equation, domain, boundary, noise level and optional seed
```

## 4. Train LGNO

One level:

```bash
python LGNO.py \
  --train_dir dataset/noise_0.05/train_data \
  --test_dir dataset/noise_0.05/test_data
```

Top-level equivalent:

```bash
python ../run_lgno.py robust \
  --train-data dataset/noise_0.05/train_data \
  --test-data dataset/noise_0.05/test_data
```

Defaults:

```text
hidden=8
epochs=10000
lr=1e-2
patience=1000
```

All levels:

```bash
python run_all_noise_lgno.py
```

This writes `lgno_noise_training_summary.csv`.

## 5. Train baselines

One method/noise level:

```bash
python MLPConv.py \
  --train_dir dataset/noise_0.05/train_data \
  --test_dir dataset/noise_0.05/test_data
```

Replace `MLPConv.py` with:

```text
MLP.py
CNN.py
DeepONet.py
oneshot.py
FNO.py
CNO.py
GNO.py
```

The archived all-model runner can be invoked with:

```bash
python run_all_noise.py
```

It regenerates data and trains every method listed in that runner. It is much
more expensive than the LGNO-only runner.

## 6. Seeded prediction evaluation

After all 11 LGNO checkpoints exist:

```bash
python evaluate_seeded_lgno.py
```

Outputs:

```text
lgno_seeded_test_results.csv
lgno_seeded_test_summary.csv
```

For the archived LGNO-versus-MLPConv comparison:

```bash
python evaluate_seeded_l2_lgno_mlpconv.py
```

This requires both sets of 11 checkpoints.

## 7. Standard held-out-sample LGNO results

| α | Final loss | Best loss | Test L2 | Test MSE | Time | Final LR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0207248 | 0.0202356 | 0.125916 | 7.51654 | 11.45 s | 1.563e-04 |
| 0.01 | 0.0425274 | 0.0424321 | 0.112098 | 5.95760 | 11.13 s | 1.563e-04 |
| 0.02 | 0.110743 | 0.110163 | 0.119380 | 6.76075 | 11.13 s | 1.563e-04 |
| 0.03 | 0.211890 | 0.211064 | 0.379220 | 68.2949 | 12.00 s | 1.563e-04 |
| 0.04 | 0.377333 | 0.366757 | 0.130422 | 8.09112 | 11.54 s | 6.250e-04 |
| 0.05 | 0.564801 | 0.561224 | 0.179148 | 15.2989 | 11.40 s | 1.563e-04 |
| 0.06 | 0.801781 | 0.797145 | 0.110627 | 5.84930 | 11.20 s | 3.125e-04 |
| 0.07 | 1.06907 | 1.06697 | 0.112381 | 6.05538 | 11.02 s | 1.563e-04 |
| 0.08 | 1.38201 | 1.38201 | 0.153706 | 11.3691 | 11.96 s | 1.563e-04 |
| 0.09 | 1.74561 | 1.73885 | 0.172400 | 14.3623 | 11.01 s | 6.250e-04 |
| 0.10 | 6.73127 | 2.29558 | 0.286023 | 39.7165 | 11.78 s | 1.0e-02 |

All runs used 233 parameters and completed 10,000 epochs. Total recorded LGNO
training time across the 11 levels was 125.62 s.

## 8. Archived single-sample baseline comparison

The table reports the standard held-out sample for each noise level. MLP and
CNN scripts are included, but their corresponding archived scan metrics were
not present in the consolidated result set.

| α | LGNO | MLPConv | OneShotPDE | DeepONet | FNO | CNO | GNO |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.125916 | 0.515082 | 0.871571 | 683.178 | 0.889263 | 1.25716 | 2.88310 |
| 0.01 | 0.112098 | 0.514102 | 0.944025 | 1.04814 | 0.892166 | 1.05579 | 3.47849 |
| 0.02 | 0.119380 | 0.510822 | 0.874626 | 1.00119 | 0.898683 | 1.36517 | 4.07797 |
| 0.03 | 0.379220 | 0.519301 | 0.804979 | 1.04765 | 0.918760 | 1.15788 | 1.53645 |
| 0.04 | 0.130422 | 0.517397 | 0.850671 | 1.04773 | 0.912544 | 1.12173 | 4.73028 |
| 0.05 | 0.179148 | 0.517601 | 0.902429 | 240.165 | 0.922871 | 1.05212 | 1.83205 |
| 0.06 | 0.110627 | 0.515940 | 0.805005 | 1.04792 | 0.926266 | 1.17270 | 2.72600 |
| 0.07 | 0.112381 | 0.519505 | 0.880303 | 4162.41 | 0.928076 | 1.14700 | 1.71277 |
| 0.08 | 0.153706 | 0.520052 | 0.852638 | 1.04767 | 0.928947 | 1.21373 | 1.98634 |
| 0.09 | 0.172400 | 0.510840 | 0.876526 | 1.04667 | 0.937049 | 1.04947 | 2.52881 |
| 0.10 | 0.286023 | 0.517075 | 0.852088 | 1.04742 | 0.925991 | 1.04826 | 0.853373 |

Large DeepONet values are archived numerical outcomes, not transcription
errors.

## 9. Seeded LGNO-versus-MLPConv results

Mean, population standard deviation, and range use seeds 447–451.

| α | LGNO mean +/- std | LGNO min–max | MLPConv mean +/- std | LGNO relative improvement |
|---:|---:|---:|---:|---:|
| 0.00 | 0.101120 +/- 0.030651 | 0.041674–0.128249 | 0.346182 +/- 0.167680 | 70.8% |
| 0.01 | 0.091171 +/- 0.023863 | 0.044967–0.112282 | 0.345900 +/- 0.166118 | 73.6% |
| 0.02 | 0.100453 +/- 0.022719 | 0.055960–0.119015 | 0.344591 +/- 0.163127 | 70.8% |
| 0.03 | 0.248572 +/- 0.121952 | 0.068300–0.438692 | 0.355637 +/- 0.161991 | 30.1% |
| 0.04 | 0.101440 +/- 0.027284 | 0.080687–0.155351 | 0.356482 +/- 0.157329 | 71.5% |
| 0.05 | 0.153808 +/- 0.027514 | 0.102823–0.185613 | 0.361435 +/- 0.151403 | 57.4% |
| 0.06 | 0.116640 +/- 0.012972 | 0.103079–0.139444 | 0.363951 +/- 0.146108 | 68.0% |
| 0.07 | 0.128019 +/- 0.018407 | 0.106965–0.158493 | 0.372344 +/- 0.141342 | 65.6% |
| 0.08 | 0.149746 +/- 0.018490 | 0.127980–0.170973 | 0.377800 +/- 0.135747 | 60.4% |
| 0.09 | 0.178401 +/- 0.015130 | 0.167803–0.208368 | 0.374998 +/- 0.126849 | 52.4% |
| 0.10 | 0.316557 +/- 0.034677 | 0.276375–0.366031 | 0.385279 +/- 0.123453 | 17.8% |

LGNO has lower mean error at every tested noise level in this five-seed
comparison.

## 10. Expected behavior

- Exactly 11 noise directories and 5 evaluation-seed directories per noise
  should be generated.
- Each run must record the correct noise path in `config.txt`; seeded
  evaluation uses it to identify the corresponding checkpoint.
- Relative L2 should remain finite.
- The single held-out sample and five-seed mean are different estimators and
  should not be mixed.
- The strongest-noise seeded LGNO mean should be approximately 0.3166 for the
  archived checkpoints.
- A successful training run ends with `All results saved successfully.`
