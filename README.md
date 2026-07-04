# LGNO reproducibility package

This directory contains the code required to generate data, train models, produce predictions, and evaluate the LGNO experiments.

Included cases:

- 1D periodic diffusion;
- 2D Burgers-type operator learning;
- 2D incompressible Navier–Stokes;
- 2D Gross–Pitaevskii;
- 3D perturbed-harmonic Schrödinger;
- robustness under multiplicative input noise.

## 1. Directory structure

```text
LGNO/
├── README.md
├── requirements.txt
├── aligned_baseline_common.py
├── lgno_common.py
├── run_lgno.py
├── 1D Diffusion/
├── 2D Burgers/
├── 2D Navier-Stokes/
├── 2D Gross-Pitaevskii/
├── 3D Schrodinger (Perturbed Harmonic)/
└── Robustness/
```

Each case directory contains its training entry point and a detailed `README.md`. Cases with synthetic data also contain `generate_data.py`. Baseline launchers use the shared implementation in `aligned_baseline_common.py`.

## 2. Environment

Required packages:

```text
numpy
torch
matplotlib
```

Recommended installation:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA is strongly recommended for the $512 \times 512$ and three-dimensional cases. PyTorch automatically falls back to CPU.

## 3. LGNO interface

Each case-level `LGNO.py` uses the same launcher interface:

```text
LGNO.py
    -> ../lgno_common.py
    -> case-specific _lgno_impl.py
```

The launcher selects the case directory and forwards the original arguments. The case-specific numerical implementation remains in `_lgno_impl.py`.

The 2D Gross–Pitaevskii case additionally provides:

```text
LGNO_unfolded.py
    -> _lgno_unfolded_impl.py
```

List available LGNO cases:

```bash
python run_lgno.py --list
```

Examples:

```bash
python run_lgno.py diffusion-1d
python run_lgno.py burgers-2d --hidden 8
python run_lgno.py navier-stokes-2d --hidden 16
python run_lgno.py gpe-2d-folded --hidden 12
python run_lgno.py gpe-2d-unfolded --hidden 12
python run_lgno.py perturbed-harmonic-3d
python run_lgno.py robust
```

Original case-specific parameters remain valid. Common aliases, where the selected implementation supports them, are:

```text
--train-data PATH
--test-data PATH
--output PATH
```

Display a case's native options with:

```bash
python run_lgno.py CASE --help
```

## 4. Standard reproduction workflow

For generated-data cases:

```bash
cd "2D Burgers"
python generate_data.py
python LGNO.py
python MLPConv.py
```

Replace `MLPConv.py` with another included baseline:

```text
MLP.py
CNN.py
DeepONet.py
oneshot.py
FNO.py
CNO.py
GNO.py
```

The exact available files and commands are documented in each case README.

For robustness experiments, `run_all_noise_lgno.py` generates all noise datasets and trains LGNO at every noise level. The all-model runner and seeded evaluation scripts are also included.

## 5. Metrics

The principal reported relative error is the relative $L^2$ error:

$$
e_{\mathrm{rel}} =
\frac{\lVert \hat{y}-y\rVert_2}
{\lVert y\rVert_2 + 10^{-20}}.
$$

The mean squared error is:

$$
e_{\mathrm{MSE}} =
\mathrm{mean}\left((\hat{y}-y)^2\right).
$$

For time-dependent cases, the documentation distinguishes among three types of errors:

- **One-step derivative error:** the error of the learned time derivative or learned operator.
- **Rollout error:** the error accumulated after repeated Euler updates.
- **Static operator error:** the direct error of the mapping $u \to f$.

The spreadsheet column **Test L2 Error** uses the primary metric for each case:

- rollout relative $L^2$ error for **1D diffusion** and **2D Navier–Stokes**;
- direct operator relative $L^2$ error for the **static**, **Gross–Pitaevskii**, and **3D Schrödinger** cases.

## 6. Recorded LGNO reference results

These values come from the completed experiment outputs documented in each case README. Runtime depends on hardware and software versions.

| Case             | Configuration             | Parameters | Primary relative L2 |   Training time |
| ---------------- | ------------------------- | ---------: | ------------------: | --------------: |
| 1D Diffusion     | hidden 2                  |         23 |   0.0120931 rollout |         73.37 s |
| 1D Diffusion     | hidden 4                  |         51 |   0.0247799 rollout |        167.15 s |
| 1D Diffusion     | hidden 8                  |        131 |   0.0550472 rollout |         26.84 s |
| 2D Burgers       | hidden 4                  |        105 |           0.0495040 |        114.01 s |
| 2D Burgers       | hidden 8                  |        233 |           0.0928579 |        227.41 s |
| 2D Burgers       | hidden 16                 |        585 |            0.228290 |        129.31 s |
| 2D Navier–Stokes | hidden 8                  |        548 |   0.0102719 rollout |         75.64 s |
| 2D Navier–Stokes | hidden 16                 |      1,188 |  0.00813705 rollout |        169.55 s |
| 2D Navier–Stokes | hidden 32                 |      2,852 |  0.00832833 rollout |         78.65 s |
| 2D GPE folded    | hidden 12                 |        370 |            0.116089 |      7,589.10 s |
| 2D GPE unfolded  | hidden 12                 |        500 |            0.378591 |      6,667.06 s |
| 3D Schrödinger   | hidden 16                 |        988 |          0.00313018 |     53,905.90 s |
| Robustness, $\alpha = 0.00$       | hidden 8 |        233 |            0.125916 |         11.45 s |
| Robustness, $\alpha = 0.10$       | hidden 8 |        233 |            0.286023 |         11.78 s |

## 7. Reproducibility notes

- Scripts normally set random seed 42.
- GPU execution can vary slightly with PyTorch, CUDA, driver, and cuDNN versions.
- Training time is not an accuracy criterion and should only be compared on equivalent hardware.
- The 2D Gross–Pitaevskii test set changes potential families.
- The 3D Schrödinger generated train and test files are each approximately 0.41 GB.
- Robustness seeded evaluation requires checkpoints for all 11 noise levels.

## 8. Case documentation

- [1D Diffusion](<1D Diffusion/README.md>)
- [2D Burgers](<2D Burgers/README.md>)
- [2D Navier–Stokes](<2D Navier-Stokes/README.md>)
- [2D Gross–Pitaevskii](<2D Gross-Pitaevskii/README.md>)
- [3D Schrödinger](<3D Schrodinger (Perturbed Harmonic)/README.md>)
- [Robustness](Robustness/README.md)
