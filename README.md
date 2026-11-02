# Quantile-Calibrated Risk Envelopes and Forward Invariance for Control-Affine Systems — code

Reproduces every number, table and data figure of the paper's numerical validation (Section VIII).

## Requirements
Python 3.9+ with numpy, pandas, matplotlib (`pip install numpy pandas matplotlib`).

## Run everything
Windows: double-click `run_all.bat`. Other systems, from this folder:
```
python calibrate.py && python certify.py && python simulate.py && python plot.py
python di_run.py && python di_plot.py
python simulate.py --dt 0.001 && python di_run.py --dt 0.001
python make_tables.py
```
Total runtime a few minutes on a laptop (the dt = 0.001 s refinement runs take most of it). LaTeX tables are written to `tables/`.

## Contents
| File | Purpose |
|---|---|
| `model.py` | Vehicle following: plant, risk functional, nominal ACC, proposed supervisor, CBF-QP baselines |
| `calibrate.py` | Calibration from 3000 independent nominal-driving episodes -> `calibration.json` |
| `certify.py` | Dense-grid check of the theorem hypotheses on the boundary -> `certificate.json` |
| `simulate.py` | Braking scenario + 1000-run random and adversarial Monte Carlo -> `simulation_summary.json`, CSVs |
| `plot.py` | Vehicle figures (`fig_*.pdf`) |
| `di_model.py`, `di_run.py`, `di_plot.py` | Double integrator: closed-form certificate, grid cross-check, Monte Carlo, figures (`di_*`, `fig_di_*`) |
| `make_tables.py` | Writes the LaTeX tables directly from the JSON results |

## Settings
Explicit Euler, time step 0.01 s (refinement runs: 0.001 s, written to `*_dt0.001.json`).
Seeds: calibration 1000, deterministic runs 2000, Monte Carlo initial conditions 3000, Monte Carlo
disturbances 3001, re-seeded for each controller so all controllers see the same disturbance realization. Deployed level: coverage-adjusted empirical quantile
(alpha = beta = 0.05, N = 3000).

## Notes
- The vehicle boundary check is a dense-grid evaluation on (v, vl) in [0, 35]^2 m/s, not an interval-arithmetic
  proof. The domain covers the largest ego speed reached in the simulations (`max_ego_speed` in
  `simulation_summary.json`).
- Tiny floating-point differences across machines can change the last digits of sub-1e-5 overshoot values
  (e.g. 7.72e-6 vs 8.08e-6); violation counts are unaffected.
