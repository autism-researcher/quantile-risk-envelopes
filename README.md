# Quantile-Calibrated Risk Envelopes for Robust Forward Invariance — code and data

Code, frozen results and hardware data for the paper
*Quantile-Calibrated Risk Envelopes for Robust Forward Invariance* (M. B. Hossain, I. M. Mehedi),
submitted to the International Journal of Robust and Nonlinear Control.

One command regenerates every number, table and data figure (about 4 minutes on a laptop):

```
pip install -r requirements.txt
run_all.bat          # Windows
bash run_all.sh      # Linux / macOS
```

Tested with Python 3.13, numpy 2.5, pandas 3.0, matplotlib 3.11. The frozen outputs in this
repository were regenerated from a clean copy with `run_all.sh` and are byte-identical to the
values in the paper (only the timing fields `runtime_us_*` differ between machines).

## Where each result comes from

| Paper | Script(s) (folder) | Output |
|---|---|---|
| Table 1, Fig. 5 (vehicle-following certificate) | `calibrate.py`, `certify.py`, `plot.py` (simulation) | `calibration.json`, `certificate.json`, `fig_certificate.pdf` |
| Table 2, Figs. 2–4 (Monte Carlo, braking scenario) | `simulate.py`, `plot.py` | `simulation_summary.json`, `mc_*.csv`, `traj_*.csv`, `fig_risk/control/montecarlo_histogram.pdf` |
| dt = 0.001 s refinement | `simulate.py --dt 0.001`, `di_run.py --dt 0.001`, `sg_run.py --dt 0.001` | `*_dt0.001.json` |
| Tables 3–4, Fig. 6 (level, braking and lead-deceleration sweeps) | `level_sweep.py`, `sg_plot.py` | `level_sweep.json`, `fig_level_sweep.pdf` |
| Supporting Information S-I (double integrator) | `di_model.py`, `di_run.py`, `di_plot.py` | `di_results.json`, `fig_di_*.pdf` |
| Supporting Information S-II (vanishing authority) | `sg_model.py`, `sg_run.py`, `sg_plot.py` | `sg_results.json`, `sg_boundary_*.csv`, `fig_sg_conditions.pdf` |
| LaTeX tables | `make_tables.py`, `make_tables_new.py` | `simulation/tables/*.tex` |
| Table 5 (hardware platform, measured values) | `platform_measurements.py` (hardware/pc) | `hardware/results/platform_measurements.json` |
| Table 6, Fig. 8 (hardware feasibility study) | `robocar_feasibility.py` (hardware/pc) | `hardware/results/robocar_feasibility.json`, `fig_robocar.pdf` |

Settings: explicit Euler, dt = 0.01 s. Seeds: calibration 1000, deterministic runs 2000, Monte Carlo
initial conditions 3000 and disturbances 3001 (re-seeded per controller). Deployed level: coverage-adjusted
empirical quantile, alpha = beta = 0.05, N = 3000 independent baseline episodes.
The vehicle certificate is a dense-grid evaluation on (v, v_l) in [0, 35]^2 m/s, not an interval-arithmetic proof.

## Hardware feasibility study (hardware/)

RoboCar 1/10 Pro (ZMP), indoor tiled floor, straight-line runs, 16 November 2026. 27 runs: 15 in ramp-command
mode, 12 in inverse-command mode (v_cmd = v + 0.4 u), the latter introduced after the first results
(see `PROTOCOL.md`). Run RND-02 reached the wall and is excluded from the disturbance analysis.

- `car/robocar_ident.py` — identification script run on the vehicle (ROS 2, namespace `/rc_23005`).
- `car/robocar_model.py` — risk model and parameters used by the PC scripts.
- `pc/robocar_certify.py` — evaluates the conditions of Theorem 1 on a grid of the boundary.
- `pc/robocar_feasibility.py` — braking, disturbance bound (0.24 s windows; INV-11..13 fitting,
  INV-14..18 held out), feasibility margins, largest feasible disturbance, ramp-mode design sweep.
- `pc/platform_measurements.py` — loop period, speed-update interval, odometry error, start-up delay.
- `data/ident_logs/ident_<run>.csv` — one file per run. Line 1: run metadata (test, command mode, rate,
  speed, command, seed, floor, start time). Columns: `t` [s], `t_pub_wall`, `phase`, `u_cmd` [m/s^2],
  `v_cmd` [m/s], `v_meas` [m/s, wheel speed], `status_age`, `n_status`, `x_int` [m, integrated distance],
  `imu_ax` [m/s^2], `imu_age`, `batt_v` [V]. `taped.csv`: distances measured with a tape.
- `PROTOCOL.md` — equipment, runs as performed, analysis, deviations from the plan.

No calibration or closed-loop runs were made on the vehicle: the feasibility condition was not met with
the worst-case measured values, so the supervisor was not run on the car.

## Licence
MIT Licence; see `LICENSE`.
