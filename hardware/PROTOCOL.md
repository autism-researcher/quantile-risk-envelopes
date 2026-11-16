# RoboCar feasibility study — protocol and deviations (Section 6.3)

## Equipment and setting
- RoboCar 1/10 Pro (ZMP): 429 x 190 x 150 mm, 3.0 kg, NVIDIA Jetson AGX Orin, wheel-speed
  sensing, IMU, 2D LiDAR. ROS 2, namespace `/rc_23005`.
- Indoor tiled floor, straight-line motion (steering 0). Speed command at 25 Hz.
- All 27 runs on 16 November 2026, 10:56-11:30 local time; motor battery 8.38 V at the first
  run and 7.76 V at the last (logged in `batt_v`).

## Runs as performed
| Mode | Runs | Purpose |
|---|---|---|
| ramp (v_cmd += u dt) | SPD-04-1..3, SPD-06-1..3 | speed and odometry check against taped distance (taped.csv; none for SPD-04-1) |
| ramp | ACC-06-1 | acceleration step from rest |
| ramp | BRK-04-1, BRK-06-1, BRK-08-1 | braking steps |
| ramp | RND-01..05 | random input, disturbance estimate (RND-02 excluded) |
| inverse (v_cmd = v + 0.4 u) | BRKI-06-1..4 | braking steps |
| inverse | INV-11..13 | random input, fitting set |
| inverse | INV-14..18 | random input, held-out set |

Command lines are given in the header of `car/robocar_ident.py`; each log's first line
records the settings of that run.

## Analysis
`pc/platform_measurements.py` (Table 5) and `pc/robocar_feasibility.py` (Table 6, Fig. 8),
run on the logs in `data/ident_logs`. Braking: mean deceleration from the brake command to
standstill. Disturbance: |dv/dt - u| over 0.24 s windows with the input shifted by 0.24 s,
moving samples only (v > 0.05 m/s). Feasibility: margin of condition (15) from
`pc/robocar_certify.py` for the risk functional of Table 6.

## Planned but not performed
The original plan continued with 200 baseline calibration runs and 150 closed-loop runs
(nominal controller, supervisor, robust CBF-QP). These were not performed because
condition (15) was not satisfied with the worst-case measured braking and disturbance.

## Deviations from the plan
| Item | Deviation | Handling |
|---|---|---|
| Identification | 15 ramp-mode runs instead of the planned 62 (battery) | limits estimated from fewer runs |
| RND-02 | reached the wall (distance stop 5.5 m beyond the end of the floor) | excluded; later runs used a 3.5 m stop |
| Command mode | inverse-command mode introduced after the ramp-mode results (disturbance larger than braking) | reported as exploratory |
| BRKI-06-3 | two attempts without motion (ROS namespace missing); their logs were overwritten by the valid run | no data lost |
| Extra runs | BRKI-06-3/4 and INV-16..18 added after the first inverse-mode results, same settings | held-out set INV-14..18 |
| Calibration and closed loop | not performed | see above |
