#!/usr/bin/env bash
# Reproduce every number, table and data figure of the paper (Linux/macOS).
set -e
cd "$(dirname "$0")/simulation"
python3 calibrate.py; python3 certify.py; python3 simulate.py; python3 plot.py
python3 di_run.py > di_run_output.txt; python3 di_plot.py
python3 simulate.py --dt 0.001; python3 di_run.py --dt 0.001 > di_run_dt0.001_output.txt
python3 level_sweep.py > level_sweep_output.txt
python3 sg_run.py > sg_run_output.txt; python3 sg_run.py --dt 0.001 > sg_run_dt0.001_output.txt; python3 sg_plot.py
python3 make_tables.py; python3 make_tables_new.py
cd ../hardware/pc
python3 platform_measurements.py --logs ../data/ident_logs --out ../results
python3 robocar_feasibility.py --logs ../data/ident_logs --out ../results
echo "ALL DONE."
