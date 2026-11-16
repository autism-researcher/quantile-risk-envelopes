@echo off
REM Reproduce every number, table and data figure of the paper (Windows).
REM Requires Python 3 with numpy, pandas, matplotlib (pip install -r requirements.txt).
setlocal
cd /d "%~dp0simulation"
echo === Vehicle following: calibration, certificate, simulations, figures ===
python calibrate.py || goto :fail
python certify.py || goto :fail
python simulate.py || goto :fail
python plot.py || goto :fail
echo === Double integrator (Supporting Information S-I) ===
python di_run.py > di_run_output.txt || goto :fail
python di_plot.py || goto :fail
echo === Time-step refinement (dt = 0.001 s) ===
python simulate.py --dt 0.001 || goto :fail
python di_run.py --dt 0.001 > di_run_dt0.001_output.txt || goto :fail
echo === Certified level and actuator authority (Section 6.2) ===
python level_sweep.py > level_sweep_output.txt || goto :fail
echo === Vanishing authority (Supporting Information S-II) ===
python sg_run.py > sg_run_output.txt || goto :fail
python sg_run.py --dt 0.001 > sg_run_dt0.001_output.txt || goto :fail
python sg_plot.py || goto :fail
echo === LaTeX tables ===
python make_tables.py || goto :fail
python make_tables_new.py || goto :fail
echo === Hardware feasibility study (Section 6.3) ===
cd /d "%~dp0hardware\pc"
python platform_measurements.py --logs ..\data\ident_logs --out ..\results || goto :fail
python robocar_feasibility.py --logs ..\data\ident_logs --out ..\results || goto :fail
echo.
echo ALL DONE.
pause
exit /b 0
:fail
echo *** A step FAILED -- see the messages above. ***
pause
exit /b 1
