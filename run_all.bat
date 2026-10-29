@echo off
REM Reproduce all simulation results for the Quantile-Calibrated Invariant Sets paper.
REM Double-click this file, or run it from cmd. Requires Python with numpy, pandas, matplotlib.
setlocal
cd /d "%~dp0code"
if errorlevel 1 goto :fail

echo === 1/4 Vehicle following: calibration ===
python calibrate.py || goto :fail
echo === 2/4 Vehicle following: boundary certificate ===
python certify.py || goto :fail
echo === 3/4 Vehicle following: simulations ===
python simulate.py || goto :fail
python plot.py || goto :fail

echo === 4/4 Double integrator ===
cd /d "%~dp0code\double_integrator"
python di_run.py > di_run_output.txt || goto :fail
python di_plot.py || goto :fail

echo === Regenerating LaTeX tables from the results ===
cd /d "%~dp0"
python make_tables.py || goto :fail

echo.
echo ALL DONE. Check: B_hat_plus in code\calibration.json should be 0.38292...
findstr /C:"B_hat_plus" code\calibration.json
pause
exit /b 0

:fail
echo.
echo *** A step FAILED -- copy the messages above and send them. ***
pause
exit /b 1
