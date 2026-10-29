@echo off
REM Reproduce all results (flat repository layout: all scripts in this folder).
REM Requires Python with numpy, pandas, matplotlib.
setlocal
cd /d "%~dp0"
echo === 1/4 Vehicle following: calibration ===
python calibrate.py || goto :fail
echo === 2/4 Vehicle following: boundary certificate ===
python certify.py || goto :fail
echo === 3/4 Vehicle following: simulations and figures ===
python simulate.py || goto :fail
python plot.py || goto :fail
echo === 4/4 Double integrator ===
python di_run.py > di_run_output.txt || goto :fail
python di_plot.py || goto :fail
echo === LaTeX tables ===
python make_tables.py || goto :fail
echo.
echo ALL DONE. Check: B_hat_plus in calibration.json should be 0.38292...
findstr /C:"B_hat_plus" calibration.json
pause
exit /b 0
:fail
echo *** A step FAILED -- copy the messages above and send them. ***
pause
exit /b 1
