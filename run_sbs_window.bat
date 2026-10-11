@echo off
setlocal
cd /d "%~dp0"

REM Simple side-by-side window for Desktop+/Bigscreen. The SteamVR player (run_depth3d_vr.bat) is recommended.
REM ================= SETTINGS (edit these) =================
REM Monitor to convert to 3D (1 = main monitor)
set SOURCE_MONITOR=1

REM Depth model: small / base / large
set MODEL=large

REM Depth model input width in pixels. Lower = faster. Try 518 or 392 if fps drops.
set DEPTH_WIDTH=700

REM Size of the side-by-side output window. Keep it at your source aspect ratio.
REM 2580x1080 = 21:9 in Half SBS (Desktop+ "HSBS" or Bigscreen), ~1290 px per eye
REM For a 16:9 monitor use 1920x1080 instead.
set WIN_W=2580
set WIN_H=1080

REM 3D look. Can also be changed live with the arrow keys.
set STRENGTH=0.025
set CONVERGENCE=0.4
REM =========================================================

echo Capturing monitor %SOURCE_MONITOR% with the "%MODEL%" model...
echo Drag the "depth3d SBS" window to ANOTHER monitor, or it will capture itself.
echo.

py -3.13 depth3d.py --source-monitor %SOURCE_MONITOR% --model %MODEL% --depth-width %DEPTH_WIDTH% --width %WIN_W% --height %WIN_H% --strength %STRENGTH% --convergence %CONVERGENCE%

echo.
echo depth3d (SBS window) exited. If there was an error, it is shown above.
pause
