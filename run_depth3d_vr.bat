@echo off
setlocal
cd /d "%~dp0"

REM ================= STARTING VALUES (edit these) =================
REM Settings you save from the in-VR menu (settings.json) override the 3D/screen values below.
REM To go back to these, press Reset + Save in the menu, or delete settings.json.

REM Monitor to convert to 3D (1 = main monitor)
set SOURCE_MONITOR=1

REM Depth model: small / base / large (drop to base if the game stutters)
set MODEL=large

REM Depth model input width in pixels. Higher = crisper depth edges, slower.
set DEPTH_WIDTH=700

REM Starting screen size (degrees), distance (metres), text sharpening (0..1)
set FOV=70
set DISTANCE=3.0
set SHARPEN=0.5

REM Starting 3D look: total depth range in cm, and which depth sits at the screen plane
set DEPTH_CM=6
set CONVERGENCE=0.4

REM --sync   = each video frame is shown with its own depth map (best for MOVIES)
REM --no-sync = lowest latency (better for GAMES)
set SYNC_FLAG=--sync

REM Mouse cursor: leave empty to show it, or set to --no-cursor to hide it
set CURSOR_FLAG=

REM Warp quality: samples searched per pixel (24 default; 16 = faster, 32 = best edges)
set SAMPLES=24

REM Depth edge refinement. REFINE_WIDTH 0 = off. EDGE_SIGMA smaller = sharper. DILATE 0 = off.
set REFINE_WIDTH=1920
set EDGE_SIGMA=0.10
set DILATE=1

REM Scene-cut sensitivity. Lower = more cuts detected.
set CUT_THRESHOLD=0.12

REM Render resolution multiplier per eye (1.0 = SteamVR's recommended size)
set RENDER_SCALE=1.0

REM If the menu laser points too high or low, set e.g. 20 or -20 (degrees)
set POINTER_TILT=0

REM Set to --debug-input to print controller button ids (helps if the menu button doesn't respond)
set DEBUG_INPUT_FLAG=
REM =================================================================

echo Start SteamVR and connect the Steam Frame BEFORE running this.
echo Capturing monitor %SOURCE_MONITOR% with the "%MODEL%" model...
echo In VR: press MENU or A on a controller to open the settings panel.
echo Fallback: hold Ctrl+Alt and press Space. Put 360/180 images and videos in the "environments" folder.
echo.

py -3.13 depth3d_vr.py --source-monitor %SOURCE_MONITOR% --model %MODEL% --depth-width %DEPTH_WIDTH% --fov %FOV% --distance %DISTANCE% --sharpen %SHARPEN% --depth-cm %DEPTH_CM% --convergence %CONVERGENCE% %SYNC_FLAG% %CURSOR_FLAG% --samples %SAMPLES% --refine-width %REFINE_WIDTH% --edge-sigma %EDGE_SIGMA% --dilate %DILATE% --cut-threshold %CUT_THRESHOLD% --render-scale %RENDER_SCALE% --pointer-tilt %POINTER_TILT% %DEBUG_INPUT_FLAG%

echo.
echo depth3d_vr exited. If there was an error, it is shown above.
pause
