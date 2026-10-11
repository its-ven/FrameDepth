@echo off
REM Run this ONCE to install everything depth3d needs (Python 3.13).
cd /d "%~dp0"

echo Installing PyTorch (CUDA 12.8 build, required for RTX 50-series)...
py -3.13 -m pip install --upgrade pip
py -3.13 -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail

echo Installing the rest...
py -3.13 -m pip install transformers numpy glfw moderngl windows-capture openvr pillow av
if errorlevel 1 goto fail

echo.
echo Done. You can now double-click run_depth3d.bat
pause
exit /b 0

:fail
echo.
echo Install failed. Check the error above.
pause
exit /b 1
