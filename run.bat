@echo off
title Air Piano
cd /d "%~dp0"
echo ===================================================
echo               STARTING AIR PIANO V1
echo ===================================================
echo.
echo Press 'Q' or 'ESC' in the window anytime to exit.
echo.
piano_env\Scripts\python.exe main.py %*
pause
