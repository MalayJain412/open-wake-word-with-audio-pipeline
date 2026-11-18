@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON="

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined PYTHON if exist "%ROOT%ovenv\Scripts\python.exe" set "PYTHON=%ROOT%ovenv\Scripts\python.exe"
if not defined PYTHON if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not defined PYTHON if exist "%ROOT%..\runtime\venv\Scripts\python.exe" set "PYTHON=%ROOT%..\runtime\venv\Scripts\python.exe"

if not defined PYTHON (
  echo A compatible Python environment was not found.
  echo Activate a virtual environment, create oenv or .venv in this folder,
  echo or provide ..\runtime\venv.
  pause
  exit /b 1
)

cd /d "%ROOT%"
"%PYTHON%" -X utf8 llamacpp_launcher.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
