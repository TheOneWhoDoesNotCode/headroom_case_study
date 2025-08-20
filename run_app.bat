@echo off
setlocal enabledelayedexpansion

REM Change to the directory of this script (project root)
pushd "%~dp0"

REM Prefer the project's venv Python if present
set "PY_EXE="
if exist ".venv\Scripts\python.exe" (
  set "PY_EXE=.venv\Scripts\python.exe"
) else (
  where python >nul 2>nul && for /f "delims=" %%i in ('where python') do (
    set "PY_EXE=%%i"
    goto :gotpy
  )
)
:gotpy
if "%PY_EXE%"=="" (
  echo Python not found. Please install Python or create the virtual environment.
  popd
  exit /b 1
)

REM Ensure Streamlit is available; if not, install minimal deps
"%PY_EXE%" -m pip show streamlit >nul 2>nul || (
  echo Installing required packages...
  "%PY_EXE%" -m pip install --upgrade pip
  "%PY_EXE%" -m pip install streamlit pandas plotly pyyaml
)

REM Start the Streamlit app. Forward any args (e.g., --server.port 8502)
echo Starting Streamlit app...
"%PY_EXE%" -m streamlit run "headroom\app\app.py" %*

popd
endlocal
