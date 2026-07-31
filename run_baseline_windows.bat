@echo off
setlocal
pushd %~dp0
call .venv\Scripts\activate
if errorlevel 1 (
  echo Virtual environment not found. Run setup_windows.bat first.
  pause
  popd
  exit /b 1
)

python -m src.member1.inspect_data
if errorlevel 1 goto :error
python -m src.member1.demo_preprocessing
if errorlevel 1 goto :error
python -m src.member1.train_logistic_regression
if errorlevel 1 goto :error
python -m src.member1.evaluate_skill_extraction --sample-size 500
if errorlevel 1 goto :error
python -m src.member1.demo_member1
if errorlevel 1 goto :error

echo.
echo Baseline pipeline completed successfully.
if "%1"=="--streamlit" (
  echo.
  echo Launching Streamlit demo in a new window...
  REM Start Streamlit using the venv-installed executable in a new window so the batch doesn't block
  if exist ".venv\Scripts\streamlit.exe" (
    start "" "%CD%\.venv\Scripts\streamlit.exe" run src/member1/streamlit_demo.py
  ) else (
    echo Streamlit executable not found in .venv\Scripts. To run the demo, install streamlit in the virtual environment:
    echo     .venv\Scripts\activate
    echo     pip install streamlit
  )
)
pause
popd
exit /b 0

:error
echo.
echo A command failed. Read the error above.
pause
popd
exit /b 1
