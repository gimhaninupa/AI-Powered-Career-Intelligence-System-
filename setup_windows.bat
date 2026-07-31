@echo off
setlocal

echo Creating Python virtual environment...
py -3.11 -m venv .venv 2>nul || py -3.14 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul
if errorlevel 1 (
  echo Python 3 was not found. Install Python and enable the py launcher.
  pause
  exit /b 1
)

call .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_baseline.txt
pip install click
echo Downloading spaCy English model (en_core_web_sm)...
python -m spacy download en_core_web_sm

if errorlevel 1 (
  echo Installation failed. Read README.md and copy the terminal error.
  pause
  exit /b 1
)

echo.
echo Setup completed successfully.
echo Activate later with: .venv\Scripts\activate
pause
