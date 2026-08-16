@echo off
setlocal
call .venv\Scripts\activate
if errorlevel 1 (
  echo Virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)

python -m pip install --upgrade pip
pip install -r requirements_bert.txt

if errorlevel 1 (
  echo BERT dependency installation failed. Google Colab is recommended.
  pause
  exit /b 1
)

echo.
echo BERT dependencies installed successfully.
pause
