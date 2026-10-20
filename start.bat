@echo off
setlocal
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto nopython

if exist ".venv\Scripts\python.exe" goto havevenv
echo Creating virtual environment...
%PY% -m venv .venv
if errorlevel 1 goto venvfail
:havevenv
call ".venv\Scripts\activate.bat"
echo Installing dependencies - this can take a few minutes the first time...
python -m pip install --upgrade pip >nul 2>nul
python -m pip install -r requirements.txt
if errorlevel 1 goto pipfail
echo Starting the app. Open the Local URL shown below in your browser.
python -m streamlit run app.py
goto end

:nopython
echo Python was not found. Install Python 3.10 - 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
goto end
:venvfail
echo Could not create the virtual environment. Check your Python installation.
goto end
:pipfail
echo Dependency installation failed. Check your internet connection and the messages above, then run start.bat again.
:end
pause
