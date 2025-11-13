@echo off
ECHO ----------------------------------------------------
ECHO  Tomato Early Blight Project Setup
ECHO ----------------------------------------------------

REM --- 1. Create Python Virtual Environment ---
ECHO Creating Python virtual environment (.venv)...
ECHO (This assumes 'python' is in your system's PATH)
python -m venv .venv
ECHO Environment created.
ECHO.

REM --- 2. Install Libraries ---
.venv\Scripts\pip.exe install -r requirements.txt

ECHO.
ECHO Library installation complete!
ECHO.

REM --- 3. Generate requirements.txt ---
ECHO Freezing environment to requirements.txt...
.venv\Scripts\pip.exe freeze > requirements.txt
ECHO.

REM --- 4. Finish ---
ECHO ----------------------------------------------------
ECHO  Setup is complete!
ECHO ----------------------------------------------------
ECHO.
ECHO To get started:
ECHO 1. Navigate to your new project folder:
ECHO    cd tomato-earlyblight
ECHO.
ECHO 2. Activate your virtual environment:
ECHO    .venv\Scripts\activate
ECHO.
ECHO 3. Start coding!
ECHO.

cd ..
PAUSE