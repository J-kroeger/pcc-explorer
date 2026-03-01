@echo off
ECHO Activating Python virtual environment...

REM Check if the venv exists before trying to activate it
IF NOT EXIST ".\venv\Scripts\activate.bat" (
    ECHO ERROR: Virtual environment not found.
    ECHO Please run the installation steps in README.md first.
    PAUSE
    EXIT /B
)

call .\venv\Scripts\activate.bat

ECHO Launching APCC Analyzer GUI...
python -m src.main_gui

ECHO GUI closed.
PAUSE