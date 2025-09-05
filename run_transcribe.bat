@echo off
SET "CONDA_PATH=%USERPROFILE%\miniconda3"
CALL "%CONDA_PATH%\Scripts\activate.bat"
CALL conda activate transcribe
CD /d %~dp0
python transcribe.py
pause