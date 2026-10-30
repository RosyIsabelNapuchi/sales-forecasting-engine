@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   Forecast adaptativo - demo con datos sinteticos
echo ============================================================
where python >nul 2>nul
if errorlevel 1 (
  echo No encontre Python. Instalalo desde https://www.python.org/downloads/
  echo y marca la casilla "Add python.exe to PATH" durante la instalacion.
  pause
  exit /b 1
)
if not exist ".venv" (
  echo Preparando el entorno por primera vez. Puede tardar unos minutos...
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo.
echo Ejecutando el pronostico (unos 4 minutos)...
python run.py
echo.
echo Listo. Se abre la carpeta con los resultados y graficos.
start "" "%~dp0reports"
pause
