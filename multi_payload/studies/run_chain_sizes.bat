@echo off
setlocal
set "SCRIPT=%~dp0chain_method.py"
set "RESULTS_ROOT=%~dp0results\chain_comparison"
if not "%~1"=="" set "RESULTS_ROOT=%~f1"
set "GRID_MODE=adaptive"
if not "%~2"=="" set "GRID_MODE=%~2"
set "CHAIN_MODE=guarded"
if not "%~3"=="" set "CHAIN_MODE=%~3"
set "SEEDS=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
set "HEADROOM=60"
set "LINK_RHO=2"
set "MAX_GRID=480"
set "CHAIN_OPTIONS="
if /I "%CHAIN_MODE%"=="guarded" set "CHAIN_OPTIONS=--guard-links "
REM if /I "%CHAIN_MODE%"=="guarded" set "CHAIN_OPTIONS=--guard-links --cone-repulsion"
if /I not "%GRID_MODE%"=="adaptive" if /I not "%GRID_MODE%"=="fixed" (
    echo ERROR: Second argument must be adaptive or fixed.
    exit /b 1
)
if /I not "%CHAIN_MODE%"=="default" if /I not "%CHAIN_MODE%"=="guarded" (
    echo ERROR: Third argument must be default or guarded.
    exit /b 1
)
if not exist "%SCRIPT%" (
    echo ERROR: Put this batch file beside chain_method.py.
    exit /b 1
)
if not exist "%RESULTS_ROOT%" mkdir "%RESULTS_ROOT%"
if not exist "%RESULTS_ROOT%" exit /b 1
REM Adaptive mode: N is the starting/minimum grid; actual grid is K_final in JSON.
REM Fixed mode: N=Kmin=Kmax, so no refinement beyond 30/200 is permitted.
REM The chain script also runs its original Gaussian baseline at the starting N.
for %%L in (crossing random_seq) do (
    for %%M in (10 20 40 70 100) do (
        for %%N in (30 200) do (
            call :run %%L %%M %%N
            if errorlevel 1 exit /b 1
        )
    )
)
echo All chain benchmarks completed.
exit /b 0

:run
set "LAYOUT=%~1"
set "M=%~2"
set "N=%~3"
set "KMAX=%MAX_GRID%"
if /I "%GRID_MODE%"=="fixed" set "KMAX=%N%"
set "RING_ARGS="
if "%LAYOUT%"=="crossing" (
    call :crossing_radius %M%
)
echo.
echo Chain: %LAYOUT%, M=%M%, Nstart=%N%, Kmax=%KMAX%, %CHAIN_MODE%...
python "%SCRIPT%" ^
    --Ms %M% --seeds %SEEDS% ^
    --N %N% --Kmin %N% --Kmax %KMAX% --tf 500 ^
    --layout %LAYOUT% %RING_ARGS% ^
    --h %HEADROOM% --link-rho %LINK_RHO% %CHAIN_OPTIONS% ^
    --out "%RESULTS_ROOT%\chain_%LAYOUT%_Nstart%N%_M%M%_h%HEADROOM%_rho%LINK_RHO%_%GRID_MODE%_%CHAIN_MODE%.json"
exit /b %ERRORLEVEL%

:crossing_radius
if "%~1"=="10" set "RING_ARGS=--ring 600"
if "%~1"=="20" set "RING_ARGS=--ring 900"
if "%~1"=="40" set "RING_ARGS=--ring 1300"
if "%~1"=="70" set "RING_ARGS=--ring 1800"
if "%~1"=="100" set "RING_ARGS=--ring 2200"
exit /b 0

