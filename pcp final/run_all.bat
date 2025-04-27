@echo off
echo ===== COMPILING AND RUNNING ALL CONVEX HULL ALGORITHMS =====
setlocal enableextensions enabledelayedexpansion

REM Clear results.txt
> results.txt echo ===== CONVEX HULL BENCHMARK RESULTS =====

REM List of convex hull algorithms
set algos=scan heaphull incremental quickheaphull quickhull

for %%a in (%algos%) do (
    echo -------------------------------------- >> results.txt
    echo Processing: %%a.cu in folder %%a >> results.txt
    echo Compiling %%a.cu ...
    nvcc %%a.cu -o %%a.exe

    if exist %%a.exe (
        echo Running %%a.exe ...
        echo. >> results.txt
        echo ========= RUNNING %%a ========= >> results.txt
        echo. >> results.txt
        %%a.exe >> results.txt
    ) else (
        echo Failed to compile %%a.cu >> results.txt
        echo [ERROR] %%a.exe failed to compile >> results.txt
    )
)

echo. >> results.txt
echo DONE AT: %DATE%  %TIME% >> results.txt
echo All executions completed. See results.txt for full logs.
pause
