@echo off
setlocal

set EXE=shell_sort.exe
set OUT=results.csv

if exist %OUT% del %OUT%

echo Running Shell sort OpenCL benchmarks...
echo Output file: %OUT%
echo.

REM input_type:
REM 0=random, 1=sorted, 2=reversed, 3=nearly_sorted
REM gap_type:
REM 0=shell, 1=knuth, 2=ciura

for %%N in (1024 4096 16384 65536 262144) do (
    for %%I in (0 1 2 3) do (
        for %%G in (0 1 2) do (
            for %%L in (32 64 128 256) do (
                echo ==== N=%%N INPUT=%%I GAP=%%G LOCAL=%%L ====
                %EXE% %%N %%I %%G %%L
            )
        )
    )
)

echo.
echo Benchmark finished.
pause