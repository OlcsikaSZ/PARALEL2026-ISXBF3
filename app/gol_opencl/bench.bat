@echo off
setlocal EnableDelayedExpansion

set EXE=gol_opencl
set OUT=results.csv

if exist %OUT% del %OUT%

echo Running OpenCL Game of Life benchmarks...
echo Output file: %OUT%
echo.

for %%S in (512 1024 2048 4096) do (
    if %%S==512 set ITERS=2000
    if %%S==1024 set ITERS=1000
    if %%S==2048 set ITERS=500
    if %%S==4096 set ITERS=250

    echo ==== SIZE %%S x %%S, iters=!ITERS! ====

    REM Naive baseline
    %EXE% --rows %%S --cols %%S --iters !ITERS! --wrap 0 --lx 16 --ly 16 --repeat 5 --warmup 1 --csv --out %OUT%

    REM Tiled 8x8
    %EXE% --rows %%S --cols %%S --iters !ITERS! --wrap 0 --tiled 1 --lx 8 --ly 8 --repeat 5 --warmup 1 --csv --out %OUT%

    REM Tiled 16x16
    %EXE% --rows %%S --cols %%S --iters !ITERS! --wrap 0 --tiled 1 --lx 16 --ly 16 --repeat 5 --warmup 1 --csv --out %OUT%

    echo.
)

echo Benchmark finished.
echo Results saved to %OUT%
endlocal