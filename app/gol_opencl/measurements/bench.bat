@echo off
setlocal EnableDelayedExpansion

cd /d %~dp0..
set EXE=gol_opencl.exe
set OUT=measurements\results.csv

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

    echo [CPU] Sequential baseline
    %EXE% --mode cpu_seq --rows %%S --cols %%S --iters !ITERS! --wrap 0 --seed 12345 --repeat 3 --warmup 1 --csv --out %OUT%
    echo.

    echo [GPU] Naive 16x16
    %EXE% --mode gpu --rows %%S --cols %%S --iters !ITERS! --wrap 0 --lx 16 --ly 16 --seed 12345 --repeat 5 --warmup 1 --csv --out %OUT%
    echo.

    echo [GPU] Tiled 4x4
    %EXE% --mode gpu --rows %%S --cols %%S --iters !ITERS! --wrap 0 --tiled 1 --lx 4 --ly 4 --seed 12345 --repeat 5 --warmup 1 --csv --out %OUT%
    echo.

    echo [GPU] Tiled 8x8
    %EXE% --mode gpu --rows %%S --cols %%S --iters !ITERS! --wrap 0 --tiled 1 --lx 8 --ly 8 --seed 12345 --repeat 5 --warmup 1 --csv --out %OUT%
    echo.

    echo [GPU] Tiled 16x16
    %EXE% --mode gpu --rows %%S --cols %%S --iters !ITERS! --wrap 0 --tiled 1 --lx 16 --ly 16 --seed 12345 --repeat 5 --warmup 1 --csv --out %OUT%

    echo.
)

echo Benchmark finished.
echo Results saved to %OUT%
endlocal
