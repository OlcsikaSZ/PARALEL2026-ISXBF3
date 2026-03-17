@echo off
setlocal EnableDelayedExpansion

set EXE=gol_opencl
set OUT=results_wrap.csv

if exist %OUT% del %OUT%

echo Running wrap comparison benchmarks...
echo Output file: %OUT%
echo.

for %%S in (1024 2048) do (
    if %%S==1024 set ITERS=1000
    if %%S==2048 set ITERS=500

    echo ==== SIZE %%S x %%S, iters=!ITERS! ====

    REM Naive wrap=0
    %EXE% --rows %%S --cols %%S --iters !ITERS! --wrap 0 --lx 16 --ly 16 --repeat 5 --warmup 1 --csv --out %OUT%

    REM Naive wrap=1
    %EXE% --rows %%S --cols %%S --iters !ITERS! --wrap 1 --lx 16 --ly 16 --repeat 5 --warmup 1 --csv --out %OUT%

    REM Tiled 16x16 wrap=0
    %EXE% --rows %%S --cols %%S --iters !ITERS! --wrap 0 --tiled 1 --lx 16 --ly 16 --repeat 5 --warmup 1 --csv --out %OUT%

    REM Tiled 16x16 wrap=1
    %EXE% --rows %%S --cols %%S --iters !ITERS! --wrap 1 --tiled 1 --lx 16 --ly 16 --repeat 5 --warmup 1 --csv --out %OUT%

    echo.
)

echo Wrap benchmark finished.
echo Results saved to %OUT%
endlocal