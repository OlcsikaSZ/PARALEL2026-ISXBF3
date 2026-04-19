@echo off
make clean
make
if errorlevel 1 exit /b 1

miller_rabin_opencl.exe 200000 1000000000001 > results\run.txt
