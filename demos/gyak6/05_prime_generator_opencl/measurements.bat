@echo off
make clean
make
if errorlevel 1 exit /b 1

prime_generator_opencl.exe 32 131072 > results\run.txt
