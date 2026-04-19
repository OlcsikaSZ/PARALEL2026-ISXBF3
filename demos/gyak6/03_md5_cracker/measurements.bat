@echo off
make clean
make
if errorlevel 1 exit /b 1

md5_cracker.exe 2000000 1048576 32 > results\run.txt
