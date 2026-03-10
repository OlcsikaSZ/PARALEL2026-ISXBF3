@echo off
if not exist results mkdir results

echo Meresek indulnak...
main.exe 1000000 64 0
main.exe 1000000 128 0
main.exe 1000000 256 0
main.exe 1000000 64 1
main.exe 1000000 128 1
main.exe 1000000 256 1

main.exe 5000000 64 0
main.exe 5000000 128 0
main.exe 5000000 256 0
main.exe 5000000 64 1
main.exe 5000000 128 1
main.exe 5000000 256 1

main.exe 10000000 64 0
main.exe 10000000 128 0
main.exe 10000000 256 0
main.exe 10000000 64 1
main.exe 10000000 128 1
main.exe 10000000 256 1

echo Kesz. Nezd meg a results\results.csv fajlt.
pause