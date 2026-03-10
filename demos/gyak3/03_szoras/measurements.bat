@echo off
if not exist results mkdir results

echo Meresek indulnak...

main.exe 1000000 64
main.exe 1000000 128
main.exe 1000000 256

main.exe 5000000 64
main.exe 5000000 128
main.exe 5000000 256

main.exe 10000000 64
main.exe 10000000 128
main.exe 10000000 256

main.exe 20000000 64
main.exe 20000000 128
main.exe 20000000 256

echo Kesz. Nezd meg a results\results.csv fajlt.
pause