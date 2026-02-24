CC=gcc
CFLAGS=-Iinclude -I"C:\Path\To\OpenCL\include"
LDFLAGS=-L"C:\Path\To\OpenCL\lib" -lOpenCL

SRC=src/kernel_loader.c
OBJ=$(SRC:.c=.o)

all: main.exe

main.exe: main.c $(SRC)
	$(CC) main.c $(SRC) $(CFLAGS) $(LDFLAGS) -o main.exe

clean:
	del *.exe