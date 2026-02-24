CC=gcc
CFLAGS=-O2 -Wall -Wextra -Iinclude
LDFLAGS=-lOpenCL -lm

SRC=main.c src/kernel_loader.c src/prime_check_opencl.c src/prime_check_utils.c

all: main.exe

main.exe: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f main.exe
