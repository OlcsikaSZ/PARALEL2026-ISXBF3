CC=gcc
CFLAGS=-O2 -Wall -Wextra -Iinclude
LDFLAGS=-lOpenCL -lm

SRC=main.c src/kernel_loader.c src/vector_add_opencl.c

all: main.exe

main.exe: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f main.exe
