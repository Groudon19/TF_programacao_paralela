#   make                 compila ./maxcut
#   make run grafo.txt saida.txt     
#   make babitonga grafo.txt saida.txt   (ajustado para os nos da Baia da Babitonga)


MPICXX   ?= mpicxx
CXXFLAGS ?= -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra
NP       ?= 2

all: maxcut

maxcut: maxcut.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

# make run <entrada> <saida>: as duas palavras depois de "run" sao os arquivos.
run: maxcut
	mpirun --bind-to none --oversubscribe -np 2 ./maxcut $(word 2,$(MAKECMDGOALS)) $(word 3,$(MAKECMDGOALS))


babitonga: maxcut
	OMP_NUM_THREADS=8 mpirun --bind-to none --oversubscribe -np 2 ./maxcut $(word 2,$(MAKECMDGOALS)) $(word 3,$(MAKECMDGOALS))

# Faz o make ignorar essas duas palavras (senao ele tenta construi-las).
%:
	@:

clean:
	rm -f maxcut saida*.txt

.PHONY: all run babitonga  clean
