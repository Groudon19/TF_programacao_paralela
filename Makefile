# Trabalho Final de Programacao Paralela -- Max-Cut com Simulated Annealing
# Ferramentas: MPI + OpenMP
#
#   make                 compila ./maxcut
#   make kit             compila o gerador e o verificador do kit (tf-maxcut)
#   make run grafo.txt saida.txt     (2 ranks MPI)
#   make babitonga grafo.txt saida.txt   (ajustado para os nos da Baia da Babitonga)
#   make test            roda um teste curto com instancias do kit
#   make avaliar         roda todas as instancias do kit (ver abaixo)

MPICXX   ?= mpicxx
CXXFLAGS ?= -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra
KIT      ?= tf-maxcut
NP       ?= 2

all: maxcut

maxcut: maxcut.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

kit:
	$(MAKE) -C $(KIT)

# make run <entrada> <saida>: as duas palavras depois de "run" sao os arquivos.
run: maxcut
	mpirun --bind-to none --oversubscribe -np 2 ./maxcut $(word 2,$(MAKECMDGOALS)) $(word 3,$(MAKECMDGOALS))


babitonga: maxcut
	OMP_NUM_THREADS=8 mpirun --bind-to none --oversubscribe -np 2 ./maxcut $(word 2,$(MAKECMDGOALS)) $(word 3,$(MAKECMDGOALS))

# Faz o make ignorar essas duas palavras (senao ele tenta construi-las).
%:
	@:

test: maxcut kit
	MAXCUT_TIME=2 mpirun --bind-to none --oversubscribe -np 2 ./maxcut $(KIT)/entradas/exemplo_n5.txt saida_teste.txt
	$(KIT)/verificador $(KIT)/entradas/exemplo_n5.txt saida_teste.txt
	MAXCUT_TIME=5 mpirun --bind-to none --oversubscribe -np $(NP) ./maxcut $(KIT)/entradas/n1000_s1.txt saida_teste.txt
	$(KIT)/verificador $(KIT)/entradas/n1000_s1.txt saida_teste.txt

# Avaliacao com o kit oficial do professor (pasta tf-maxcut): roda cada
# instancia de exemplo e imprime o corte calculado pelo verificador oficial.
# Uso: make avaliar [TEMPO=80]   (TEMPO = segundos de annealing por instancia)
TEMPO ?= 80
avaliar: maxcut kit
	@for f in $(KIT)/entradas/*.txt; do \
	  MAXCUT_QUIET=1 MAXCUT_TIME=$(TEMPO) mpirun --bind-to none --oversubscribe -np $(NP) ./maxcut $$f saida_avaliacao.txt; \
	  printf "%-32s corte = " $$f; $(KIT)/verificador $$f saida_avaliacao.txt; \
	done; rm -f saida_avaliacao.txt

clean:
	rm -f maxcut saida*.txt
	$(MAKE) -C $(KIT) clean

.PHONY: all kit run babitonga test avaliar clean
