# Trabalho Final de Programacao Paralela -- Max-Cut com Simulated Annealing
# Ferramentas: MPI + OpenMP
#
#   make                 compila ./maxcut
#   make tools           compila o gerador de grafos de teste
#   make run IN=grafo.txt OUT=saida.txt [NP=2]
#   make test            gera grafos de exemplo e roda um teste curto

MPICXX   ?= mpicxx
CXXFLAGS ?= -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra
NP       ?= 2
IN       ?= grafos/treino.txt
OUT      ?= saida.txt

all: maxcut

maxcut: maxcut.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

tools: tools/gen_graph

tools/gen_graph: tools/gen_graph.cpp
	$(CXX) -O2 -std=c++17 -o $@ $<

# --bind-to none: cada rank usa varias threads OpenMP, entao nao pode ficar
# preso a um unico nucleo. --oversubscribe permite NP maior que o n. de slots.
run: maxcut
	mpirun --bind-to none --oversubscribe -np $(NP) ./maxcut $(IN) $(OUT)

grafos/exemplo.txt:
	@mkdir -p grafos
	printf "5 6\n1 2 1\n1 3 -1\n2 3 1\n2 4 1\n3 5 1\n4 5 -1\n" > $@

grafos/treino.txt: tools/gen_graph
	@mkdir -p grafos
	./tools/gen_graph 1000 5000 1 > $@

test: maxcut grafos/exemplo.txt grafos/treino.txt
	MAXCUT_TIME=2 mpirun --bind-to none --oversubscribe -np 2 ./maxcut grafos/exemplo.txt saida_exemplo.txt
	python3 tools/verify.py grafos/exemplo.txt saida_exemplo.txt
	MAXCUT_TIME=5 mpirun --bind-to none --oversubscribe -np $(NP) ./maxcut grafos/treino.txt saida_treino.txt
	python3 tools/verify.py grafos/treino.txt saida_treino.txt

# Avaliacao com o kit oficial do professor (pasta tf-maxcut): roda cada
# instancia de exemplo e imprime o corte calculado pelo verificador oficial.
# Uso: make avaliar [TEMPO=80]   (TEMPO = segundos de annealing por instancia)
KIT   ?= tf-maxcut
TEMPO ?= 80
avaliar: maxcut
	$(MAKE) -C $(KIT)
	@for f in $(KIT)/entradas/*.txt; do \
	  MAXCUT_QUIET=1 MAXCUT_TIME=$(TEMPO) mpirun --bind-to none --oversubscribe -np $(NP) ./maxcut $$f saida_avaliacao.txt; \
	  printf "%-32s corte = " $$f; $(KIT)/verificador $$f saida_avaliacao.txt; \
	done; rm -f saida_avaliacao.txt

clean:
	rm -f maxcut tools/gen_graph saida*.txt

.PHONY: all tools run test avaliar clean
