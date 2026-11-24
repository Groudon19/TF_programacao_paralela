# TF de Programação Paralela: Max-Cut com Simulated Annealing (MPI + OpenMP)

- `maxcut.cpp` resolve o Max-Cut com *Simulated Annealing*, usando MPI e OpenMP.
- `Makefile` compila e executa o programa.

## Executar na Baia da Babitonga

Na pasta do projeto, compile e execute:

```
make -B maxcut
make babitonga grafo.txt saida.txt
```

O programa termina sozinho em cerca de TEMPO s e grava em `saida.txt` uma linha com `n` caracteres `0` ou `1`.
O log vai para a saída de erro.

## Requisitos

Open MPI (`mpicxx` e `mpirun`), GCC com OpenMP e `make`.
