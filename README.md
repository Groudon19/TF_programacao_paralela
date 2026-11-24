# Trabalho Final de Programação Paralela: Max-Cut com Simulated Annealing

**Ferramentas usadas:** MPI + OpenMP, as duas no caminho crítico do cálculo.

## Como compilar e executar

```bash
make run grafo.txt saida.txt           # compila (se preciso) e executa com 2 ranks MPI
```

Isso equivale a `make` seguido de
`mpirun --bind-to none --oversubscribe -np 2 ./maxcut grafo.txt saida.txt`. É preciso ter `mpicxx`
com suporte a OpenMP. Os caminhos relativos partem da pasta do projeto.

O programa recebe os dois argumentos pedidos: o grafo e o arquivo de saída. Ele para sozinho
quando o annealing chega a **80 s** desde o início, deixando folga para os 90 s do limite.
O log vai para `stderr` e a saída tem uma única linha com `n` caracteres `0`/`1`.

- **Threads por rank:** se `OMP_NUM_THREADS` não estiver definido, cada rank usa
  `núcleos físicos / ranks no nó`, limitado a 8.
- **Quantos ranks usar:** a recomendação é `NP ≈ núcleos físicos / 6..8`.
- **Afinidade de CPU:** o programa também funciona com `mpirun -np X` sem `--bind-to none`, porque
  ele mesmo libera a afinidade de CPU (veja `release_cpu_binding` em `maxcut.cpp`).

## Resultados nas instâncias oficiais (kit do professor, pasta `tf-maxcut`)

Os cortes foram calculados pelo **verificador oficial** (`tf-maxcut/verificador.c`), com
`mpirun --bind-to none -np 2` e 3 execuções por instância, como na avaliação.
A referência é o Algoritmo 1 sequencial (10.000 varreduras, melhor de 3 sementes), tirada do
README do kit.

| Instância | Referência | Execução 1 | Execução 2 | Execução 3 | **Mediana** | Tempo de relógio |
|---|---:|---:|---:|---:|---:|---:|
| `n20000_s1` (Média) | 23.668 | 23.759 | 23.766 | 23.749 | **23.759** (+91) | 80,2–80,3 s |
| `n100000_s1` (Grande) | 116.548 | 117.018 | 117.030 | 116.985 | **117.018** (+470) | 80,2–80,3 s |

Nas instâncias pequenas do kit, o programa chegou ao valor de referência nas 3 execuções:
`exemplo_n5` = 3, `n20_s1` = 18, `n100_s1` = 118 e `n1000_s1` = 1.178.
Para repetir esse teste: `make avaliar` (ou `make avaliar TEMPO=10`, para um teste rápido).

Ferramentas de teste (todas do kit, pasta `tf-maxcut`):

```bash
make kit                                            # compila o gerador e o verificador do kit
./tf-maxcut/gerador 100000 42 > g.txt               # grafo novo (n, semente [, grau médio = 10])
./tf-maxcut/verificador g.txt saida.txt             # confere o formato e imprime o valor do corte
make test                                           # teste curto: exemplo_n5 + n1000_s1
```

## Estratégia de paralelização

### 1. OpenMP: paralelismo dentro da cadeia (cadeia única com coloração)

Como o próprio enunciado observa, rodar mais cadeias independentes rende pouco nas instâncias grandes.
O que melhora o corte é fazer **mais varreduras por segundo**. Por isso:

1. **Coloração gulosa do grafo** (Welsh-Powell, maior grau primeiro). Vértices da mesma cor não são
   vizinhos. Os grafos do trabalho saem com cerca de 8 cores.
2. **Renumeração dos vértices por cor.** Cada cor fica contígua na memória, e um `#pragma omp for`
   divide a cor entre as threads.
3. **Uma varredura percorre as cores em sequência.** Entre uma cor e a próxima há a barreira
   implícita do `omp for`. Como vértices da mesma cor são independentes, a execução equivale a um SA
   sequencial válido, só que em outra ordem de visita.
4. **Ganho incremental.** Cada vértice guarda o seu ganho `Δi` (como no Algoritmo 1 do enunciado).
   Quando `i` troca de lado, os vizinhos são atualizados com `Δj += 2·wij·si·sj`. Dois vértices da
   mesma cor podem ter um vizinho em comum, então essa soma é `omp atomic`. Testamos sem o atomic e
   o corte ficou errado.
5. **Aceitação por tabela.** `e^{Δ/T}` é pré-calculado para cada valor inteiro de `Δ` a cada
   varredura. Cada thread tem o seu gerador xorshift.

### 2. MPI: várias cadeias síncronas

- **Uma cadeia por rank.** Todas seguem o mesmo cronograma de temperatura, guiado pelo relógio:
  `T = T0·(T1/T0)^(t/tempo_total)`, com `T0 = 2,0` e `T1 = 0,1`.
- **Sincronização a cada ~2 s.**
  1. `MPI_Allgather` junta os cortes de todos os ranks.
  2. `MPI_Bcast` envia o estado do melhor rank.
  3. A **metade pior** dos ranks recomeça desse estado, com a sua própria semente.

  É a variante síncrona citada no enunciado [1].
- **Final.** Cada rank faz uma busca local gulosa (só aceita `Δ > 0`). Uma redução (`MPI_Allreduce`)
  escolhe o melhor corte global, e o rank dono grava o arquivo.
- **Leitura.** O rank 0 lê o arquivo inteiro com um único `fread` e o distribui por `MPI_Bcast`.
  Um parser próprio substitui o `fscanf`, e a preparação leva cerca de 0,1 s.

## Testes de desempenho com o nosso gerador (i7-12700H, WSL2 Ubuntu, OpenMPI 5, g++ 15)

Os grafos foram gerados com um gerador próprio, no mesmo formato do enunciado. As sementes secretas da competição não são
conhecidas, então os valores absolutos mudam de grafo para grafo.

Varreduras por segundo na instância de 100.000 vértices, com 1 rank:

| Versão | 1 thread | 6 threads |
|---|---|---|
| Ganho recalculado pelos vizinhos a cada visita | 684/s | 2.765/s |
| Ganho incremental (versão final) | 2.195/s | ~6.000–9.000/s |

A referência sequencial do enunciado faz cerca de 500 varreduras/s (2 ms cada).

Execuções completas (80 s de annealing):

| Instância | Ranks × threads | Varreduras por rank | Corte | Tempo de relógio (com mpirun) |
|---|---|---|---|---|
| Grande (100k) | 1 × 8 | 394.531 | 117.750 | 80,2 s |
| Grande (100k) | 2 × 5 | 297.395 | 117.703 | 80,3 s |
| Grande (100k) | 3 × 3 | 265.333 | 117.726 | 80,2 s |
| Média (20k) | 1 × 8 | 2.573.294 | 23.410 | 80,2 s |
| Média (20k) | 3 × 3 | 1.446.276 | 23.405 | 80,2 s |

Todas as saídas passaram na verificação de formato e de valor do corte. Nos grafos pequenos (`n = 20`) e no exemplo
do enunciado, o programa encontrou o ótimo da força bruta.

**Comparação justa, no mesmo grafo de 100 mil vértices:** o Algoritmo 1 do enunciado,
implementado fielmente (sequencial, T0 = 1,5, T1 = 0,3):

| Versão | Varreduras | Tempo | Corte |
|---|---|---|---|
| Referência sequencial | 1.000 | 2,1 s | 116.607 |
| Referência sequencial | 10.000 | 21,1 s | 117.177 |
| Referência sequencial | 38.000 | 78,7 s | 117.260 |
| **Este trabalho (MPI + OpenMP)** | ~395.000 por cadeia | 80 s | **117.750** |

Com o mesmo tempo, o programa paralelo faz **+490 pontos**. A referência ganhou só 83 pontos indo
de 10 mil para 38 mil varreduras, então perto do topo cada ponto custa muito tempo. Nesta máquina
(laptop com núcleos P e E), as combinações de ranks × threads ficaram todas dentro do ruído
(cerca de ±40 pontos entre sementes).

## Parâmetros opcionais (variáveis de ambiente)

| Variável | Padrão | Significado |
|---|---|---|
| `MAXCUT_TIME` | 80 | segundos (desde o início) em que o annealing termina |
| `MAXCUT_T0` / `MAXCUT_T1` | 2.0 / 0.1 | temperatura inicial / final |
| `MAXCUT_SYNC` | 2.0 | intervalo entre sincronizações MPI (s) |
| `MAXCUT_SEED` | aleatória | semente fixa, para reproduzir uma execução |
| `MAXCUT_QUIET` | — | se definida, não imprime o log |
| `OMP_NUM_THREADS` | automático | threads por rank |
