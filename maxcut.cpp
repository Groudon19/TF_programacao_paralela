// =============================================================================
//  Corte Maximo (Max-Cut) com Simulated Annealing paralelo -- MPI + OpenMP
//  Programacao Paralela 2026/2 -- UDESC
//
//  Uso:  mpirun -np <P> ./maxcut <grafo.txt> <saida.txt>
//
//  Estrategia de paralelizacao (duas ferramentas, ambas no caminho critico):
//
//   * OpenMP -- paralelismo DENTRO de uma cadeia (Tabela III, "cadeia unica
//     com coloracao"). O grafo e colorido de forma gulosa: vertices da mesma
//     cor nao sao vizinhos, entao podem decidir trocar de lado ao mesmo tempo
//     sem condicao de corrida. Uma varredura percorre as cores em sequencia,
//     com uma barreira entre cores, e os vertices de cada cor sao divididos
//     entre as threads. O resultado e exatamente um SA sequencial valido (com
//     uma ordem de visita diferente), so que com muito mais varreduras/s.
//
//   * MPI -- uma cadeia por rank (cadeias multiplas SINCRONAS, ref. [1] do
//     enunciado). Todas seguem o mesmo cronograma de temperatura, guiado pelo
//     relogio. Em pontos de sincronizacao periodicos (a cada ~2 s) os ranks
//     trocam o valor do corte; a metade pior recebe uma copia do estado do
//     melhor rank e continua dali com sua propria semente. No final, o melhor
//     corte global e escolhido por reducao e o rank dono grava a saida.
//
//  Parametros opcionais por variavel de ambiente (todos tem padrao):
//     MAXCUT_TIME   instante (s, desde o inicio) em que o annealing termina [80]
//     MAXCUT_T0     temperatura inicial                                     [2.0]
//     MAXCUT_T1     temperatura final                                       [0.10]
//     MAXCUT_SYNC   intervalo entre sincronizacoes MPI (s)                  [2.0]
//     MAXCUT_SEED   semente fixa (para reproduzir execucoes)                [aleatoria]
//     MAXCUT_QUIET  se definido, nao imprime o log em stderr
//     OMP_NUM_THREADS  threads por rank (padrao: nucleos do no / ranks no no)
// =============================================================================
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

using std::vector;

// ----------------------------------------------------------------------------
// Utilidades
// ----------------------------------------------------------------------------
static int g_rank = 0, g_size = 1;
static bool g_quiet = false;

#define LOG(...)                                         \
    do {                                                 \
        if (g_rank == 0 && !g_quiet) {                   \
            std::fprintf(stderr, __VA_ARGS__);           \
            std::fflush(stderr);                         \
        }                                                \
    } while (0)

static double env_double(const char *name, double def) {
    const char *s = std::getenv(name);
    return (s && *s) ? std::atof(s) : def;
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static uint64_t splitmix64(uint64_t &x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Gerador xorshift64* (rapido, um por thread)
struct alignas(64) Rng {
    uint64_t s;
    inline uint32_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return (uint32_t)((s * 0x2545F4914F6CDD1DULL) >> 32);
    }
};

// ----------------------------------------------------------------------------
// Grafo em formato CSR, ja renumerado por cor
// ----------------------------------------------------------------------------
struct Graph {
    int n = 0;
    long long m = 0;
    int maxdeg = 0;
    vector<int> off;          // off[i]..off[i+1]: vizinhos do vertice i
    vector<uint32_t> adj;     // (vizinho << 1) | (peso == +1)
    vector<int> colorStart;   // vertices da cor c: colorStart[c]..colorStart[c+1]
    vector<int> origOf;       // id interno -> id original (0-based)
    int ncolors = 0;
};

// Le o arquivo inteiro no rank 0 e distribui os bytes para os demais ranks.
static vector<char> read_and_broadcast(const char *path) {
    vector<char> buf;
    long long size = 0;
    if (g_rank == 0) {
        FILE *f = std::fopen(path, "rb");
        if (!f) {
            std::fprintf(stderr, "Erro: nao foi possivel abrir %s\n", path);
            size = -1;
        } else {
            std::fseek(f, 0, SEEK_END);
            size = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            buf.resize(size + 1);
            size_t rd = std::fread(buf.data(), 1, size, f);
            std::fclose(f);
            size = (long long)rd;
        }
    }
    MPI_Bcast(&size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if (size < 0) {
        MPI_Finalize();
        std::exit(1);
    }
    buf.resize(size + 1);
    // Bcast em blocos para nao estourar o limite de int do MPI
    const long long CH = 1LL << 30;
    for (long long p = 0; p < size; p += CH) {
        int cnt = (int)std::min(CH, size - p);
        MPI_Bcast(buf.data() + p, cnt, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    buf[size] = '\0';
    return buf;
}

// Parser de inteiros simples e rapido (bem mais rapido que fscanf)
struct Parser {
    const char *p, *end;
    bool next(long long &out) {
        while (p < end && !(*p == '-' || *p == '+' || (*p >= '0' && *p <= '9'))) {
            if (*p == '#') {  // ignora comentarios ate o fim da linha
                while (p < end && *p != '\n') ++p;
            } else {
                ++p;
            }
        }
        if (p >= end) return false;
        bool neg = false;
        if (*p == '-' || *p == '+') {
            neg = (*p == '-');
            ++p;
        }
        long long v = 0;
        while (p < end && *p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
        out = neg ? -v : v;
        return true;
    }
};

static Graph build_graph(const vector<char> &text) {
    Parser ps{text.data(), text.data() + text.size() - 1};
    long long n = 0, m = 0;
    if (!ps.next(n) || !ps.next(m) || n <= 0) {
        if (g_rank == 0) std::fprintf(stderr, "Erro: cabecalho invalido\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    vector<int> eu, ev;
    vector<int8_t> ew;
    eu.reserve(m);
    ev.reserve(m);
    ew.reserve(m);
    for (long long k = 0; k < m; ++k) {
        long long u, v, w;
        if (!ps.next(u) || !ps.next(v) || !ps.next(w)) break;
        if (u < 1 || u > n || v < 1 || v > n || u == v || w == 0) continue;  // laco nunca e cortado
        eu.push_back((int)(u - 1));
        ev.push_back((int)(v - 1));
        ew.push_back(w > 0 ? 1 : -1);
    }

    // CSR no numero original
    const int N = (int)n;
    vector<int> deg(N, 0);
    for (size_t k = 0; k < eu.size(); ++k) {
        deg[eu[k]]++;
        deg[ev[k]]++;
    }
    vector<int> off0(N + 1, 0);
    for (int i = 0; i < N; ++i) off0[i + 1] = off0[i] + deg[i];
    vector<int> nb0(off0[N]);
    vector<int8_t> w0(off0[N]);
    {
        vector<int> pos(off0.begin(), off0.end() - 1);
        for (size_t k = 0; k < eu.size(); ++k) {
            int a = eu[k], b = ev[k];
            nb0[pos[a]] = b; w0[pos[a]++] = ew[k];
            nb0[pos[b]] = a; w0[pos[b]++] = ew[k];
        }
    }

    // Coloracao gulosa (Welsh-Powell: maior grau primeiro). Deterministica,
    // entao todos os ranks obtem exatamente a mesma numeracao interna.
    vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return deg[a] > deg[b]; });
    vector<int> color(N, -1);
    int maxdeg = 0;
    for (int i = 0; i < N; ++i) maxdeg = std::max(maxdeg, deg[i]);
    vector<int> stamp(maxdeg + 2, -1);
    int ncolors = 0;
    for (int v : order) {
        for (int e = off0[v]; e < off0[v + 1]; ++e) {
            int c = color[nb0[e]];
            if (c >= 0) stamp[c] = v;
        }
        int c = 0;
        while (stamp[c] == v) ++c;
        color[v] = c;
        ncolors = std::max(ncolors, c + 1);
    }

    // Renumera: vertices da mesma cor ficam contiguos (melhor para o "omp for")
    Graph g;
    g.n = N;
    g.m = (long long)eu.size();
    g.maxdeg = maxdeg;
    g.ncolors = ncolors;
    g.colorStart.assign(ncolors + 1, 0);
    for (int i = 0; i < N; ++i) g.colorStart[color[i] + 1]++;
    for (int c = 0; c < ncolors; ++c) g.colorStart[c + 1] += g.colorStart[c];
    vector<int> newId(N);
    g.origOf.resize(N);
    {
        vector<int> pos(g.colorStart.begin(), g.colorStart.end() - 1);
        for (int i = 0; i < N; ++i) {
            int id = pos[color[i]]++;
            newId[i] = id;
            g.origOf[id] = i;
        }
    }
    g.off.assign(N + 1, 0);
    for (int id = 0; id < N; ++id) g.off[id + 1] = g.off[id] + deg[g.origOf[id]];
    g.adj.resize(g.off[N]);
    for (int id = 0; id < N; ++id) {
        int v = g.origOf[id];
        int p = g.off[id];
        for (int e = off0[v]; e < off0[v + 1]; ++e)
            g.adj[p++] = ((uint32_t)newId[nb0[e]] << 1) | (w0[e] > 0 ? 1u : 0u);
    }
    return g;
}

// ----------------------------------------------------------------------------
// Funcoes do Simulated Annealing (spins s_i em {-1,+1}; s_i=+1 => lado 1)
//
// Ganho de trocar i de lado (eq. 2 do enunciado):
//     Delta_i = sum_j w_ij (2[x_j = x_i] - 1) = s_i * sum_j w_ij s_j
// ----------------------------------------------------------------------------
static inline int gain(const Graph &g, const int8_t *s, int i) {
    int h = 0;
    const uint32_t *a = g.adj.data();
    for (int e = g.off[i], E = g.off[i + 1]; e < E; ++e) {
        uint32_t x = a[e];
        int sj = s[x >> 1];
        h += (x & 1u) ? sj : -sj;
    }
    return s[i] * h;
}

// Recalcula do zero o ganho Delta_i de todos os vertices (O(m), em paralelo).
// Usado no inicio e sempre que o estado e substituido por outro (troca MPI).
static void compute_gains(const Graph &g, const int8_t *s, int *D) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < g.n; ++i) D[i] = gain(g, s, i);
}

// Troca o vertice i de lado e atualiza Delta de i e de seus vizinhos.
// Para o vizinho j:  Delta_j += 2 * w_ij * s_i(novo) * s_j.
// Os vizinhos tem cor diferente da de i, entao ninguem le D[j] nesta fase;
// mas dois vertices da mesma cor podem ter um vizinho j em comum, por isso
// a soma em D[j] e atomica.
static inline void flip(const Graph &g, int8_t *s, int *D, int i) {
    const int si = -s[i];
    s[i] = (int8_t)si;
    D[i] = -D[i];
    const uint32_t *a = g.adj.data();
    for (int e = g.off[i], E = g.off[i + 1]; e < E; ++e) {
        uint32_t x = a[e];
        int j = (int)(x >> 1);
        int delta = ((x & 1u) ? 2 : -2) * si * s[j];
#pragma omp atomic
        D[j] += delta;
    }
}

// Valor exato do corte (cada aresta aparece duas vezes no CSR)
static long long cut_value(const Graph &g, const int8_t *s) {
    long long total = 0;
#pragma omp parallel for schedule(static) reduction(+ : total)
    for (int i = 0; i < g.n; ++i) {
        int si = s[i];
        for (int e = g.off[i]; e < g.off[i + 1]; ++e) {
            uint32_t x = g.adj[e];
            if (s[x >> 1] != si) total += (x & 1u) ? 1 : -1;
        }
    }
    return total / 2;
}

// Tabela de aceitacao: aceita Delta = -k se rng < thr[k]  (thr[k] = 2^32 e^{-k/T})
static void fill_threshold(vector<uint32_t> &thr, double T) {
    for (size_t k = 0; k < thr.size(); ++k) {
        double p = (k == 0) ? 1.0 : std::exp(-(double)k / T);
        double v = p * 4294967296.0;
        thr[k] = (v >= 4294967295.0) ? 0xFFFFFFFFu : (uint32_t)v;
    }
}

// Busca local gulosa final: troca enquanto houver ganho estritamente positivo
static void greedy_polish(const Graph &g, int8_t *s, double deadline) {
    for (int it = 0; it < 1000 && now_s() < deadline; ++it) {
        long long flips = 0;
#pragma omp parallel reduction(+ : flips)
        for (int c = 0; c < g.ncolors; ++c) {
#pragma omp for schedule(static)
            for (int i = g.colorStart[c]; i < g.colorStart[c + 1]; ++i) {
                if (gain(g, s, i) > 0) {
                    s[i] = (int8_t)-s[i];
                    ++flips;
                }
            }
        }
        if (flips == 0) break;
    }
}

// ----------------------------------------------------------------------------
// Ajuste do numero de threads por rank
// ----------------------------------------------------------------------------
static void setup_threads(MPI_Comm local) {
    int local_size = 1;
    MPI_Comm_size(local, &local_size);
    const char *env = std::getenv("OMP_NUM_THREADS");
    if (env && *env) return;  // o usuario escolheu explicitamente
    int ncpu = omp_get_num_procs();
    int ncores = ncpu;
#ifdef __linux__
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 0) ncpu = (int)online;
    // Conta nucleos fisicos: hyperthreading atrapalha um laco cheio de barreiras.
    int phys = 0;
    for (int c = 0; c < ncpu; ++c) {
        char path[128];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        FILE *f = std::fopen(path, "r");
        if (!f) continue;
        int first = -1;
        if (std::fscanf(f, "%d", &first) == 1 && first == c) ++phys;  // 1a thread de cada nucleo
        std::fclose(f);
    }
    ncores = phys > 0 ? phys : ncpu;
#endif
    // Medimos que a varredura colorida deixa de escalar por volta de 6-8
    // threads por cadeia; nucleos alem disso rendem mais como outros ranks MPI.
    const int MAX_THREADS_PER_RANK = 8;
    int t = std::max(1, std::min(MAX_THREADS_PER_RANK, ncores / local_size));
    omp_set_num_threads(t);
}

// O mpirun, por padrao, prende cada rank a 1 nucleo (ou poucos). O runtime do
// OpenMP (libgomp) le essa mascara ao carregar o programa, ANTES do main: ve
// mais threads que CPUs e faz as threads dormirem nas barreiras -- medimos a
// varredura ~3.5x mais lenta. Se a mascara estiver restrita, liberamos todas as
// CPUs e reexecutamos o proprio binario uma unica vez (antes do MPI_Init, o
// processo e o mesmo PID, entao o mpirun nao percebe). Com "mpirun --bind-to
// none" (como no Makefile) isso nao acontece.
static void release_cpu_binding(char **argv) {
#ifdef __linux__
    if (std::getenv("MAXCUT_REEXEC")) return;
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    cpu_set_t cur;
    if (online <= 1 || sched_getaffinity(0, sizeof(cur), &cur) != 0) return;
    if (CPU_COUNT(&cur) >= online) return;
    cpu_set_t all;
    CPU_ZERO(&all);
    for (long c = 0; c < online && c < CPU_SETSIZE; ++c) CPU_SET(c, &all);
    if (sched_setaffinity(0, sizeof(all), &all) != 0) return;
    setenv("MAXCUT_REEXEC", "1", 1);
    execv("/proc/self/exe", argv);  // so retorna se falhar; entao seguimos assim mesmo
#else
    (void)argv;
#endif
}

// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
    const double t_start = now_s();
    release_cpu_binding(argv);
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    g_quiet = std::getenv("MAXCUT_QUIET") != nullptr;

    if (argc < 3) {
        if (g_rank == 0) std::fprintf(stderr, "Uso: %s <grafo> <saida>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &local);
    setup_threads(local);
    const int nthreads = omp_get_max_threads();

    const double ANNEAL_END = env_double("MAXCUT_TIME", 80.0);
    const double T0 = env_double("MAXCUT_T0", 2.0);
    const double T1 = env_double("MAXCUT_T1", 0.10);
    const double SYNC = std::max(0.05, env_double("MAXCUT_SYNC", 2.0));

    // ---------------- leitura e preparacao ----------------
    vector<char> text = read_and_broadcast(argv[1]);
    Graph g = build_graph(text);
    vector<char>().swap(text);
    const int n = g.n;
    LOG("[maxcut] n=%d m=%lld maxdeg=%d cores=%d | ranks=%d threads/rank=%d | setup %.2fs\n",
        n, g.m, g.maxdeg, g.ncolors, g_size, nthreads, now_s() - t_start);

    // Relogio comum: rank 0 decide quanto tempo resta e todos usam o mesmo valor
    MPI_Barrier(MPI_COMM_WORLD);
    double remaining = ANNEAL_END - (now_s() - t_start);
    MPI_Bcast(&remaining, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double a_start = now_s();
    const double budget = std::max(0.5, remaining);
    const int K = std::max(1, (int)std::lround(budget / SYNC));  // n. de sincronizacoes

    // Sementes: livres (relogio + rank), ou fixas via MAXCUT_SEED
    uint64_t seed;
    if (std::getenv("MAXCUT_SEED")) seed = (uint64_t)std::atoll(std::getenv("MAXCUT_SEED"));
    else seed = (uint64_t)std::random_device{}() ^ ((uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    seed ^= 0x9E3779B97F4A7C15ULL * (uint64_t)(g_rank + 1);
    vector<Rng> rngs(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        rngs[t].s = splitmix64(seed);
        if (rngs[t].s == 0) rngs[t].s = 1;
    }

    // Estado inicial aleatorio
    vector<int8_t> spin(n), best(n), recv(n);
    {
        Rng r{splitmix64(seed) | 1};
        for (int i = 0; i < n; ++i) spin[i] = (r.next() & 1u) ? 1 : -1;
    }
    long long bestCut = cut_value(g, spin.data());
    best = spin;
    vector<int> D(n);  // D[i] = ganho de trocar i de lado
    compute_gains(g, spin.data(), D.data());

    vector<uint32_t> thr(g.maxdeg + 1);
    long long sweeps = 0;
    const double logRatio = std::log(T1 / T0);
    // Corte atual, mantido somando o ganho de cada troca aceita. Permite
    // guardar a melhor solucao assim que ela aparece (e nao so nas
    // sincronizacoes), o que importa em grafos pequenos, onde o estado muda
    // muito entre uma sincronizacao e outra.
    long long curCut = bestCut;

    // ---------------- annealing ----------------
    for (int k = 1; k <= K; ++k) {
        const double seg_end = a_start + budget * k / K;
        int stop = 0;
        double T = T0;
        long long seg_sweeps = 0;

#pragma omp parallel
        {
            Rng rng = rngs[omp_get_thread_num()];
            while (true) {
#pragma omp single
                {
                    // aqui todas as threads ja passaram pela barreira da ultima cor
                    if (curCut > bestCut) {
                        bestCut = curCut;
                        std::memcpy(best.data(), spin.data(), n);
                    }
                    double t = now_s();
                    if (t >= seg_end) {
                        stop = 1;
                    } else {
                        double f = (t - a_start) / budget;  // fracao do tempo ja usada
                        T = T0 * std::exp(logRatio * f);
                        fill_threshold(thr, T);
                        ++seg_sweeps;
                    }
                }  // barreira implicita: todos veem stop/thr atualizados
                if (stop) break;

                // Uma varredura: cor por cor, vertices da cor em paralelo
                for (int c = 0; c < g.ncolors; ++c) {
#pragma omp for schedule(static) reduction(+ : curCut)
                    for (int i = g.colorStart[c]; i < g.colorStart[c + 1]; ++i) {
                        int d = D[i];
                        if (d >= 0 || rng.next() < thr[-d]) {
                            flip(g, spin.data(), D.data(), i);
                            curCut += d;
                        }
                    }  // barreira implicita entre cores
                }
            }
            rngs[omp_get_thread_num()] = rng;
        }
        sweeps += seg_sweeps;

        // ---------- sincronizacao MPI entre as cadeias ----------
        long long cut = cut_value(g, spin.data());
        if (cut != curCut && g_rank == 0 && !g_quiet)
            std::fprintf(stderr, "[maxcut] aviso: corte incremental %lld != exato %lld\n", curCut, cut);
        curCut = cut;
        if (cut > bestCut) {
            bestCut = cut;
            best = spin;
        }
        if (g_size > 1) {
            vector<long long> cuts(g_size);
            MPI_Allgather(&cut, 1, MPI_LONG_LONG, cuts.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
            int bestRank = 0;
            for (int r = 1; r < g_size; ++r)
                if (cuts[r] > cuts[bestRank]) bestRank = r;
            // posicao de cada rank no ranking (0 = melhor)
            int myPos = 0;
            for (int r = 0; r < g_size; ++r)
                if (cuts[r] > cut || (cuts[r] == cut && r < g_rank)) ++myPos;
            if (k < K) {
                MPI_Bcast(g_rank == bestRank ? spin.data() : recv.data(), n, MPI_SIGNED_CHAR,
                          bestRank, MPI_COMM_WORLD);
                // A metade pior recomeca a partir do melhor estado atual
                if (g_rank != bestRank && myPos >= (g_size + 1) / 2) {
                    spin = recv;
                    compute_gains(g, spin.data(), D.data());
                    curCut = cuts[bestRank];
                }
            }
            if (k % 5 == 0 || k == K)
                LOG("[maxcut] t=%5.1fs T=%.3f sweeps=%lld melhor_corte=%lld\n", now_s() - t_start, T,
                    sweeps, cuts[bestRank]);
        } else if (k % 5 == 0 || k == K) {
            LOG("[maxcut] t=%5.1fs T=%.3f sweeps=%lld corte=%lld\n", now_s() - t_start, T, sweeps, cut);
        }
    }

    // ---------------- polimento guloso + escolha do melhor global ----------------
    greedy_polish(g, spin.data(), now_s() + 2.0);
    long long cut = cut_value(g, spin.data());
    if (cut > bestCut) {
        bestCut = cut;
        best = spin;
    }
    greedy_polish(g, best.data(), now_s() + 1.0);
    bestCut = cut_value(g, best.data());

    // Reducao do melhor corte global (MAXLOC em dois passos, portavel p/ long long)
    long long globalBest = 0;
    MPI_Allreduce(&bestCut, &globalBest, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    int cand = (bestCut == globalBest) ? g_rank : g_size;
    int owner = 0;
    MPI_Allreduce(&cand, &owner, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (g_rank == owner) {
        vector<char> out(n + 1);
        for (int id = 0; id < n; ++id) out[g.origOf[id]] = (best[id] > 0) ? '1' : '0';
        out[n] = '\n';
        FILE *f = std::fopen(argv[2], "wb");
        if (!f) {
            std::fprintf(stderr, "Erro: nao foi possivel criar %s\n", argv[2]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::fwrite(out.data(), 1, out.size(), f);
        std::fclose(f);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    LOG("[maxcut] FIM corte=%lld (rank %d) | varreduras por rank=%lld (%.0f/s) | tempo total %.2fs\n",
        globalBest, owner, sweeps, sweeps / budget, now_s() - t_start);

    MPI_Comm_free(&local);
    MPI_Finalize();
    return 0;
}
