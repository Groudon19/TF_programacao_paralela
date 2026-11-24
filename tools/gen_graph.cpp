// Gerador de grafos aleatorios no formato do trabalho (para testes locais).
// Imita o gerador do enunciado: m arestas distintas sorteadas uniformemente,
// sem lacos, peso +1 ou -1 com igual probabilidade.
//
// Uso: ./gen_graph <n> <m> <semente> > grafo.txt
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_set>

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Uso: %s <n> <m> <semente>\n", argv[0]);
        return 1;
    }
    long long n = std::atoll(argv[1]), m = std::atoll(argv[2]);
    std::mt19937_64 rng(std::strtoull(argv[3], nullptr, 10));
    std::uniform_int_distribution<long long> pick(1, n);
    std::unordered_set<uint64_t> seen;
    seen.reserve(m * 2);
    std::printf("%lld %lld\n", n, m);
    long long k = 0;
    while (k < m) {
        long long u = pick(rng), v = pick(rng);
        if (u == v) continue;
        if (u > v) std::swap(u, v);
        if (!seen.insert((uint64_t)u * (uint64_t)(n + 1) + (uint64_t)v).second) continue;
        int w = (rng() & 1) ? 1 : -1;
        std::printf("%lld %lld %d\n", u, v, w);
        ++k;
    }
    return 0;
}
