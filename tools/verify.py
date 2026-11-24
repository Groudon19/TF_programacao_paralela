#!/usr/bin/env python3
"""Verificador independente: confere o formato da saida e calcula o corte.

Uso: python3 verify.py <grafo> <saida> [--brute]
  --brute  tambem calcula o corte maximo por forca bruta (so para n <= 24)
"""
import sys


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = f.read().split()
    n, m = int(data[0]), int(data[1])
    edges = []
    for k in range(m):
        u, v, w = int(data[2 + 3 * k]), int(data[3 + 3 * k]), int(data[4 + 3 * k])
        edges.append((u - 1, v - 1, w))

    with open(sys.argv[2]) as f:
        sol = f.read().strip()
    if len(sol) != n or any(c not in "01" for c in sol):
        print(f"SAIDA INVALIDA: tamanho {len(sol)} (esperado {n}) ou caractere invalido -> vale zero")
        sys.exit(2)

    cut = sum(w for u, v, w in edges if sol[u] != sol[v])
    print(f"OK  n={n} m={m}  corte={cut}")

    if "--brute" in sys.argv:
        if n > 24:
            print("forca bruta so para n <= 24")
            return
        best = max(
            sum(w for u, v, w in edges if ((x >> u) ^ (x >> v)) & 1)
            for x in range(1 << (n - 1))
        )
        print(f"forca bruta: corte maximo = {best}  ({'OTIMO' if best == cut else 'NAO otimo'})")


if __name__ == "__main__":
    main()
