"""Eksperymenty do kolokwium AiN (Zawody1.pdf).

NIE MODYFIKUJE EvolutionaryAlgorithm.py; jedynie importuje GeneticAlgorithm.
Domyslnie: reprezentacja real, n=5,15,30, 100 powtorzen i 10000*n ocen.
Zapisuje kazdy skonczony przebieg osobno, aby mozna bylo wznowic prace.

Przyklady:
    python Eksport_Zawody_AiN.py --dimensions 5 --representation real
    python Eksport_Zawody_AiN.py --dimensions 5 15 30 --representation gray
    python Eksport_Zawody_AiN.py --dimensions 5 --runs 1 --evals 120 --functions rosenbrock
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from EvolutionaryAlgorithm import GeneticAlgorithm

POPULATION_SIZE = 100
BITS_PER_DIMENSION = 16
DEFAULT_RUNS = 100
DIMENSIONS = (5, 15, 30)


def rosenbrock(x):
    """Generalized Rosenbrock, A.1.3, F*=0, x*=(1,...,1)."""
    return sum(100.0 * (x[j + 1] - x[j] ** 2) ** 2 + (x[j] - 1.0) ** 2
               for j in range(len(x) - 1))


def salomon(x):
    """Salomon, A.2.4, F*=0, x*=(0,...,0)."""
    radius = math.sqrt(sum(v * v for v in x))
    return 1.0 - math.cos(2 * math.pi * radius) + 0.1 * radius


def whitley(x):
    """Whitley, A.2.5, full matrix, F*=0, x*=(1,...,1)."""
    total = 0.0
    for xk in x:
        for xj in x:
            y = 100.0 * (xk - xj * xj) ** 2 + (1.0 - xj) ** 2
            total += y * y / 4000.0 - math.cos(y) + 1.0
    return total


# Ograniczenia dokladnie z Zawody1.pdf (nie z szerszej tabeli w ksiazce).
FUNCTIONS = {
    "rosenbrock": (rosenbrock, -30.0, 30.0),
    "salomon": (salomon, -100.0, 100.0),
    "whitley": (whitley, -10.24, 10.24),
}


class EvaluationLimit(Exception):
    pass


def run_one(function, lower, upper, representation, dimensions, max_evals):
    ga = GeneticAlgorithm(dimensions=dimensions, bits_per_dimension=BITS_PER_DIMENSION,
                          lower_bound=lower, upper_bound=upper,
                          start_pop_size=POPULATION_SIZE, representation=representation)
    history = np.empty(max_evals, dtype=np.float64)
    count = 0
    best = float('inf')
    # Ograniczenie pamieci cache. Ponowne sortowania tych samych osobnikow
    # nie sa nowymi wywolaniami F. Ostatnie 2000 obiektow obejmuje populacje
    # i rodzicow biezacej generacji; nie dotykamy kodu samego algorytmu.
    cache = OrderedDict()
    cache_limit = 2000

    def tracked_objective(individual):
        nonlocal count, best
        key = id(individual)
        cached = cache.get(key)
        if cached is not None and cached[0] is individual:
            cache.move_to_end(key)
            return cached[1]
        if count >= max_evals:
            raise EvaluationLimit()
        values = ga.decode_solution(individual['x']) if representation == 'gray' else individual['x']
        value = float(function(values))
        if not math.isfinite(value):
            raise ValueError('Funkcja celu zwrocila wartosc niefinitywna')
        cache[key] = (individual, value)
        cache.move_to_end(key)
        if len(cache) > cache_limit:
            cache.popitem(last=False)
        best = min(best, value)
        history[count] = best
        count += 1
        if count == max_evals:
            raise EvaluationLimit()
        return value

    ga.objective_value = tracked_objective
    try:
        ga.run(n_iter=max_evals, verbose=False)
    except EvaluationLimit:
        pass
    if count != max_evals:
        raise RuntimeError(f'Zapisano {count} ocen zamiast {max_evals}')
    return history


def run_path(root, function_name, dimensions, representation, run_num):
    return root / f'n{dimensions}' / function_name / representation / f'run_{run_num:03d}.npz'


def main():
    p = argparse.ArgumentParser(description='Eksperymenty AiN, Zawody1.pdf')
    p.add_argument('--dimensions', nargs='+', type=int, default=list(DIMENSIONS), choices=DIMENSIONS)
    p.add_argument('--functions', nargs='+', choices=list(FUNCTIONS), default=list(FUNCTIONS))
    p.add_argument('--representation', choices=['real', 'gray'], default='real')
    p.add_argument('--runs', type=int, default=DEFAULT_RUNS)
    p.add_argument('--evals', type=int, default=None,
                   help='TYLKO do testow; domyslnie 10000*n')
    p.add_argument('--output', type=Path, default=Path('wyniki_zawody'))
    p.add_argument('--seed', type=int, default=20261126)
    p.add_argument('--force', action='store_true', help='przelicz ukonczone przebiegi')
    args = p.parse_args()
    if args.runs <= 0 or (args.evals is not None and args.evals < 1):
        p.error('runs i evals musza byc dodatnie')
    args.output.mkdir(parents=True, exist_ok=True)

    for n in args.dimensions:
        max_evals = args.evals if args.evals is not None else n * 10000
        for fn in args.functions:
            function, lower, upper = FUNCTIONS[fn]
            print(f'\nSTART {fn}, n={n}, rep={args.representation}, {args.runs} przebiegow x {max_evals} ocen', flush=True)
            for run_num in range(1, args.runs + 1):
                target = run_path(args.output, fn, n, args.representation, run_num)
                if target.exists() and not args.force:
                    # Tylko prawidlowo zapisane wyniki z ta sama konfiguracja mozna pomijac.
                    with np.load(target, allow_pickle=False) as saved:
                        if int(saved['evals']) == max_evals and int(saved['dimensions']) == n and len(saved['best']) == max_evals:
                            print(f'  {run_num}/{args.runs} juz istnieje (pomijam)', flush=True)
                            continue
                    raise RuntimeError(f'Istniejacy plik nie pasuje do konfiguracji: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                seed_key = f'{args.seed}:{fn}:{n}:{args.representation}:{run_num}'.encode()
                seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:8], 'big') % (2 ** 63)
                random.seed(seed)
                t0 = time.perf_counter()
                best = run_one(function, lower, upper, args.representation, n, max_evals)
                temp = target.with_name(target.name + '.tmp')
                with temp.open('wb') as stream:
                    np.savez_compressed(stream, best=best, evals=max_evals, dimensions=n,
                                        seed=seed, final_best=best[-1])
                temp.replace(target)
                print(f'  {run_num}/{args.runs} wynik={best[-1]:.8g}, czas={time.perf_counter()-t0:.1f}s', flush=True)
    print('\nGOTOWE. Dane kazdego przebiegu zapisano w osobnym pliku NPZ.', flush=True)


if __name__ == '__main__':
    main()
