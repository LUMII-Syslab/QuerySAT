import math
import random

import networkx as nx
import numpy as np
from cnfgen import RandomKCNF, CliqueFormula, GraphColoringFormula
from pysat.solvers import Glucose4

from data.k_sat import KSAT
from utils.sat import run_external_solver, build_dimacs_file


class SAT_3(KSAT):
    """
    Dataset with random 3-SAT instances at the satisfiability threshold from CNFGen library.
    """

    def __init__(self, data_dir, min_vars=5, max_vars=100, force_data_gen=False, **kwargs) -> None:
        super(SAT_3, self).__init__(data_dir, min_vars=min_vars, max_vars=max_vars, force_data_gen=force_data_gen,
                                    **kwargs)
        self.train_size = 100000
        self.test_size = 10000
        self.min_vars = min_vars
        self.max_vars = max_vars

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
        formula_count = 0
        while formula_count <= size:
            n_vars = random.randint(self.min_vars, self.max_vars)
            n_clauses = 4.258 * n_vars + 58.26 * np.power(n_vars, -2 / 3.)
            n_clauses = int(n_clauses)

            F = RandomKCNF(3, n_vars, n_clauses)
            clauses = list(F.clauses())
            iclauses = [F._compress_clause(x) for x in clauses]

            if n_vars > 200:
                dimacs = build_dimacs_file(iclauses, n_vars)
                is_sat, solution = run_external_solver(dimacs)
            else:
                with Glucose4(bootstrap_with=iclauses) as solver:
                    is_sat = solver.solve()
                    solution = solver.get_model()

            if is_sat:
                formula_count += 1
                yield n_vars, iclauses, solution


class Clique(KSAT):
    """
    Dataset with random sat instances from triangle detection in graphs.
    Using Erdos-Renyi graphs with edge probability such that it is triangle-free with probability 0.5
    """

    def __init__(self, data_dir, min_vertices=4, max_vertices=40, force_data_gen=False, **kwargs) -> None:
        super(Clique, self).__init__(data_dir, min_vars=min_vertices, max_vars=max_vertices,
                                     force_data_gen=force_data_gen, **kwargs)
        self.train_size = 100000
        self.test_size = 10000
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self.clique_size = 3

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
        total_generated = 0
        while total_generated <= size:
            n_vertices = random.randint(self.min_vertices, self.max_vertices)
            # generate a random graph with such sparsity that a triangle is expected with probability 0.5.
            # eps = 0.2
            # p = 0.7 * ((1 + eps) * np.log(n_vertices)) / n_vertices # less exact formula
            p = 3 ** (1 / 3) / (n_vertices * (2 - 3 * n_vertices + n_vertices ** 2)) ** (1 / 3)
            G = nx.generators.erdos_renyi_graph(n_vertices, p=p)
            F = CliqueFormula(G, self.clique_size)

            n_vars = len(list(F.variables()))
            clauses = list(F.clauses())
            iclauses = [F._compress_clause(x) for x in clauses]
            with Glucose4(bootstrap_with=iclauses) as solver:
                is_sat = solver.solve()
                solution = solver.get_model()

            if is_sat:
                total_generated += 1
                yield n_vars, iclauses, solution


class KColor(KSAT):
    """
    Generates the clauses for colorability formula
    The formula encodes the fact that the graph :math:`G` has a coloring
    with color set ``colors``. This means that it is possible to
    assign one among the elements in ``colors``to that each vertex of
    the graph such that no two adjacent vertices get the same color.
    """

    def __init__(self, data_dir, min_vertices=5, max_vertices=40, force_data_gen=False, **kwargs) -> None:
        super(KColor, self).__init__(data_dir, min_vars=min_vertices, max_vars=max_vertices, force_data_gen=force_data_gen, **kwargs)
        self.train_size = 100000
        self.test_size = 10000
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
        total_generated = 0
        while total_generated <= size:
            n_vertices = random.randint(self.min_vertices, self.max_vertices)
            p = ((math.log(n_vertices) * (1 + 0.2)) / n_vertices) + 0.05  # Generate mostly connected sparse graphs

            G = nx.generators.erdos_renyi_graph(n_vertices, p=p)
            if not nx.is_connected(G):
                continue

            n_colors = random.randint(3, 5)  # Same as NeuroSAT paper

            F = GraphColoringFormula(G, n_colors)
            n_vars = len(list(F.variables()))
            clauses = list(F.clauses())
            iclauses = [F._compress_clause(x) for x in clauses]

            with Glucose4(bootstrap_with=iclauses) as solver:
                is_sat = solver.solve()
                solution = solver.get_model()

            if is_sat:
                total_generated += 1
                yield n_vars, iclauses, solution
