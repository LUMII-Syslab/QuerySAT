import subprocess
from pathlib import Path
from typing import Tuple

import tensorflow as tf

from utils.iterable import elements_to_str


def remove_unused_vars(nvars, clauses):
    used_vars = set()
    n = 0
    max_v = 0
    for clause in clauses:
        for lit in clause:
            if lit == 0:
                continue
            v = abs(lit)
            if v > max_v:
                max_v = v
            if v not in used_vars:
                used_vars.add(v)
                n += 1
    if n == nvars and max_v == n:
        return nvars, clauses  # do not change since all the variables are used
    # otherwise not all variables are used (or the wrong number specified)

    n = 0
    d = {}
    new_clauses = []
    for clause in clauses:
        new_clause = []
        for lit in clause:
            if lit == 0:
                continue
            v = abs(lit)
            if v in d:
                new_v = d[v]
            else:
                n += 1
                new_v = n
                d[v] = new_v
            if lit > 0:
                new_clause.append(new_v)
            else:
                new_clause.append(-new_v)
        new_clauses.append(new_clause)

    return n, new_clauses


def build_dimacs_file(clauses: list, n_vars: int, comments: list = None):
    dimacs = []

    dimacs += comments if comments else []
    dimacs += [f"p cnf {n_vars} {len(clauses)}"]

    clauses = [elements_to_str(c) for c in clauses]
    dimacs += [f"{' '.join(c)} 0" for c in clauses]

    return "\n".join(dimacs)


def run_external_solver(input_dimacs: str, solver_exe: str = "binary/treengeling_linux") -> Tuple[bool, list]:
    """
    :param input_dimacs: Correctly formatted DIMACS file as string
    :param solver_exe: Absolute or relative path to solver executable [supports treengeling, lingeling, plingling]
    :return: returns True if formula is satisfiable and False otherwise, and solutions in form [1,2,-3, ...]
    """
    exe_path = Path(solver_exe).resolve()
    output = subprocess.run([str(exe_path)], input=input_dimacs, stdout=subprocess.PIPE, universal_newlines=True)
    satisfiable = [line for line in output.stdout.split("\n") if line.startswith("s ")]
    if len(satisfiable) > 1:
        raise ValueError("More than one satisifiability line returned!")

    is_sat = satisfiable[0].split()[-1]
    if is_sat != "SATISFIABLE" and is_sat != "UNSATISFIABLE":
        raise ValueError("Unexpected satisfiability value!")

    is_sat = is_sat == "SATISFIABLE"

    if is_sat:
        variables = [line[1:].strip() for line in output.stdout.split("\n") if line.startswith("v ")]
        solution = [int(var) for line in variables for var in line.split()][:-1]
    else:
        solution = []

    return is_sat, solution


def is_batch_sat(predictions: tf.Tensor, adj_matrix: tf.SparseTensor):
    variables = tf.round(tf.sigmoid(predictions))
    literals = tf.concat([variables, 1 - variables], axis=0)
    clauses_sat = tf.sparse.sparse_dense_matmul(adj_matrix, literals)
    clauses_sat = tf.clip_by_value(clauses_sat, 0, 1)

    return tf.reduce_min(clauses_sat)


def walksat(input_dimacs: str,
            solver_exe: str = "binary/walksat_linux",
            ) -> Tuple[bool, list, float]:
    """
    WalkSAT v56 (https://gitlab.com/HenryKautz/Walksat)

    :param input_dimacs: Correctly formatted DIMACS file as string
    :param solver_exe: Absolute or relative path to solver executable
    :return: returns True if formula is satisfiable and False otherwise, solutions in form [1,2,-3, ...] and time
    """
    exe_path = Path(solver_exe).resolve()
    output = subprocess.run([str(exe_path), "-solcnf", "-gsat", "-cutoff", "500K"], input=input_dimacs,
                            stdout=subprocess.PIPE, universal_newlines=True)

    if output.returncode != 0:
        raise RuntimeError("WalkSAT: Unexpected return code ", output.returncode)

    result = output.stdout.strip()
    result = result.split("\n")

    time_elapsed = [x for x in result if x.startswith("total elapsed seconds")][0]
    time_elapsed = time_elapsed.strip().split(" = ")[-1]
    time_elapsed = float(time_elapsed)

    sat = [x for x in result if x == "ASSIGNMENT FOUND"]
    if len(sat) > 1:
        raise RuntimeError("Only one ASSIGNMENT FOUND should be present in output")
    sat = bool(sat)

    if not sat:
        return sat, [], time_elapsed

    solution = [x.strip().split(" ") for x in result if x.startswith("v ")]
    solution = [int(x[-1]) for x in solution]

    return sat, solution, time_elapsed


def is_graph_sat(predictions: tf.Tensor, adj_matrix: tf.SparseTensor, clauses_matrix: tf.SparseTensor):
    """
    :param predictions: Model outputs as logits
    :param adj_matrix: Literals - Clauses adjacency matrix
    :param clauses_matrix: Graph - Clauses adjacency matrix
    :return: vector of elements in {0,1}, where 1 - graph SAT, 0 - graph UNSAT
    """
    variables = tf.round(tf.sigmoid(predictions))
    literals = tf.concat([variables, 1 - variables], axis=0)
    clauses_sat = tf.sparse.sparse_dense_matmul(adj_matrix, literals, adjoint_a=True)
    clauses_sat = tf.clip_by_value(clauses_sat, 0, 1)

    clauses_sat_in_g = tf.sparse.sparse_dense_matmul(clauses_matrix, clauses_sat)
    clauses_total_in_g = tf.expand_dims(tf.sparse.reduce_sum(clauses_matrix, axis=-1), axis=-1)

    return tf.clip_by_value(clauses_sat_in_g + 1 - clauses_total_in_g, 0, 1)
