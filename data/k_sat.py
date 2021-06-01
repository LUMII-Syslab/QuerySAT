import random

import numpy as np
import tensorflow as tf
from pysat.solvers import Cadical

from data.dimac import DIMACDataset
from metrics.sat_metrics import SATAccuracy, StepStatistics


class KSAT(DIMACDataset):
    """ Dataset from NeuroSAT paper, just for variables. Dataset generates k-SAT
    instances with variable count in [min_size, max_size].
    """

    def __init__(self, data_dir, force_data_gen=False,
                 min_vars=3, max_vars=100, test_size=10000, **kwargs) -> None:
        super(KSAT, self).__init__(data_dir, min_vars, max_vars, force_data_gen=force_data_gen, **kwargs)
        self.filter = self.__prepare_filter()
        self.train_size = 100000
        self.test_size = test_size
        self.min_vars = min_vars
        self.max_vars = max_vars

        self.p_k_2 = 0.3
        self.p_geo = 0.4

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
        for _ in range(size):
            n_vars = random.randint(self.min_vars, self.max_vars)

            solver = Cadical()
            iclauses = []

            while True:
                k_base = 1 if random.random() < self.p_k_2 else 2
                k = k_base + np.random.geometric(self.p_geo)
                iclause = self.__generate_k_iclause(n_vars, k)

                solver.add_clause(iclause)
                is_sat = solver.solve()

                if is_sat:
                    iclauses.append(iclause)
                else:
                    break

            iclause_unsat = iclause
            iclause_sat = [-iclause_unsat[0]] + iclause_unsat[1:]

            iclauses.append(iclause_unsat)
            # yield only SAT instance
            # yield n_vars, self.prune(iclauses)

            iclauses[-1] = iclause_sat
            yield n_vars, self.remove_duplicate_clauses(iclauses)

    @staticmethod
    def __generate_k_iclause(n, k):
        vs = np.random.choice(n, size=min(n, k), replace=False)
        return [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]

    @staticmethod
    def remove_duplicate_clauses(clauses):
        return list({tuple(sorted(x)) for x in clauses})

    def create_adj_matrices(self, data):
        var_count = tf.reduce_sum(data["variable_count"])
        clauses_count = tf.reduce_sum(data["clauses_in_formula"])

        shape = [tf.shape(data["adj_indices_neg"])[0], 1]
        offset = tf.ones(shape, dtype=tf.int32) * var_count
        zeros = tf.zeros(shape, dtype=tf.int32)
        offset = tf.concat([offset, zeros], axis=-1)
        offset = tf.cast(offset, tf.int64)
        neg = data["adj_indices_neg"] + offset

        lit_shape = self.create_shape(var_count * 2, clauses_count)
        var_shape = self.create_shape(var_count, clauses_count)
        adj_matrix_lit = tf.concat([data["adj_indices_pos"], neg], axis=0)
        adj_matrix_lit = self.create_adjacency_matrix(adj_matrix_lit, lit_shape)

        adj_matrix_pos = self.create_adjacency_matrix(data["adj_indices_pos"], var_shape)
        adj_matrix_neg = self.create_adjacency_matrix(data["adj_indices_neg"], var_shape)

        graph_count = tf.shape(data["variable_count"])
        graph_id = tf.range(0, graph_count[0])
        variables_mask = tf.repeat(graph_id, data["variable_count"])
        clauses_mask = tf.repeat(graph_id, data["clauses_in_formula"])

        clauses_enum = tf.range(0, var_shape[1], dtype=tf.int32)
        c_g_indices = tf.stack([clauses_mask, clauses_enum], axis=1)
        c_g_indices = tf.cast(c_g_indices, tf.int64)
        clauses_graph_adj = self.create_adjacency_matrix(c_g_indices,
                                                         self.create_shape(tf.cast(graph_count[0], tf.int64),
                                                                           var_shape[1]))

        variables_enum = tf.range(0, var_shape[0], dtype=tf.int32)
        v_g_indices = tf.stack([variables_mask, variables_enum], axis=1)
        v_g_indices = tf.cast(v_g_indices, tf.int64)
        variables_graph_adj = self.create_adjacency_matrix(v_g_indices,
                                                           self.create_shape(tf.cast(graph_count[0], tf.int64),
                                                                             var_shape[0]))

        return {
            "adjacency_matrix_pos": adj_matrix_pos,
            "adjacency_matrix_neg": adj_matrix_neg,
            "adjacency_matrix": adj_matrix_lit,
            "clauses": tf.cast(data["batched_clauses"], tf.int32),
            "variables_in_graph": data["variable_count"],
            "normal_clauses": data["clauses"],
            "clauses_graph_adj": clauses_graph_adj,
            "variables_graph_adj": variables_graph_adj,
            "solutions": data["solutions"]
        }

    @staticmethod
    def create_shape(variables, clauses):
        dense_shape = tf.stack([variables, clauses])
        return tf.cast(dense_shape, tf.int64)

    @staticmethod
    def create_adjacency_matrix(indices, dense_shape):
        return tf.sparse.SparseTensor(indices,
                                      tf.ones(tf.shape(indices)[0], dtype=tf.float32),
                                      dense_shape=dense_shape)

    def prepare_dataset(self, dataset: tf.data.Dataset):
        return dataset.map(self.create_adj_matrices, tf.data.experimental.AUTOTUNE)

    def filter_model_inputs(self, step_data) -> dict:
        return self.filter(step_data)

    def metrics(self, initial=False) -> list:
        return [SATAccuracy(), StepStatistics()]

    @staticmethod
    def __prepare_filter():
        return lambda step_data: {
            "adj_matrix": step_data["adjacency_matrix"],
            "clauses_graph": step_data["clauses_graph_adj"],
            "variables_graph": step_data["variables_graph_adj"],
            "solutions": step_data["solutions"]
        }
