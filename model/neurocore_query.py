import math

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model

from loss.sat import softplus_mixed_loss, softplus_loss
from model.mlp import MLP
from utils.parameters_log import *
from utils.sat import is_batch_sat


class NeuroCoreQuery(Model):
    def __init__(self, optimizer, test_rounds=64, train_rounds=32, **kwargs):
        super(NeuroCoreQuery, self).__init__(**kwargs)
        self.optimizer = optimizer

        self.feature_maps = 128
        self.train_rounds = train_rounds
        self.test_rounds = test_rounds
        self.norm_axis = 0
        self.norm_eps = 1e-6
        self.n_update_layers = 2
        self.n_score_layers = 2

        self.L_updates = MLP(self.n_update_layers + 1, 2 * self.feature_maps + self.feature_maps, self.feature_maps,
                             activation=tf.nn.relu6, name="L_u")
        self.C_updates = MLP(self.n_update_layers + 1, self.feature_maps + self.feature_maps, self.feature_maps,
                             activation=tf.nn.relu6, name="C_u")

        init = tf.constant_initializer(1.0 / math.sqrt(self.feature_maps))
        self.L_init_scale = self.add_weight(name="L_init_scale", shape=[], initializer=init)
        self.C_init_scale = self.add_weight(name="C_init_scale", shape=[], initializer=init)

        self.LC_scale = self.add_weight(name="LC_scale", shape=[], initializer=tf.constant_initializer(0.1))
        self.CL_scale = self.add_weight(name="CL_scale", shape=[], initializer=tf.constant_initializer(0.1))

        self.variables_query = MLP(self.n_update_layers + 1, self.feature_maps, self.feature_maps,
                                   name="variables_query", do_layer_norm=False)
        self.V_score = MLP(self.n_score_layers + 1, 2 * self.feature_maps, 1, activation=tf.nn.relu6, name="V_score")

        self.self_supervised = False

    def call(self, adj_matrix, clauses_graph, variables_graph, training=None, mask=None):
        shape = tf.shape(adj_matrix)  # inputs is sparse factor matrix
        n_lits = shape[0]
        n_vars = n_lits // 2
        n_clauses = shape[1]

        L = tf.ones(shape=[n_vars, self.feature_maps], dtype=tf.float32) * self.L_init_scale
        C = tf.ones(shape=[n_clauses, self.feature_maps], dtype=tf.float32) * self.C_init_scale

        loss = 0.
        supervised_loss = 0.

        rounds = self.train_rounds if training else self.test_rounds
        logits = tf.zeros([n_vars, 1])
        cl_adj_matrix = tf.sparse.transpose(adj_matrix)
        query = tf.zeros([n_vars, 128])

        step_logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        step_queries = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for steps in tf.range(rounds):
            lit1, lit2 = tf.split(L, 2, axis=1)
            literals = tf.concat([lit1, lit2], axis=0)
            LC_msgs = tf.sparse.sparse_dense_matmul(cl_adj_matrix, literals) * self.LC_scale

            with tf.GradientTape() as grad_tape:
                query = self.variables_query(L)
                step_queries = step_queries.write(steps, query)
                clauses_loss = softplus_loss(query, cl_adj_matrix)
                step_loss = tf.reduce_sum(clauses_loss)

            variables_grad = tf.convert_to_tensor(grad_tape.gradient(step_loss, query)) * self.G_scale

            C = self.C_updates(tf.concat([C, clauses_loss, LC_msgs], axis=-1))
            C = tf.debugging.check_numerics(C, message="C after update")
            C = normalize(C, axis=self.norm_axis, eps=self.norm_eps)
            C = tf.debugging.check_numerics(C, message="C after norm")

            CL_msgs = tf.sparse.sparse_dense_matmul(adj_matrix, C) * self.CL_scale
            CL_msgs1, CL_msgs2 = tf.split(CL_msgs, 2, axis=0)
            L = self.L_updates(tf.concat([L, CL_msgs1, CL_msgs2, variables_grad], axis=-1))
            L = tf.debugging.check_numerics(L, message="L after update")
            L = normalize(L, axis=self.norm_axis, eps=self.norm_eps)
            L = tf.debugging.check_numerics(L, message="L after norm")

            logits = self.V_score(L)  # (n_vars, 1)
            step_logits = step_logits.write(steps, logits)

            is_sat = is_batch_sat(logits, cl_adj_matrix)
            if is_sat == 1:
                break

            per_clause_loss = softplus_mixed_loss(logits, cl_adj_matrix)
            per_graph_loss = tf.sparse.sparse_dense_matmul(clauses_graph, per_clause_loss)
            loss += tf.reduce_sum(tf.sqrt(per_graph_loss + 1e-6))

            L = tf.stop_gradient(L) * 0.2 + L * 0.8
            C = tf.stop_gradient(C) * 0.2 + C * 0.8

            if training and steps == 0:
                self.query_stats(adj_matrix, cl_adj_matrix, logits, n_clauses, n_vars, query, "0")

            if training and steps == 16:
                self.query_stats(adj_matrix, cl_adj_matrix, logits, n_clauses, n_vars, query, "16")

        if training:
            self.query_stats(adj_matrix, cl_adj_matrix, logits, n_clauses, n_vars, query)

        if training:
            self.log_differences(n_vars, step_logits, steps, "logits_diff")

        if training:
            self.log_differences(n_vars, step_queries, steps, "queries_diff")

        return logits, loss / tf.cast(rounds, tf.float32) + supervised_loss, steps

    def log_differences(self, n_vars, tensor_array, steps, name_scope):
        with tf.name_scope(name_scope):
            for i in range(1, self.train_rounds, 1):
                if i < steps:
                    new = tensor_array.read(i)
                    old = tensor_array.read(i - 1)
                    new = tf.round(tf.sigmoid(new))
                    old = tf.round(tf.sigmoid(old))
                    matching = tf.cast(tf.equal(new, old), tf.float32)
                    matching = tf.reduce_sum(matching, axis=0) / tf.cast(n_vars, tf.float32)
                    same = tf.reduce_mean(matching)

                    tf.summary.scalar(f"step_{i}_vs_{i - 1}", same)

    def query_stats(self, adj_matrix, cl_adj_matrix, logits, n_clauses, n_vars, query, step: str = "last"):
        current_labels = tf.round(tf.sigmoid(logits))
        round_query = tf.round(tf.sigmoid(query))
        query_logits_match = tf.cast(tf.equal(current_labels, round_query), tf.float32)

        lit_in_n_clauses = tf.sparse.reduce_sum(adj_matrix, axis=-1)
        lit1, lit2 = tf.split(lit_in_n_clauses, 2, axis=0)
        vars_count = lit1 + lit2
        in_n_clauses_matched = tf.reduce_mean(tf.expand_dims(vars_count, axis=-1) * query_logits_match)
        in_n_clauses_not_matched = tf.reduce_mean(tf.expand_dims(vars_count, axis=-1) * (1 - query_logits_match))

        query_matching_values = tf.sigmoid(query) * (1 - query_logits_match)
        query_not_matching_values = tf.sigmoid(query) * (1 - query_logits_match)
        not_matching_mean = tf.reduce_mean(query_not_matching_values)
        not_matching_median = tfp.stats.percentile(query_not_matching_values, 50.0, interpolation='midpoint')

        matching_mean = tf.reduce_mean(query_matching_values)
        matching_median = tfp.stats.percentile(query_matching_values, 50.0, interpolation='midpoint')

        query_logits_match = tf.reduce_sum(query_logits_match, axis=0) / tf.cast(n_vars, tf.float32)
        query_logits_match = tf.reduce_mean(query_logits_match)

        lit = tf.concat([round_query, 1 - round_query], axis=0)
        sat_clauses = tf.sparse.sparse_dense_matmul(cl_adj_matrix, lit)
        sat_clauses = tf.clip_by_value(sat_clauses, 0, 1)
        sat_clauses = tf.reduce_sum(sat_clauses, axis=0) / tf.cast(n_clauses, tf.float32)
        sat_clauses = tf.reduce_mean(sat_clauses)

        with tf.name_scope(f"query_stats_{step}"):
            tf.summary.scalar("query_logits_match", query_logits_match)
            tf.summary.scalar("sat_clauses", sat_clauses)

            tf.summary.scalar("vars_in_clauses_matched", in_n_clauses_matched)
            tf.summary.scalar("vars_in_clauses_not_matched", in_n_clauses_not_matched)

            tf.summary.scalar("not_matching_mean", not_matching_mean)
            tf.summary.scalar("not_matching_median", not_matching_median)

            tf.summary.scalar("matching_mean", matching_mean)
            tf.summary.scalar("matching_median", matching_median)

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)
                                  ])
    def train_step(self, adj_matrix, clauses_graph, variables_graph, solutions):
        with tf.GradientTape() as tape:
            logits, loss, steps = self.call(adj_matrix, clauses_graph, variables_graph, training=True)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": loss,
            "gradients": gradients,
            "steps_taken": steps
        }

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)
                                  ])
    def predict_step(self, adj_matrix, clauses_graph, variables_graph, solutions):
        predictions, loss, steps = self.call(adj_matrix, clauses_graph, variables_graph, training=False)

        return {
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1),
            "steps_taken": steps
        }

    def get_config(self):
        return {HP_MODEL: self.__class__.__name__,
                HP_FEATURE_MAPS: self.feature_maps,
                HP_TRAIN_ROUNDS: self.train_rounds,
                HP_TEST_ROUNDS: self.test_rounds,
                HP_MLP_LAYERS: self.n_update_layers
                }


def normalize(x, axis, eps):
    mean, variance = tf.nn.moments(x, axes=[axis], keepdims=True)
    return tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=eps)
