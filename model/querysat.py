import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from layers.normalization import PairNorm
from loss.sat import softplus_loss, softplus_mixed_loss
from model.mlp import MLP
from utils.parameters_log import *
from utils.sat import is_batch_sat


class QuerySAT(Model):

    def __init__(self, optimizer: Optimizer,
                 feature_maps=128, msg_layers=3,
                 vote_layers=3, train_rounds=32, test_rounds=64,
                 query_maps=128, **kwargs):
        super().__init__(**kwargs, name="QuerySAT")
        self.train_rounds = train_rounds
        self.test_rounds = test_rounds
        self.optimizer = optimizer
        self.use_message_passing = False
        self.skip_first_rounds = 0
        self.logit_maps = 8

        update_layers = 3
        output_layers = 2
        query_layers = 2
        clauses_layers = 2

        update_scale = 1.8
        output_scale = 1
        clauses_scale = 1.6
        query_scale = 1.2

        self.variables_norm = PairNorm(subtract_mean=True)
        self.clauses_norm = PairNorm(subtract_mean=True)

        self.update_gate = MLP(update_layers, int(feature_maps * update_scale), feature_maps, name="update_gate")
        self.variables_output = MLP(output_layers, int(feature_maps * output_scale), self.logit_maps,
                                    name="variables_output")
        self.variables_query = MLP(query_layers, int(query_maps * query_scale), query_maps, name="variables_query")
        self.clause_mlp = MLP(clauses_layers, int(feature_maps * clauses_scale), feature_maps + 1 * query_maps,
                              name="clause_update")

        self.lit_mlp = MLP(msg_layers, query_maps * 4, query_maps * 2, name="lit_query")

        self.feature_maps = feature_maps
        self.query_maps = query_maps
        self.vote_layers = vote_layers

    def call(self, adj_matrix, clauses_graph=None, variables_graph=None, training=None, mask=None):
        shape = tf.shape(adj_matrix)
        n_vars = shape[0] // 2
        n_clauses = shape[1]

        variables = tf.ones([n_vars, self.feature_maps])
        clause_state = tf.ones([n_clauses, self.feature_maps])

        rounds = self.train_rounds if training else self.test_rounds

        last_logits, step, unsupervised_loss, supervised_loss, clause_state, variables = self.loop(adj_matrix,
                                                                                                   clause_state,
                                                                                                   clauses_graph,
                                                                                                   rounds,
                                                                                                   training,
                                                                                                   variables,
                                                                                                   variables_graph)

        if training:
            last_clauses = softplus_loss(last_logits, adj_matrix=tf.sparse.transpose(adj_matrix))
            tf.summary.histogram("clauses", last_clauses)
            last_layer_loss = tf.reduce_sum(
                softplus_mixed_loss(last_logits, adj_matrix=tf.sparse.transpose(adj_matrix)))
            tf.summary.histogram("logits", tf.abs(last_logits))
            tf.summary.scalar("last_layer_loss", last_layer_loss)

            tf.summary.scalar("steps_taken", step)
            tf.summary.scalar("supervised_loss", supervised_loss)

        return last_logits, unsupervised_loss + supervised_loss, step

    def loop(self, adj_matrix, clause_state, clauses_graph, rounds, training, variables, variables_graph):
        step_losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        cl_adj_matrix = tf.sparse.transpose(adj_matrix)
        shape = tf.shape(adj_matrix)
        n_clauses = shape[1]
        n_vars = shape[0] // 2
        last_logits = tf.zeros([n_vars, self.logit_maps])
        lit_degree = tf.reshape(tf.sparse.reduce_sum(adj_matrix, axis=1), [n_vars * 2, 1])
        degree_weight = tf.math.rsqrt(tf.maximum(lit_degree, 1))
        var_degree_weight = 4 * tf.math.rsqrt(tf.maximum(lit_degree[:n_vars, :] + lit_degree[n_vars:, :], 1))
        rev_lit_degree = tf.reshape(tf.sparse.reduce_sum(cl_adj_matrix, axis=1), [n_clauses, 1])
        rev_degree_weight = tf.math.rsqrt(tf.maximum(rev_lit_degree, 1))

        supervised_loss = 0.
        best_logit_map = tf.zeros([n_vars], dtype=tf.int32)

        variables_graph_norm = variables_graph / tf.sparse.reduce_sum(variables_graph, axis=-1, keepdims=True)
        clauses_graph_norm = clauses_graph / tf.sparse.reduce_sum(clauses_graph, axis=-1, keepdims=True)

        for step in tf.range(rounds):
            # make a query for solution, get its value and gradient
            with tf.GradientTape() as grad_tape:
                # add some randomness to avoid zero collapse in normalization
                v1 = tf.concat([variables, tf.random.normal([n_vars, 4])], axis=-1)
                query = self.variables_query(v1)
                clauses_loss = softplus_loss(query, cl_adj_matrix)
                step_loss = tf.reduce_sum(clauses_loss)

            variables_grad = tf.convert_to_tensor(grad_tape.gradient(step_loss, query))
            variables_grad = variables_grad * var_degree_weight
            # calculate new clause state
            clauses_loss *= 4

            if self.use_message_passing:
                var_msg = self.lit_mlp(variables, training=training)
                lit1, lit2 = tf.split(var_msg, 2, axis=1)
                literals = tf.concat([lit1, lit2], axis=0)
                clause_messages = tf.sparse.sparse_dense_matmul(cl_adj_matrix, literals)
                clause_messages *= rev_degree_weight
                clause_unit = tf.concat([clause_state, clause_messages, clauses_loss], axis=-1)
            else:
                clause_unit = tf.concat([clause_state, clauses_loss], axis=-1)
            clause_data = self.clause_mlp(clause_unit, training=training)

            variables_loss_all = clause_data[:, 0:self.query_maps]
            new_clause_value = clause_data[:, self.query_maps:]
            new_clause_value = self.clauses_norm(new_clause_value, clauses_graph_norm, training=training)
            clause_state = new_clause_value + 0.1 * clause_state

            # Aggregate loss over edges
            variables_loss = tf.sparse.sparse_dense_matmul(adj_matrix, variables_loss_all)
            variables_loss *= degree_weight
            variables_loss_pos, variables_loss_neg = tf.split(variables_loss, 2, axis=0)

            # calculate new variable state
            unit = tf.concat([variables_grad, variables, variables_loss_pos, variables_loss_neg], axis=-1)
            new_variables = self.update_gate(unit)
            new_variables = self.variables_norm(new_variables, variables_graph_norm, training=training)
            variables = new_variables + 0.1 * variables

            # calculate logits and loss
            logits = self.variables_output(variables)

            per_clause_loss = softplus_mixed_loss(logits, cl_adj_matrix)
            per_graph_loss = tf.sparse.sparse_dense_matmul(clauses_graph, per_clause_loss)
            per_graph_loss = tf.sqrt(per_graph_loss + 1e-6) - tf.sqrt(1e-6)
            costs = tf.square(tf.range(1, self.logit_maps + 1, dtype=tf.float32))
            per_graph_loss_avg = tf.reduce_sum(tf.sort(per_graph_loss, axis=-1, direction='DESCENDING') * costs) / tf.reduce_sum(costs)
            logit_loss = tf.reduce_sum(per_graph_loss_avg)

            best_logit_map = tf.cast(tf.argmin(per_graph_loss, axis=-1), tf.float32)
            best_logit_map = tf.expand_dims(best_logit_map, axis=-1)
            best_logit_map = tf.sparse.sparse_dense_matmul(variables_graph, best_logit_map, adjoint_a=True)
            best_logit_map = tf.cast(tf.squeeze(best_logit_map, axis=-1), tf.int32)

            step_losses = step_losses.write(step, logit_loss)
            last_logits = logits

            out_logits = tf.gather(logits, best_logit_map, batch_dims=1)
            out_logits = tf.expand_dims(out_logits, axis=-1)
            is_sat = is_batch_sat(out_logits, cl_adj_matrix)
            if is_sat == 1:
                break

            # due to the loss at each level, gradients accumulate on the backward pass and may become very large for the first layers
            # reduce the gradient magnitude to remedy this
            variables = tf.stop_gradient(variables) * 0.2 + variables * 0.8
            clause_state = tf.stop_gradient(clause_state) * 0.2 + clause_state * 0.8

        if training:
            logits_round = tf.round(tf.sigmoid(last_logits))
            logits_different = tf.reduce_mean(tf.abs(logits_round[:, 0:1] - logits_round[:, 1:]))
            tf.summary.scalar("logits_different", logits_different)

            logits_mean = tf.reduce_mean(last_logits, axis=-1, keepdims=True)
            logits_dif = last_logits - logits_mean
            tf.summary.histogram("logits_dif", logits_dif)

        unsupervised_loss = tf.reduce_sum(step_losses.stack()) / tf.cast(rounds, tf.float32)
        out_logits = tf.gather(last_logits, best_logit_map, batch_dims=1)
        out_logits = tf.expand_dims(out_logits, axis=-1)
        return out_logits, step, unsupervised_loss, supervised_loss, clause_state, variables

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)
                                  ])
    def train_step(self, adj_matrix, clauses_graph, variables_graph, solutions):
        with tf.GradientTape() as tape:
            _, loss, step = self.call(adj_matrix, clauses_graph, variables_graph, training=True)
            train_vars = self.trainable_variables
            gradients = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(gradients, train_vars))

        return {
            "steps_taken": step,
            "loss": loss,
            "gradients": gradients
        }

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)
                                  ])
    def predict_step(self, adj_matrix, clauses_graph, variables_graph, solutions):
        predictions, loss, step = self.call(adj_matrix, clauses_graph, variables_graph, training=False)

        return {
            "steps_taken": step,
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1)
        }

    def get_config(self):
        return {HP_MODEL: self.__class__.__name__,
                HP_FEATURE_MAPS: self.feature_maps,
                HP_QUERY_MAPS: self.query_maps,
                HP_TRAIN_ROUNDS: self.train_rounds,
                HP_TEST_ROUNDS: self.test_rounds,
                HP_MLP_LAYERS: self.vote_layers,
                }
