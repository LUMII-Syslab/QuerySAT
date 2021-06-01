import tensorflow as tf


def softplus_mixed_loss(variable_predictions: tf.Tensor, adj_matrix: tf.SparseTensor, eps=1e-8):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param adj_matrix: variables-clauses graph represented as adjacency matrix
    :param eps: small value to avoid taking log from 0
    :return: returns per clause loss which is log_loss multiplied with linear loss
    """
    clauses_val = softplus_loss(variable_predictions, adj_matrix)
    log_clauses = -(tf.math.log(1 - clauses_val + eps) - tf.math.log(1 + eps))
    return clauses_val * log_clauses


def softplus_loss(variable_predictions: tf.Tensor, adj_matrix: tf.SparseTensor):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param adj_matrix: Sparse tensor with dense shape literals x clauses
    :return: returns per clause loss in range [0..1] - 0 if clause is satisfied, 1 if not satisfied
    """

    literals = tf.concat([variable_predictions, -variable_predictions], axis=0)
    literals = tf.nn.softplus(literals)
    clauses_val = tf.sparse.sparse_dense_matmul(adj_matrix, literals)
    clauses_val = tf.exp(-clauses_val)

    return clauses_val


def linear_loss(variable_predictions: tf.Tensor, adj_matrix: tf.SparseTensor):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param adj_matrix: Sparse tensor with dense shape literals x clauses
    :return: returns per clause loss in range [0..] - 0 if clause is satisfied, >0 if not satisfied
    """
    variable_predictions = tf.sigmoid(variable_predictions)
    literals = tf.concat([variable_predictions, 1 - variable_predictions], axis=0)
    clauses_val = tf.sparse.sparse_dense_matmul(adj_matrix, literals)
    clauses_val = tf.nn.relu(1 - clauses_val)

    return clauses_val
