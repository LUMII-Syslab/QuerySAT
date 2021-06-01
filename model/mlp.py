import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Dense


class MLP(Model):
    def __init__(self, layer_count, hidden_nmap,
                 out_nmap, activation=tf.nn.leaky_relu,
                 out_activation=None, **kwargs):
        super().__init__(**kwargs)
        self.dense_layers = [Dense(hidden_nmap, activation=activation) for _ in range(layer_count - 1)]
        self.dense_layers.append(Dense(out_nmap, activation=out_activation, bias_initializer='zeros'))

    @tf.function
    def call(self, inputs, training=None):
        current = inputs
        for layer in self.dense_layers:
            current = layer(current, training=training)
        return current


class NeuroCoreMLP(Model):
    def __init__(self, layer_count, hidden_nmap, out_nmap, activation=tf.nn.relu6, **kwargs):
        super().__init__(**kwargs)
        self.dense_layers = [Dense(hidden_nmap, kernel_initializer='glorot_normal', activation=activation) for _ in range(layer_count - 1)]
        self.dense_layers.append(Dense(out_nmap, kernel_initializer='glorot_normal'))

    @tf.function
    def call(self, inputs, training=None):
        current = inputs
        for layer in self.dense_layers:
            current = layer(current, training=training)

        return current
