import json

import numpy as np
import scipy as spy
import tensorflow as tf


def load_model(model_config, adj_mat, input_dim, output_dim):
    with open(model_config, 'r') as jfile:
        config_dict = json.load(jfile)
    return dict(
        GCN=GCN
    )[config_dict['name']](adj_mat, input_dim, output_dim, config_dict['hidden_dims'], activation=config_dict['activation'])


def get_activation(activation_name):
    return dict(Relu=tf.nn.relu)[activation_name]


class GraphDense(tf.keras.layers.Layer):
    
    def __init__(self, agg_mat, input_dim, output_dim):
        super(GraphDense, self).__init__()
        self.agg_mat = agg_mat
        self.weight = tf.Variable(tf.keras.initializers.HeNormal()(shape=[input_dim, output_dim]))
    
    def call(self, inputs):
        return tf.matmul(self.agg_mat, tf.matmul(inputs, self.weight))
    
    def get_weight(self):
        return self.weight


class GCN(tf.keras.Model):
    
    def __init__(self, adj_mat, input_dim, output_dim, hidden_layers,
                 activation):
        super(GCN, self).__init__()
        adj_hat = np.identity(adj_mat.shape[0]) + adj_mat
        degree_mat = np.diag(1. / np.sqrt(np.sum(adj_hat, axis=1)))
        self.agg_mat = tf.cast(degree_mat @ adj_hat @ degree_mat,dtype=tf.float32)
        
        self.activation = get_activation(activation)
        self.gcn_layers = []
        for in_dim, out_dim in zip([input_dim] + hidden_layers,
                                   hidden_layers + [output_dim]):
            self.gcn_layers.append(GraphDense(self.agg_mat, in_dim, out_dim))
    
    def __call__(self, in_x):
        for layer in self.gcn_layers:
            in_x = layer(in_x)
            in_x = self.activation(in_x)
        return tf.nn.softmax(in_x, axis=1)
    
    def get_trainable_parameters(self):
        return [layer.get_weight() for layer in self.gcn_layers]
        
        
