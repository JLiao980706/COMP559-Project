import json
from xml.etree.ElementInclude import include

import numpy as np
import scipy as spy
import tensorflow as tf


def load_model(model_config, adj_mat, input_dim, output_dim):
    with open(model_config, 'r') as jfile:
        config_dict = json.load(jfile)
    return dict(
        GCN=GCN
    )[config_dict['name']](adj_mat, input_dim, output_dim,
                           config_dict['hidden_dims'],
                           config_dict['activation'], config_dict['dropout'],
                           config_dict['drop_input'], config_dict['drop_name'],
                           config_dict['initializer'], config_dict['bias'])


def get_activation(activation_name):
    return dict(
        Relu=tf.nn.relu,
        Id=lambda x: x
    )[activation_name]
    

def get_dropout_layer(drop_name):
    return dict(
        sparse=SparseDropout,
        vanilla=tf.keras.layers.Dropout
    )[drop_name]


def initialize_variables(shape, method):
    if method == 'glorot':
        init_range = np.sqrt(6.0 / (shape[0]+shape[1]))
        initial = tf.random.uniform(shape, minval=-init_range,
                                    maxval=init_range, dtype=tf.float32)
        return tf.Variable(initial)
    elif method == 'builtin_glorot':
        return tf.Variable(tf.keras.initializers.GlorotUniform()(shape=shape))
    elif method == 'glorot_normal':
        return tf.Variable(tf.keras.initializers.GlorotNormal()(shape=shape))
    elif method == 'he_normal':
        return tf.Variable(tf.keras.initializers.HeNormal()(shape=shape))
    elif method == 'he_uniform':
        return tf.Variable(tf.keras.initializers.HeUniform()(shape=shape))
    else:
        raise Exception('Initialization Method not Found')


class GraphDense(tf.keras.layers.Layer):
    
    def __init__(self, agg_mat, input_dim, output_dim, initializer, bias,
                 activation):
        super(GraphDense, self).__init__()
        self.agg_mat = agg_mat
        self.weight = initialize_variables([input_dim, output_dim], initializer)
        self.bias = np.zeros((output_dim,), dtype=np.float32)
        self.activ_func = get_activation(activation)
        if bias:
            self.bias = tf.Variable(self.bias)
    
    def call(self, inputs):
        out_x = tf.matmul(self.agg_mat, tf.matmul(inputs, self.weight)) +\
            self.bias
        return self.activ_func(out_x)
    
    def get_weights(self):
        return [self.weight]
    

class SparseDropout(tf.keras.layers.Layer):
    
    def __init__(self, drop_prob):
        super(SparseDropout, self).__init__()
        self.drop_prob = drop_prob
    
    def call(self, inputs, training):
        if not training:
            return inputs
        nonzero_idx = tf.sparse.from_dense(inputs).indices.numpy()
        mask_val = np.random.choice([0, 1], size=(nonzero_idx.shape[0],),
                                    p=[self.drop_prob, 1-self.drop_prob])
        mask = spy.sparse.coo_array((mask_val, (nonzero_idx[:,0],
                                                nonzero_idx[:,1])),
                                    shape=inputs.shape).toarray()
        return tf.multiply(inputs, mask) / (1 - self.drop_prob)


class GCN(tf.keras.Model):
    
    def __init__(self, adj_mat, input_dim, output_dim, hidden_layers,
                 activation, dropout, drop_input, drop_name, initializer, bias):
        super(GCN, self).__init__()
        adj_hat = np.identity(adj_mat.shape[0]) + adj_mat
        degree_mat = np.diag(1. / np.sqrt(np.sum(adj_hat, axis=1)))
        self.agg_mat = tf.cast(degree_mat @ adj_hat @ degree_mat,
                               dtype=tf.float32)
        self.gcn_layers = []
        drop_layer_init = get_dropout_layer(drop_name)
        self.gcn_layers.append(drop_layer_init(drop_input))
        for in_dim, out_dim in zip([input_dim] + hidden_layers[:-1],
                                   hidden_layers):
            self.gcn_layers.append(GraphDense(self.agg_mat, in_dim, out_dim,
                                              initializer, bias, activation))
            self.gcn_layers.append(drop_layer_init(dropout))
        self.gcn_layers.append(GraphDense(self.agg_mat, hidden_layers[-1],
                                          output_dim, initializer, bias, 'Id'))
    
    def call(self, in_x, training=False):
        for layer in self.gcn_layers:
            in_x = layer(in_x, training=training)
        return in_x
    
    def get_trainable_parameters(self):
        return [param for l in self.gcn_layers for param in l.get_weights()]
    
    def weight_decay(self, reg, include_last):
        if include_last:
            reg_layers = self.gcn_layers
        else:
            reg_layers = self.gcn_layers[:-1]
        reg_sum = 0.
        for layer in reg_layers:
            for weight in layer.get_weights():
                reg_sum += tf.nn.l2_loss(weight)
        return reg * tf.nn.l2_loss(weight)
        
