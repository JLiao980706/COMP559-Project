import argparse
import numpy as np

from data_utils import accuracy


def label_propagation_binary(transition, label, train_mask, num_iter):
    pred = np.zeros(train_mask.shape[0])
    pred[train_mask == 1] = label
    if num_iter == 0:
        P_uu = transition[train_mask == 0][:,train_mask == 0]
        P_ul = transition[train_mask == 0][:,train_mask == 1]
        pred[train_mask == 0] =  np.linalg.inv(
            np.identity((1 - train_mask).sum) - P_uu) @ \
                P_ul @ label[train_mask == 1]
    else:
        for _ in range(num_iter):
            pred = transition @ pred
            pred[train_mask == 1] = label
    return pred


def label_spreading_binary(transition, label, train_mask, num_iter, alpha):
    pred = np.zeros(train_mask.shape[0])
    pred[train_mask == 1] = label
    pred_cpy = np.copy(pred)
    if num_iter == 0:
        return (1 - alpha) * \
            np.linalg.inv(np.identity(train_mask.shape[0]) - \
                alpha * transition) @ pred
    else:
        for _ in range(num_iter):
            pred = alpha * transition @ pred + (1 - alpha) * pred_cpy
        return pred



def ova_wrapper(binary_algo, one_hot_label):
    result = []
    for i in range(one_hot_label.shape[1]):
        label = - np.ones(one_hot_label.shape[0])
        label[one_hot_label[i] == 1] = 1
        result.append(binary_algo(label))
    output_one_hot = np.zeros_like(result[0])
    output_one_hot[:,np.argmax(np.stack(result, axis=0), axis=1)] = 1
    return output_one_hot


def label_propagation(adjacency, one_hot_label, train_mask, num_iter=0):
    deg_inv = np.diag(1. / np.sum(adjacency, axis=1))
    transition = deg_inv @ adjacency
    binary_algo = lambda label: label_propagation(transition, label, train_mask,
                                                  num_iter)
    return ova_wrapper(binary_algo, one_hot_label)
        

def label_spreading(adjacency, one_hot_label, train_mask, num_iter=0,
                    alpha=0.5):
    deg_inv_sqr = np.diag(np.sqrt(np.sum(adjacency, axis=1)))
    transition = deg_inv_sqr @ adjacency @ deg_inv_sqr
    binary_algo = lambda label: label_spreading_binary(transition, label,
                                                       train_mask, num_iter,
                                                       alpha)
    return ova_wrapper(binary_algo, one_hot_label)


if __name__ == '__main__':
    pass
