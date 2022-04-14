import argparse
from cProfile import label
import numpy as np

from data_utils import load_cora, split_data, accuracy, IoU


def label_propagation_binary(transition, label, train_mask, num_iter):
    pred = np.zeros(train_mask.shape[0])
    pred[train_mask == 1] = label[train_mask == 1]
    if num_iter == 0:
        P_uu = transition[train_mask == 0][:,train_mask == 0]
        P_ul = transition[train_mask == 0][:,train_mask == 1]
        pred[train_mask == 0] =  np.linalg.inv(
            np.identity(int((1 - train_mask).sum())) - P_uu) @ \
                P_ul @ label[train_mask == 1]
    else:
        for _ in range(num_iter):
            pred = transition @ pred
            pred[train_mask == 1] = label[train_mask == 1]
    return pred


def label_spreading_binary(transition, label, train_mask, num_iter, alpha):
    pred = np.zeros(train_mask.shape[0])
    pred[train_mask == 1] = label[train_mask == 1]
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
        label[one_hot_label[:,i] == 1] = 1
        result.append(binary_algo(label))
    output_one_hot = np.zeros_like(one_hot_label)
    output_one_hot[range(one_hot_label.shape[0]),np.argmax(np.stack(result, axis=1), axis=1)] = 1
    return output_one_hot


def label_propagation(adjacency, one_hot_label, train_mask, num_iter=0):
    deg_inv = np.diag(1. / np.sum(adjacency, axis=1))
    transition = deg_inv @ adjacency
    binary_algo = lambda label: label_propagation_binary(transition, label, train_mask,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=['label_prop', 'label_spread'])
    parser.add_argument('--train_size', '-t', type=int, default=20)
    parser.add_argument('--num_iter', '-n', type=int, default=0)
    parser.add_argument('--alpha', '-a', type=float, default=0.5)
    parser.add_argument('--undirected', '-u', type=bool, default=True)
    parser.add_argument('--metrics', '-m', type=str, nargs='*',
                        choices=['Accuracy', 'IoU'], default=['Accuracy'])
    args = parser.parse_args()
    adj_mat, _, labels = load_cora()
    train_mask, _, test_mask = split_data(labels,
                                          train_each_class=args.train_size,
                                          validation=0)
    metrics_dict = dict(
        Accuracy=accuracy,
        IoU=IoU
    )
    
    if args.undirected:
        adj_mat = np.maximum(adj_mat, adj_mat.T)
    degrees = np.sum(adj_mat, axis=1)
    ns_adj = adj_mat[degrees > 0][:, degrees > 0]
    ns_labels = labels[degrees > 0]
    ns_train_mask = train_mask[degrees > 0]
    
    if args.method == 'label_prop':
        ns_pred = label_propagation(ns_adj, ns_labels, ns_train_mask,
                                    num_iter=args.num_iter)
    elif args.method == 'label_spread':
        ns_pred = label_spreading(ns_adj, ns_labels, ns_train_mask,
                                  num_iter=args.num_iter, alpha=args.alpha)
    pred = np.zeros_like(labels)
    pred[degrees > 0] = ns_pred
    rand_idx = np.random.choice(int(labels.max() - labels.min()) + 1,
                                          size=(labels.shape[0] - \
                                              ns_pred.shape[0],)) + \
                                                  labels.min()

    pred[degrees == 0, rand_idx.astype(np.int32)] = 1
    metrics = {name: metrics_dict[name] for name in args.metrics}
    for name, metric in metrics.items():
        eval_result = metric(pred, labels, test_mask)
        print(f'{name} is {eval_result:.3f}.')
