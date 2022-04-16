import tensorflow as tf
import numpy as np

from data_utils import load_mask, accuracy
import models
from cora_pipeline import train, cross_entropy_loss, evaluate


def run_experiment(cora, train_size, num_splits, repeats, model_config,
                   setting):
    adj_mat, features, labels = cora
    train_matrix = np.zeros((num_splits, repeats))
    val_matrix = np.zeros((num_splits, repeats))
    test_matrix = np.zeros((num_splits, repeats))
    setting['epochs'] = 10 # TODO: comment out this line
    for split_idx in range(num_splits):
        train_mask, val_mask, test_mask = load_mask(train_size, split_idx)
        for repeat_idx in range(repeats):
            model = models.load_model(model_config, adj_mat, features.shape[1],
                                      labels.shape[1])
            train(model, features, labels, train_mask, val_mask, 
                  cross_entropy_loss, setting['epochs'], setting['optimizer'],
                  setting['lr'], reg=setting['reg'], 
                  reg_last=setting['reg_last'], metrics=setting['metrics'],
                  verbose=setting['verbose'], record=setting['record'])
            _, train_metrics = evaluate(model, features, labels, train_mask,
                                        cross_entropy_loss,
                                        {'Accuracy': accuracy})
            _, val_metrics = evaluate(model, features, labels, val_mask,
                                      cross_entropy_loss,
                                      {'Accuracy': accuracy})
            _, test_metrics = evaluate(model, features, labels, test_mask,
                                       cross_entropy_loss,
                                       {'Accuracy': accuracy})
            train_matrix[split_idx, repeat_idx] = train_metrics['Accuracy']
            val_matrix[split_idx, repeat_idx] = val_metrics['Accuracy']
            test_matrix[split_idx, repeat_idx] = test_metrics['Accuracy']
    return train_matrix, val_matrix, test_matrix


def train_size_curve(cora, num_splits, repeats, model_config, setting):
    train_points = []
    valid_points = []
    test_points = []
    setting['epochs'] = 10 # TODO: comment out this line
    for train_size in [10, 20, 30, 40, 50]:
        train_mat, val_mat, test_mat = run_experiment(cora, train_size,
                                                      num_splits, repeats,
                                                      model_config, setting)
        train_points.append(train_mat.mean(), train_mat.std())
        valid_points.append(val_mat.mean(), val_mat.std())
        test_points.append(test_mat.mean(), test_mat.std())
    return train_points, valid_points, test_points


def hyper_param(cora, train_size, num_splits, repeats, model_config, setting):
    lr_range = [0.1, 0.03, 0.01, 0.003, 0.001]
    dropout_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
    weight_decay = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    setting['epochs'] = 10 # TODO: comment out this line
    result_matrix = np.zeros((5, 5, 5))
    for lr_idx, lr in enumerate(lr_range):
        for dr_idx, dr in enumerate(dropout_rate):
            for wd_idx, wd in enumerate(weight_decay):
                print(f'Learning Rate Idx: {lr_idx}')
                print(f'Dropout Rate Idx: {dr_idx}')
                print(f'Weight Decay Idx: {wd_idx}')
                model_config['dropout'] = dr
                model_config['drop_input'] = dr
                setting['lr'] = lr
                setting['reg'] = wd
                _, _, test_matrix = run_experiment(cora, train_size, num_splits,
                                                   repeats, model_config,
                                                   setting)
                result_matrix[lr_idx, dr_idx, wd_idx] = test_matrix.mean()
    return result_matrix
                

def dropout_curve(cora, train_size, num_splits, repeats, model_config, setting):
    train_points = []
    valid_points = []
    test_points = []
    for dropout_rate in (np.arange(10) / 10.):
        model_config['dropout'] = dropout_rate
        model_config['drop_input'] = dropout_rate
        train_mat, val_mat, test_mat = run_experiment(cora, train_size,
                                                      num_splits, repeats,
                                                      model_config, setting)
        train_points.append(train_mat.mean(), train_mat.std())
        valid_points.append(val_mat.mean(), val_mat.std())
        test_points.append(test_mat.mean(), test_mat.std())
    return train_points, valid_points, test_points


if __name__ == '__main__':
    pass
    
