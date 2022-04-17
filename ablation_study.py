import argparse
import json
import time
from tracemalloc import start
import tensorflow as tf
import numpy as np

from data_utils import load_mask, accuracy, load_cora
import models
from cora_pipeline import train, cross_entropy_loss, evaluate


def run_experiment(cora, train_size, num_splits, repeats, model_config,
                   setting):
    adj_mat, features, labels = cora
    train_matrix = np.zeros((num_splits, repeats))
    val_matrix = np.zeros((num_splits, repeats))
    test_matrix = np.zeros((num_splits, repeats))
    # setting['epochs'] = 10 # TODO: comment out this line
    for split_idx in range(num_splits):
        train_mask, val_mask, test_mask = load_mask(train_size, split_idx)
        for repeat_idx in range(repeats):
            print(f'Running Split {split_idx}; repeat {repeat_idx}.')
            model = models.load_model(model_config, adj_mat, features.shape[1],
                                      labels.shape[1])
            train(model, features, labels, train_mask, val_mask, 
                  cross_entropy_loss, setting['epochs'], setting['optimizer'],
                  setting['lr'], reg=setting['reg'], 
                  reg_last=setting['reg_last'], metrics={'Accuracy': accuracy})
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
    # setting['epochs'] = 10 # TODO: comment out this line
    for train_size in [10, 20, 30, 40, 50]:
        print(f'==========TrainSize {train_size}==========')
        train_mat, val_mat, test_mat = run_experiment(cora, train_size,
                                                      num_splits, repeats,
                                                      model_config, setting)
        train_points.append((train_mat.mean(), train_mat.std()))
        valid_points.append((val_mat.mean(), val_mat.std()))
        test_points.append((test_mat.mean(), test_mat.std()))
    return train_points, valid_points, test_points


def hyper_param(cora, train_size, num_splits, repeats, model_config, setting):
    lr_range = [0.1, 0.03, 0.01, 0.003, 0.001]
    dropout_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
    weight_decay = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    # setting['epochs'] = 10 # TODO: comment out this line
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
    # setting['epochs'] = 10 # TODO: comment out this line
    for dropout_rate in (np.arange(10) / 10.):
        print(f'==========DropoutRate {dropout_rate}==========')
        model_config['dropout'] = dropout_rate
        model_config['drop_input'] = dropout_rate
        train_mat, val_mat, test_mat = run_experiment(cora, train_size,
                                                      num_splits, repeats,
                                                      model_config, setting)
        train_points.append((train_mat.mean(), train_mat.std()))
        valid_points.append((val_mat.mean(), val_mat.std()))
        test_points.append((test_mat.mean(), test_mat.std()))
    return train_points, valid_points, test_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, choices=['train_size',
                                                              'dropout',
                                                              'hyperparam',
                                                              'single'])
    parser.add_argument('model_config', type=str)
    parser.add_argument('--num_splits', '-n', type=int, default=3)
    parser.add_argument('--num_repeats', '-m', type=int, default=2)
    parser.add_argument('--train_size', '-t', type=int, default=20)
    parser.add_argument('--preprocess', '-p', type=bool, default=True)
    parser.add_argument('--symmetrize', '-s', type=bool, default=True)
    parser.add_argument('--epochs', '-e', type=int, default=1)
    parser.add_argument('--optimizer', '-o', type=str, choices=['SGD', 'Adam'],
                        default='Adam')
    parser.add_argument('--lr', '-l', type=float, default=0.1)
    parser.add_argument('--reg', '-a', type=float, default=1e-3)
    parser.add_argument('--reg_last', '-c', type=bool, default=False)
    parser.add_argument('--result_fname', '-f', type=str, default='result')
    args = parser.parse_args()
    result_dict = vars(args)
    
    cora = load_cora(args.preprocess, args.symmetrize)
    with open(args.model_config, 'r') as jfile:
        config_dict = json.load(jfile)
        
    start_time = time.time()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    if args.experiment_name == 'train_size':
        result_dict['train_points'], \
        result_dict['valid_points'], \
        result_dict['test_points'] = train_size_curve(cora, args.num_splits,
                                                      args.num_repeats,
                                                      config_dict, result_dict)
    elif args.experiment_name == 'dropout':
        result_dict['train_points'], \
        result_dict['valid_points'], \
        result_dict['test_points'] = dropout_curve(cora, args.train_size,
                                                   args.num_splits, 
                                                   args.num_repeats,
                                                   config_dict, result_dict)
    elif args.experiment_name == 'hyperparam':
        result_mat = hyper_param(cora, args.train_size, args.num_splits,
                                 args.num_repeats, config_dict, result_dict)
        result_dict['hyper_param_mat'] = result_mat.tolist()
    else:
        train_mat, val_mat, test_mat = run_experiment(cora, args.train_size,
                                                      args.num_splits,
                                                      args.repeats,
                                                      config_dict, result_dict)
        result_dict['train_matrix'] = train_mat.tolist()
        result_dict['valid_matrix'] = val_mat.tolist()
        result_dict['test_matrix'] = test_mat.tolist()
    print(f'Total Time: {time.time() - start_time:.3f}')
    with open(args.result_fname, 'w+') as new_jfile:
        new_jfile.write(json.dumps(result_dict))
    
