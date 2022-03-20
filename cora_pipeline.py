import argparse
import json

import tensorflow as tf
import numpy as np

import models
from data_utils import load_cora, split_data, accuracy, IoU


def get_optimizer(name, lr):
    return dict(
        SGD=tf.keras.optimizers.SGD,
        Adam=tf.keras.optimizers.Adam
    )[name](learning_rate=lr)


def cross_entropy_loss(pred, labels, masks):
    return - tf.reduce_mean(
        tf.reduce_sum(tf.multiply(masks, tf.mutiply(labels, tf.math.log(pred))),
                      axis=-1))


def train(model, features, labels, train_mask, valid_mask, loss_func, epochs,
          optimizer_name, lr, metrics={}, verbose=0, record=0):
    optimizer = get_optimizer(optimizer_name, lr)
    train_loss_hist = {}
    train_metrics_hist = {name: {} for name in metrics.keys()}
    valid_loss_hist = {}
    valid_metrics_hist = {name: {} for name in metrics.keys()}
    
    for ep_idx in range(epochs):
        
        with tf.GradientTape() as tape:
            model_output = model(features)
            loss = loss_func(model_output, labels, train_mask)
        
        grads = tape.gradient(loss, model.get_trainable_parameters())
        optimizer.apply_gradients(zip(grads, model.get_trainable_parameters()))
        
        if verbose > 0 and (ep_idx + 1) % verbose == 0:
            train_loss, train_metrics = evaluate(model, features, labels,
                                                 train_mask, loss_func, metrics)
            valid_loss, valid_metrics = evaluate(model, features, labels,
                                                 valid_mask, loss_func, metrics)
            print(f'==========EPOCH #{ep_idx + 1}==========')
            print(f'Training Loss: {train_loss:.4f}')
            for name, val in train_metrics.items():
                print(f'Training {name}: {val:.4f}')
            print(f'Validation Loss: {valid_loss:.4f}')
            for name, val in valid_metrics.items():
                print(f'Validation {name}: {val:.4f}')
        
        if record > 0 and (ep_idx + 1) % record == 0:
            train_loss, train_metrics = evaluate(model, features, labels,
                                                 train_mask, loss_func, metrics)
            valid_loss, valid_metrics = evaluate(model, features, labels,
                                                 valid_mask, loss_func, metrics)
            train_loss_hist[ep_idx + 1] = train_loss
            valid_loss_hist[ep_idx + 1] = valid_loss
            for name, val_dict in train_metrics_hist.items():
                val_dict[ep_idx + 1] = train_metrics[name]
            for name, val_dict in valid_metrics_hist.items():
                val_dict[ep_idx + 1] = valid_metrics[name]

    return train_loss_hist, train_metrics_hist, valid_loss_hist,\
        valid_metrics_hist


def evaluate(model, features, labels, mask, loss_func, metrics={}):
    model_output = model(features)
    loss_val = loss_func(model_output, labels, mask).numpy()
    pred_labels = np.around(model_output.numpy())
    return loss_val, {mname: mf(pred_labels, labels, mask) for
                      mname, mf in metrics.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_config', type=str)
    parser.add_argument('--train_size', '-t', type=int, default=20)
    parser.add_argument('--valid_size', '-v', type=int, default=500)
    parser.add_argument('--epochs', '-e', type=int, default=1)
    parser.add_argument('--optimizer', '-o', type=str, choices=['SGD', 'Adam'],
                        default='SGD')
    parser.add_argument('--lr', '-l', type=float, default=0.1)
    parser.add_argument('--metrics', '-m', type=str, nargs='*',
                        choices=['Accuracy', 'IoU'], default=['Accuracy'])
    parser.add_argument('--verbose', '-b', type=int, default=0)
    parser.add_argument('--record', '-r', type=int, default=0)
    parser.add_argument('--result_fname', '-f', type=str, default='result')
    parser.add_argument('--model_fname', '-g', type=str, default=None)
    args = parser.parse_args()
    adj_mat, features, labels = load_cora()
    train_mask, val_mask, test_mask = split_data(labels, train_each_class=args.train_size, validation=args.valid_size)
    metrics_dict = dict(
        Accuracy=accuracy,
        IoU=IoU
    )
    metrics = {name: metrics_dict[name] for name in args.metrics}
    
    model = models.load_model(args.model_config, adj_mat, features.shape[1], labels.shape[1])
    tloss, vloss, tmetrics, vmetrics = train(model, features, labels,
                                             train_mask, val_mask,
                                             cross_entropy_loss, args.epochs,
                                             args.optimizer, args.lr,
                                             metrics=metrics,
                                             verbose=args.verbose,
                                             record=args.record)
    test_loss, test_metrics = evaluate(model, features, labels, test_mask,
                                       cross_entropy_loss, metrics)
    result_dict = dict(
        training_loss=[float(loss_val) for loss_val in tloss],
        validation_loss=[float(loss_val) for loss_val in vloss],
        test_loss=float(test_loss)
        )
    
    for metric_name in args.metrics:
        result_dict['training_' + metric_name] = [float(mval) for mval 
                                                  in tmetrics[metric_name]]
        result_dict['validation_' + metric_name] = [float(mval) for mval 
                                                    in vmetrics[metric_name]]
        result_dict['test_' + metric_name] = float(test_metrics[metric_name])
        
    with open(args.fname + '.txt', 'w+') as rfile:
        rfile.write(json.dumps(result_dict))
    
    if args.model_name is not None:
        model.save(args.model_fname + '.tm')
