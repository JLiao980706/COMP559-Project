import numpy as np
import scipy as spy
import tensorflow as tf

from cora_pipeline import train, cross_entropy_loss


def ensemble_predict(models, features):
    probs = []
    for model in models:
        probs.append(tf.nn.softmax(model(features), axis=-1).numpy())
    return np.stack(probs).mean(axis=0)


def train_ensenble(model_gen, num_models, portion, features, labels, train_mask,
                   valid_mask, loss_func, epochs, optimizer_name, lr, reg=5e-4, 
                   reg_last=False, patience=7, metrics={}):
    models = [model_gen() for _ in range(num_models)]
    for model in models:
        new_mask = np.zeros(train_mask.shape)
        for category_idx in range(labels.shape[1]):
            val_sample_ind = np.multiply(labels[:, category_idx], train_mask)
            nonzero_idx = spy.sparse.coo_array(val_sample_ind).nonzero()
            mask_val = np.random.choice([0, 1], size=nonzero_idx[0].shape[0],
                                        p=[1 - portion, portion])
            second_mask = spy.sparse.coo_array(mask_val, nonzero_idx,
                                               shape=val_sample_ind.shape
                                               ).toarray()
            new_mask += second_mask
        cur_train_mask = np.multiply(train_mask, new_mask)
        train(model, features, labels, cur_train_mask, valid_mask, loss_func,
              epochs, optimizer_name, lr, reg=reg, reg_last=reg_last,
              metrics=metrics, patience=patience)
    return models
    
            