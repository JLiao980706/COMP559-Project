import numpy as np
from scipy import sparse


def load_cora(normalize, symmetrize):
    feat_label = sparse.load_npz('./data/cora_features_raw.npz').toarray()
    adj = sparse.load_npz('./data/cora_adjacency.npz').toarray()
    features = feat_label[:,:-7]
    if normalize:
        feat_norm = features.sum(axis=1)
        features[feat_norm > 0] /= feat_norm[feat_norm > 0].reshape((-1, 1))
    if symmetrize:
        adj = np.minimum(adj + adj.T, 1)
    return adj, features, feat_label[:, -7:]
        

def split_data(cora_labels, train_each_class=20, validation=500):
    train_mask = np.zeros((cora_labels.shape[0]))
    for class_idx in range(cora_labels.shape[1]):
        train_mask[np.random.choice(np.argwhere(cora_labels[:, class_idx])[:,0],
                                   size=train_each_class, replace=False)] = 1
    remaining_mask = 1 - train_mask
    val_idx = np.random.choice(np.argwhere(remaining_mask)[:,0],
                               size=validation, replace=False)
    remaining_mask[val_idx] = 2
    return train_mask, remaining_mask + train_mask - 1, \
        2 - 2 * train_mask - remaining_mask


def load_mask(train_size, idx):
    fname = 'data_split/mask_train' + str(train_size) + '_' + str(idx) + '.npz'
    data_dict = np.load(fname)
    return data_dict['train_mask'], data_dict['val_mask'], \
        data_dict['test_mask']


def accuracy(pred, label, mask):
    return np.multiply(mask.reshape((-1, 1)), np.multiply(pred, label)).sum() /\
        mask.sum()


def IoU(pred, label, mask):
    mask = mask.reshape((-1, 1))
    intersection = np.multiply(mask, np.multiply(pred, label))
    union = np.multiply(mask, 1 - np.multiply(1 - pred, 1 - mask))
    return np.divide(intersection.sum(axis=0), union.sum(axis=0)).mean()


