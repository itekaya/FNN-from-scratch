# training_utils.py

import os
import pickle
import numpy as np
import pandas as pd

from ActivFunctions import (
    relu,
    sigmoid,
    tanh,
    softmax_vec,
    identity,
    leaky_relu,
)
from gradient_descent import (
    train_minibatch_sgd,
    train_minibatch_sgd_momentum,
    train_minibatch_rmsprop,
    train_minibatch_nag,
    train_minibatch_adam,
)


# ACTIVATION MAPPING

ACTIVATION_FUNCS = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "identity": identity,
    "leaky_relu": leaky_relu,
}

OUTPUT_ACTIVATION_FUNCS = {
    "softmax_vec": softmax_vec,
}


# DATA LOADERS

def load_csv_dataset(train_path, test_path, label_column="label", normalize=True):
    """
    Generic CSV loader:
    - One row per sample
    - One column is the label (label_column)
    - All other columns are numeric features 
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df[label_column].to_numpy()
    X_train = train_df.drop(columns=[label_column]).to_numpy().astype(np.float32)

    y_test = test_df[label_column].to_numpy()
    X_test = test_df.drop(columns=[label_column]).to_numpy().astype(np.float32)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    # (features, samples)
    X_train = X_train.T
    X_test = X_test.T

    return (X_train, y_train), (X_test, y_test)


def load_cifar10_python(root, normalize=True):
    """
    Load original CIFAR-10

    Returns:
        (X_train, y_train), (X_test, y_test)
    where X_* has shape (features, samples) = (3072, N).
    """

    def _load_batch(filename):
        path = os.path.join(root, filename)
        with open(path, "rb") as f:
            batch = pickle.load(f, encoding="latin1")
        X = batch["data"].astype(np.float32)         # (10000, 3072)
        y = np.array(batch["labels"], dtype=np.int64)
        return X, y

    # training batches 1..5
    X_list = []
    y_list = []
    for i in range(1, 6):
        Xb, yb = _load_batch(f"data_batch_{i}")
        X_list.append(Xb)
        y_list.append(yb)

    X_train = np.concatenate(X_list, axis=0)   # (50000, 3072)
    y_train = np.concatenate(y_list, axis=0)   # (50000,)

    # test batch
    X_test, y_test = _load_batch("test_batch")  # (10000, 3072), (10000,)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    # (features, samples)
    X_train = X_train.T   # (3072, 50000)
    X_test = X_test.T     # (3072, 10000)

    return (X_train, y_train), (X_test, y_test)


def load_dataset(cfg):
    """
    Wrapper that chooses the right loader based on cfg.dataset_type.
    """
    if cfg.dataset_type == "csv":
        return load_csv_dataset(
            cfg.train_path,
            cfg.test_path,
            label_column=cfg.label_column,
            normalize=cfg.normalize,
        )
    elif cfg.dataset_type == "cifar10":
        return load_cifar10_python(
            cfg.cifar10_root,
            normalize=cfg.normalize,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}")


# UTILS: one-hot, loss+L2, optimizer dispatch, metrics

def one_hot(y, num_classes):
    out = np.zeros((num_classes, y.shape[0]))
    out[y, np.arange(y.shape[0])] = 1
    return out


def compute_loss_with_l2(network, X, Y, l2_coeff, loss_fn):
    """
    Total loss = data loss (MSE or CrossEntropy) + L2 penalty (if l2_coeff > 0).
    Bias weights (last column) are not regularized.
    """
    _, a_values = network.forward(X)
    outputs = a_values[-1]

    data_loss = loss_fn(Y, outputs)

    if l2_coeff != 0.0:
        l2 = 0.0
        for W in network.weights_list:
            W_no_bias = W[:, :-1]
            l2 += np.sum(W_no_bias**2)
        data_loss += 0.5 * l2_coeff * l2

    return data_loss


def train_one_epoch(net, X, Y, cfg, loss_derivative):
    """
    Train the network for exactly ONE epoch using the optimizer specified in cfg.
    All common hyperparameters are taken from cfg, including grad_clip.
    """
    opt = cfg.optimizer.lower()
    gc = cfg.grad_clip

    if opt == "sgd":
        return train_minibatch_sgd(
            network=net,
            inputs=X,
            targets=Y,
            epochs=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            loss_derivative=loss_derivative,
            l1_coeff=cfg.l1_coeff,
            l2_coeff=cfg.l2_coeff,
            grad_clip=gc,
        )

    if opt == "sgd_momentum":
        return train_minibatch_sgd_momentum(
            network=net,
            inputs=X,
            targets=Y,
            epochs=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            loss_derivative=loss_derivative,
            momentum=cfg.momentum,
            l1_coeff=cfg.l1_coeff,
            l2_coeff=cfg.l2_coeff,
            grad_clip=gc,
        )

    if opt == "nag":
        return train_minibatch_nag(
            network=net,
            inputs=X,
            targets=Y,
            epochs=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            loss_derivative=loss_derivative,
            momentum=cfg.momentum,
            l1_coeff=cfg.l1_coeff,
            l2_coeff=cfg.l2_coeff,
            grad_clip=gc,
        )

    if opt == "rmsprop":
        return train_minibatch_rmsprop(
            network=net,
            inputs=X,
            targets=Y,
            epochs=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            loss_derivative=loss_derivative,
            beta=cfg.rmsprop_beta,
            epsilon=cfg.epsilon,
            l1_coeff=cfg.l1_coeff,
            l2_coeff=cfg.l2_coeff,
            grad_clip=gc,
        )

    if opt == "adam":
        return train_minibatch_adam(
            network=net,
            inputs=X,
            targets=Y,
            epochs=1,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            loss_derivative=loss_derivative,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            epsilon=cfg.epsilon,
            l1_coeff=cfg.l1_coeff,
            l2_coeff=cfg.l2_coeff,
            grad_clip=gc,
        )

    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def accuracy(network, X, Y):
    _, a_values = network.forward(X)
    preds = np.argmax(a_values[-1], axis=0)
    labels = np.argmax(Y, axis=0)
    return np.mean(preds == labels)
