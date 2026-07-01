import numpy as np
import wandb

from FNN import FNN
from LossFunctions import (
    SoftmaxCrossEntropy,
    SoftmaxCrossEntropyDerivative,
    MeanSquaredError,
    MeanSquaredErrorDerivative,
)

from training_utils import (
    ACTIVATION_FUNCS,
    OUTPUT_ACTIVATION_FUNCS,
    load_dataset,
    one_hot,
    compute_loss_with_l2,
    train_one_epoch,
    accuracy,
)


# DEFAULT CONFIG
DEFAULT_CONFIG = dict(
    # --- data config ---
    dataset_type="csv",                 # "csv" or "cifar10"
    train_path="Data/fashion-mnist_train.csv",   # used when dataset_type == "csv"
    test_path="Data/fashion-mnist_test.csv",     # used when dataset_type == "csv"
    cifar10_root="Data/cifar",     # used when dataset_type == "cifar10"
    label_column="label",               # name of label column for CSV datasets
    num_classes=10,                     # e.g. 10 for Fashion-MNIST or CIFAR-10
    normalize=True,                     # divide inputs by 255 if data is 0–255

    # --- architecture ---
    num_hidden_layers=3, # number of hidden layers
    n_hidden_units=256,  # neurons per hidden layer (same in each hidden layer)
    activation_hidden="relu", # "relu", "tanh", "sigmoid", "identity", "leaky_relu"
    activation_output="softmax_vec",
    weights_init="HeNor", # "Zero", "RandomUni", "RandomNor", "XavUni", "XavNor", "HeUni", "HeNor"

    # --- training ---
    epochs=15,
    learning_rate=5e-4,
    batch_size=128,
    optimizer="adam", # "sgd", "sgd_momentum", "nag", "rmsprop", "adam"

    # --- optimizer-specific ---
    momentum=0.9, # sgd_momentum, nag
    rmsprop_beta=0.9, # rmsprop (beta)
    beta1=0.9, # adam
    beta2=0.999, # adam
    epsilon=1e-8, # adam, rmsprop

    # --- regularization ---
    l1_coeff=0.0,
    l2_coeff=0.0,

    # --- loss as hyperparameter ---
    loss_name="cross_entropy", # "cross_entropy" or "mse"

    # --- data split ---
    val_fraction=0.1,

    # --- gradient clipping ---
    grad_clip=0.0,
)

# Main training function
def main():
    # Initialize wandb and load config
    run = wandb.init(config=DEFAULT_CONFIG, project="ffnn-from-scratch")
    cfg = wandb.config  # this holds defaults + sweep overrides

    # --- choose loss function + derivative based on cfg.loss_name ---
    if cfg.loss_name == "cross_entropy":
        loss_fn = SoftmaxCrossEntropy
        loss_deriv = SoftmaxCrossEntropyDerivative
    elif cfg.loss_name == "mse":
        loss_fn = MeanSquaredError
        loss_deriv = MeanSquaredErrorDerivative
    else:
        raise ValueError(f"Unknown loss_name: {cfg.loss_name}")

    # --- build hidden layer sizes from num_hidden_layers + n_hidden_units ---
    n_hidden_units = [cfg.n_hidden_units] * cfg.num_hidden_layers

    # --- map string activations to functions ---
    act_hidden_fn = ACTIVATION_FUNCS[cfg.activation_hidden]
    act_output_fn = OUTPUT_ACTIVATION_FUNCS[cfg.activation_output]

    # --- load data (CSV or CIFAR-10) + split train/val ---
    (X_train_full, y_train_full), (X_test, y_test) = load_dataset(cfg)

    Y_train_full = one_hot(y_train_full, num_classes=cfg.num_classes)
    Y_test = one_hot(y_test, num_classes=cfg.num_classes)

    N = X_train_full.shape[1]
    indices = np.random.permutation(N)
    val_size = int(N * cfg.val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train = X_train_full[:, train_idx]
    Y_train = Y_train_full[:, train_idx]

    X_val = X_train_full[:, val_idx]
    Y_val = Y_train_full[:, val_idx]

    # build FNN 
    input_dim = X_train.shape[0]
    num_classes = cfg.num_classes

    layer_sizes = [input_dim] + list(n_hidden_units) + [num_classes]
    activations = [act_hidden_fn] * cfg.num_hidden_layers + [act_output_fn]

    net = FNN(
        weights_info=layer_sizes,
        activ_functions_info=activations,
        method_ini=cfg.weights_init,
    )

    #  training loop (log train/val metrics each epoch) 
    for epoch in range(cfg.epochs):
        net = train_one_epoch(net, X_train, Y_train, cfg, loss_deriv)

        train_loss = compute_loss_with_l2(net, X_train, Y_train, cfg.l2_coeff, loss_fn)
        train_acc = accuracy(net, X_train, Y_train)

        val_loss = compute_loss_with_l2(net, X_val, Y_val, cfg.l2_coeff, loss_fn)
        val_acc = accuracy(net, X_val, Y_val)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "learning_rate": cfg.learning_rate,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "grad_clip": cfg.grad_clip,
                "l1_coeff": cfg.l1_coeff,
                "l2_coeff": cfg.l2_coeff,
            }
        )

    # final test metrics 
    test_loss = compute_loss_with_l2(net, X_test, Y_test, cfg.l2_coeff, loss_fn)
    test_acc = accuracy(net, X_test, Y_test)

    wandb.log(
        {
            "final_test_loss": test_loss,
            "final_test_acc": test_acc,
        }
    )

    print(
        f"[RUN DONE] opt={cfg.optimizer}, act={cfg.activation_hidden}, "
        f"layers={cfg.num_hidden_layers}, width={cfg.n_hidden_units} -> "
        f"test_acc={test_acc:.4f}"
    )

    run.finish()


if __name__ == "__main__":
    main()
