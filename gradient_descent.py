import numpy as np
from LossFunctions import *
import SuppFunctions  # for clip_gradient
from ErrorClasses import *


# Helper: make sure inputs/targets have correct shape
def _prepare_data(network, inputs, targets):
    inputs = np.array(inputs, dtype=float)
    targets = np.array(targets, dtype=float)

    # If targets are (n_samples, n_outputs), fix them
    if targets.shape[0] != network.weights_list[-1].shape[0]:
        targets = targets.T
        if(targets.shape[0] != network.weights_list[-1].shape[0]):
            raise NotSupportedInputGiven("Network training","Targets does not match network outputs")

    n_samples = inputs.shape[1]
    return inputs, targets, n_samples


# 1. MINI-BATCH STOCHASTIC GRADIENT DESCENT

def train_minibatch_sgd(network,
                        inputs,
                        targets,
                        epochs,
                        learning_rate,
                        batch_size,
                        loss_derivative,
                        l1_coeff=0.0,
                        l2_coeff=0.0,
                        grad_clip=0.0):
    """
    Train the FNN using mini-batch SGD.

    inputs:  np.ndarray of shape (n_features, n_samples)
    targets: np.ndarray of shape (n_outputs, n_samples)
    """

    inputs, targets, n_samples = _prepare_data(network, inputs, targets)

    for epoch in range(epochs):

        # Shuffle sample indices so that batches are random each epoch
        indices = np.random.permutation(n_samples)

        # Process the dataset in chunks of size batch_size
        for start in range(0, n_samples, batch_size):

            # Integer indices for this batch
            batch_idx = indices[start:start + batch_size]

            # Batch data
            x_batch = inputs[:, batch_idx]
            y_batch = targets[:, batch_idx]
            B = x_batch.shape[1]  # effective batch size

            # Forward pass on the batch
            z_values, a_values = network.forward(x_batch)

            # Backward pass (gradients summed over batch)
            grad_W = network.backward(z_values, a_values, y_batch, loss_derivative)

            # Weight update
            for i in range(len(network.weights_list)):
                W = network.weights_list[i]

                # Convert summed gradient to average gradient
                g = grad_W[i] / B

                # ----- L2 regularization (no bias) -----
                if l2_coeff != 0.0:
                    reg = l2_coeff * W.copy()
                    reg[:, 0] = 0.0  # do not regularize bias column
                    g += reg

                # ----- L1 regularization (no bias) -----
                if l1_coeff != 0.0:
                    reg_l1 = l1_coeff * np.sign(W.copy())
                    reg_l1[:, 0] = 0.0
                    g += reg_l1

                # Optional gradient clipping
                if grad_clip != 0.0:
                    g = SuppFunctions.clip_gradient(g, grad_clip)

                # Gradient descent step
                network.weights_list[i] = W - learning_rate * g

    return network


# 1b. MINI-BATCH SGD WITH MOMENTUM

def train_minibatch_sgd_momentum(network,
                                 inputs,
                                 targets,
                                 epochs,
                                 learning_rate,
                                 batch_size,
                                 loss_derivative,
                                 momentum=0.9,
                                 l1_coeff=0.0,
                                 l2_coeff=0.0,
                                 grad_clip=0.0):
    """
    Train the FNN using mini-batch SGD with classical momentum.
    """

    inputs, targets, n_samples = _prepare_data(network, inputs, targets)

    # Velocity (momentum) for each layer
    v = [np.zeros_like(W) for W in network.weights_list]

    for epoch in range(epochs):

        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):

            batch_idx = indices[start:start + batch_size]

            x_batch = inputs[:, batch_idx]
            y_batch = targets[:, batch_idx]
            B = x_batch.shape[1]

            # Forward / Backward on this batch
            z_values, a_values = network.forward(x_batch)
            grad_W = network.backward(z_values, a_values, y_batch, loss_derivative)

            for i in range(len(network.weights_list)):
                W = network.weights_list[i]
                g = grad_W[i] / B

                # L2 regularization (no bias)
                if l2_coeff != 0.0:
                    reg = l2_coeff * W.copy()
                    reg[:, 0] = 0.0
                    g += reg

                # L1 regularization (no bias)
                if l1_coeff != 0.0:
                    reg_l1 = l1_coeff * np.sign(W.copy())
                    reg_l1[:, 0] = 0.0
                    g += reg_l1

                if grad_clip != 0.0:
                    g = SuppFunctions.clip_gradient(g, grad_clip)

                # Momentum update: v = mu * v - lr * g
                v[i] = momentum * v[i] - learning_rate * g

                # Weight update: w = w + v
                network.weights_list[i] = W + v[i]

    return network


# 2. MINI-BATCH RMSPROP

def train_minibatch_rmsprop(network,
                            inputs,
                            targets,
                            epochs,
                            learning_rate,
                            batch_size,
                            loss_derivative,
                            beta=0.9,
                            epsilon=1e-8,
                            l1_coeff=0.0,
                            l2_coeff=0.0,
                            grad_clip=0.0):
    """
    Train the FNN using mini-batch RMSprop.

    beta: decay rate for the running average of squared gradients
    """

    inputs, targets, n_samples = _prepare_data(network, inputs, targets)

    # Running average of squared gradients for each layer
    r = [np.zeros_like(W) for W in network.weights_list]

    for epoch in range(epochs):

        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):

            batch_idx = indices[start:start + batch_size]

            x_batch = inputs[:, batch_idx]
            y_batch = targets[:, batch_idx]
            B = x_batch.shape[1]

            # Forward / Backward on this batch
            z_values, a_values = network.forward(x_batch)
            grad_W = network.backward(z_values, a_values, y_batch, loss_derivative)

            # RMSprop update per layer
            for i in range(len(network.weights_list)):

                g = grad_W[i] / B
                W = network.weights_list[i]

                # L2 regularization (no bias)
                if l2_coeff != 0.0:
                    reg = l2_coeff * W.copy()
                    reg[:, 0] = 0.0
                    g += reg

                # L1 regularization (no bias)
                if l1_coeff != 0.0:
                    reg_l1 = l1_coeff * np.sign(W.copy())
                    reg_l1[:, 0] = 0.0
                    g += reg_l1

                if grad_clip != 0.0:
                    g = SuppFunctions.clip_gradient(g, grad_clip)

                # Update running average of squared gradients
                r[i] = beta * r[i] + (1.0 - beta) * (g * g)

                # RMSprop parameter update
                update = learning_rate * g / (np.sqrt(r[i]) + epsilon)

                # Gradient descent step
                network.weights_list[i] = W - update

    return network


# 3. MINI-BATCH NESTEROV ACCELERATED GRADIENT (NAG)

def train_minibatch_nag(network,
                        inputs,
                        targets,
                        epochs,
                        learning_rate,
                        batch_size,
                        loss_derivative,
                        momentum=0.9,
                        l1_coeff=0.0,
                        l2_coeff=0.0,
                        grad_clip=0.0):
    """
    Train the FNN using mini-batch Nesterov Accelerated Gradient (NAG).
    """

    inputs, targets, n_samples = _prepare_data(network, inputs, targets)

    # Velocity (momentum) for each layer, same shape as weights
    v = [np.zeros_like(W) for W in network.weights_list]

    for epoch in range(epochs):

        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):

            batch_idx = indices[start:start + batch_size]

            x_batch = inputs[:, batch_idx]
            y_batch = targets[:, batch_idx]
            B = x_batch.shape[1]

            # 1. Look-ahead step: w_lookahead = w + momentum * v
            for i in range(len(network.weights_list)):
                network.weights_list[i] = network.weights_list[i] + momentum * v[i]

            # 2. Forward / Backward at look-ahead weights
            z_values, a_values = network.forward(x_batch)
            grad_W = network.backward(z_values, a_values, y_batch, loss_derivative)

            # 3. Update velocity and weights (Nesterov rule)
            for i in range(len(network.weights_list)):

                g = grad_W[i] / B
                W = network.weights_list[i]

                # L2 regularization (no bias)
                if l2_coeff != 0.0:
                    reg = l2_coeff * W.copy()
                    reg[:, 0] = 0.0
                    g += reg

                # L1 regularization (no bias)
                if l1_coeff != 0.0:
                    reg_l1 = l1_coeff * np.sign(W.copy())
                    reg_l1[:, 0] = 0.0
                    g += reg_l1

                if grad_clip != 0.0:
                    g = SuppFunctions.clip_gradient(g, grad_clip)

                # Save previous velocity
                v_prev = v[i].copy()

                # NAG velocity update:
                v[i] = momentum * v[i] - learning_rate * g

                # We are currently at w_look = w_t + momentum * v_prev.
                # We want final w_{t+1} = w_t + v_new.
                # => w_{t+1} = w_look + (v_new - momentum * v_prev)
                network.weights_list[i] = W + (v[i] - momentum * v_prev)

    return network


# 4. MINI-BATCH ADAM OPTIMIZER

def train_minibatch_adam(network,
                         inputs,
                         targets,
                         epochs,
                         learning_rate,
                         batch_size,
                         loss_derivative,
                         beta1=0.9,
                         beta2=0.999,
                         epsilon=1e-8,
                         l1_coeff=0.0,
                         l2_coeff=0.0,
                         grad_clip=0.0):

    inputs, targets, n_samples = _prepare_data(network, inputs, targets)

    # Initialize Adam moment vectors for each layer.
    m = [np.zeros_like(W) for W in network.weights_list]  # first moment
    v = [np.zeros_like(W) for W in network.weights_list]  # second moment

    t = 0  # Adam time step counter (increments per batch)

    # Loop over dataset
    for epoch in range(epochs):

        # Shuffle data order this epoch
        indices = np.random.permutation(n_samples)

        # Mini-batch loop
        for start in range(0, n_samples, batch_size):

            batch_idx = indices[start:start + batch_size]

            # Extract batch: shapes (features, B) and (outputs, B)
            x_batch = inputs[:, batch_idx]
            y_batch = targets[:, batch_idx]
            B = x_batch.shape[1]

            # Forward and Backward on this batch
            z_values, a_values = network.forward(x_batch)
            grad_W = network.backward(z_values, a_values, y_batch, loss_derivative)

            # Increment Adam time step
            t += 1

            # Adam update for each layer
            for i in range(len(network.weights_list)):

                g = grad_W[i] / B
                W = network.weights_list[i]

                # L2 regularization (no bias)
                if l2_coeff != 0.0:
                    reg = l2_coeff * W.copy()
                    reg[:, 0] = 0.0
                    g += reg

                # L1 regularization (no bias)
                if l1_coeff != 0.0:
                    reg_l1 = l1_coeff * np.sign(W.copy())
                    reg_l1[:, 0] = 0.0
                    g += reg_l1

                if grad_clip != 0.0:
                    g = SuppFunctions.clip_gradient(g, grad_clip)

                # 1. Update biased first moment estimate
                m[i] = beta1 * m[i] + (1.0 - beta1) * g

                # 2. Update biased second moment estimate
                v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)

                # 3. Bias corrections
                m_hat = m[i] / (1.0 - beta1 ** t)
                v_hat = v[i] / (1.0 - beta2 ** t)

                # 4. Compute parameter update
                update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # 5. Apply update (gradient descent direction)
                network.weights_list[i] = W - update

    return network
