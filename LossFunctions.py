import numpy as np
#this is .py file with loss functions and their derivatives


# ================================================
# Mean Squared Error (MSE)
# ================================================
def MeanSquaredError(targets, predictions):
    diff = predictions - targets
    return 0.5 * np.sum(diff ** 2)

def MeanSquaredErrorDerivative(targets, predictions):
    return (predictions - targets)
# ================================================
# Binary Cross-Entropy (for sigmoid output)
# ================================================
def BinaryCrossEntropy(targets, predictions, eps=1e-12):
    #
    predictions = np.clip(predictions, eps, 1 - eps)
    #
    return -np.mean(targets * np.log(predictions) +
                    (1 - targets) * np.log(1 - predictions))

def BinaryCrossEntropyDerivative(targets, predictions, eps=1e-12):
    predictions = np.clip(predictions, eps, 1 - eps)
    return (predictions - targets) / (predictions * (1 - predictions))
#Cross-Entropy
def CrossEntropy(targets, predictions):
    #
    #prob = predictions[np.arrange(len(targets)),targets]
    #
    loss = -np.sum(targets * np.log(predictions + 1e-15))#1e-15 is added for numerical stability ->predictions can become zero, which lead to divide by zero error
    #
    return loss
def CrossEntropyDerivative(targets, predictions):
    #
    #grad = np.zeros_like(predictions)
    #grad[targets,np.arrange(predictions.shape[1])] = -1 / predictions[targets,np.arrange(predictions.shape[1])]
    #
    output = -targets / (predictions +1e-15)
    #
    return output
# ================================================
# Softmax + Cross-Entropy (for multi-class)
# ================================================
#this is only used in backpropagation and for this purpose only derivative is needed
def SoftmaxCrossEntropyDerivative(target_one_hot, softmax_vals):
    return (softmax_vals - target_one_hot)
