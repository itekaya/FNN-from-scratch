import numpy as np
import matplotlib.pyplot as plt

#this is .py file with model testing functions 


#this function calculates confusion matrix elements (true positive, true negative, false positive, false negative) based on predictions and ground truths for binary classification 
def getConfMatCompBin(ground_truth,predictions):
    #calculating elements
    true_positive = np.sum((predictions == 1)&(ground_truth == 1))
    true_negative = np.sum((predictions == 0)&(ground_truth == 0))
    false_positive = np.sum((predictions == 1)&(ground_truth == 0))
    false_negative = np.sum((predictions == 0)&(ground_truth == 1))
    #joinding all components into list so they can be outputted together
    confComp = [true_positive,true_negative,false_positive,false_negative]
    #returning results
    return confComp
#this function calculates accuracy of model results (based on confusion matrix results) for binary classification
def getAccuracyBin(true_positive,true_negative,false_positive,false_negative):
    #calculating accuracy (accuracy is ratio of true results out of all results)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) 
    #returning results
    return accuracy
#this function calculates precision of model results (based on confusion matrix results) for binary classification
def getPrecisionBin(true_positive,false_positive):
    #calculating precision (precision is ratio of true positive predictions out of all positive predicitions)
    precision = (true_positive) / (true_positive + false_positive) 
    #returning results
    return precision
##this function calculates recall of model results (based on confusion matrix results) for binary classification
def getRecallBin(true_positive,false_negative):
    #calculating recal (precision is ratio of true positives out of prediciton that are positve in ground truth)
    recall = (true_positive) / (true_positive + false_negative) 
    #returning results
    return recall
#this function gets accuracy of multi class model
def getAccuracy(ground_truth,predictions):
    #number of examples to process (and divide by) is acertained
    total = predictions.size
    #setting variable for counting
    accurate = 0
    #iterating through examples
    for iExample in range(total):
        #getting pair of values (prediction and ground truth) for current examples
        predictions_this = predictions[iExample]
        ground_truth_this = ground_truth[iExample]
        #evaluating if correct class was predicted
        if((predictions_this == ground_truth_this)):
            #if correct class was predicted, than counter for accurate predictions is increased.
            accurate = accurate + 1
    #getting final results (ratio of correct predictions to all examples)
    accuracy = accurate/total
    #returning results
    return accuracy
#
def getConfMatCompMulti(ground_truth,predictions, num_classes=10):
    #
    cm = np.zeros((num_classes, num_classes), dtype=int)
    #
    for t, p in zip(ground_truth, predictions):
        cm[t, p] += 1
    return cm
#
def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()