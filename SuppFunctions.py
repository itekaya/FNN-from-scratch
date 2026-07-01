import numpy as np
from ActivFunctions import *
from ErrorClasses import *
#this is .py file with supporting functions 


#this function checks if np array is of desired data type and if not changes it into desired data type
def ensureDtypeNpArray(array_in,data_type):
    #verification of dtype
    if(array_in.dtype == data_type):#if data type is as desired, then input array does not change
        array_out = array_in
    else:#if not the same, then it is different and needs to be changed
        array_out = np.asarray(array_in,data_type)
    #output out
    return array_out
#this function evaluates if data type of given array is composed of rational numbers
def isRationalNumber(array):
    #verification of data type is rational
    answer = ((np.issubdtype(array.dtype, np.integer)) | (np.issubdtype(array.dtype, np.floating)))
    #output out
    return answer
#this function verifies if given 2D array is row vector
def isRowVector(array):
    #getting shape of array
    shape = array.shape
    #checking if requirments for row vector are met, i.e only one row and at least one column (second argument is more of verification if given array was 2D)
    answer = (shape[0] == 1 )& (shape[1] >= 1)
    #returning results
    return answer
#this function verifies if given 2D array is column vector
def isColVector(array):
    #getting shape of array
    shape = array.shape
    #checking if requirments for column vector are met, i.e only one column and at least one row (first argument is more of verification if given array was 2D)
    answer = (shape[0] >= 1) & (shape[1] == 1)
    #returning results
    return answer
#this function verifies if given 2D array is array (basically it is checke dif it is empty in any of dimensions)
def isArray(array):
    #getting shape of array
    shape = array.shape
    #checking if requirments for array are met, i.e. having informations in 2 dimensions
    answer = (shape[0] >= 1) & (shape[1] >= 1)
    return answer
#this function ensures that given input would be suitable for input propagation through ANN
def getProperInputArray(array):
    #for proper representation the input, the input vector needs to be a column vector or array. It is checked if that's the case and changed if not.
    if(isRowVector(array)):#if it is vector input, then it has to be in column format. So, if it is in row format than it has to be transposed.
        input_format = array.T
    elif(isColVector(array)):
        input_format = array
    elif(isArray(array)):
        input_format = array
    else:#if none of above true, then given variable is in non implemented format. Appropriate error must be passed to user. 
        raise NotSupportedInputGiven("input propagation","Given input is in dimensions for which input propagation is not implemented")
    #input is returned in proper format
    return input_format
#this function adds input values to represent bias (1s) into given input
def addBiasInput(input):
    #getting shape of given input to know how big holding variable should be initialized
    shapeInput = input.shape
    #initializing holding variable for assignment of inputs
    newInput = np.empty([(shapeInput[0] + 1),shapeInput[1]],dtype = input.dtype)#as single input is assumed to be in column format, than additional row has to be added for bias input
    #assigning values to proper places, i.e. bias inputs as first elements as biases are [0] elements in neuron weights
    newInput[0,:] = 1.0
    newInput[1:,:] = input
    #returning input with added bias input
    return newInput
#this function puts matrix multiplication results (input propagation) through activation functions for a layer. As in this implementation each neuron in layer can have different activation function each neurons activation needs to be done separately
def activationLayer(input,activ_functions_list):
    #to create properly sized output variable (pre-allocation) information of input(output of matrix multiplication) shape is needed
    shapeInput = input.shape
    #output variable is declared for pre-allocation
    output = np.empty(shapeInput)
    #iterating through neurons as each need to be calculated separetely
    for iNeuron in range(input.shape[0]):
        #selecting input for current neuron (row vector for current neuron)
        input_current = input[iNeuron,:]
        #getting activation function of current neuron
        activ_function_current = activ_functions_list[iNeuron]
        #puttign input through activation function
        output_current = activ_function_current(input_current)
        #assignign neuron output to output variable
        output[iNeuron,:] = output_current
    #returning the output
    return output
#this function calculates loss derivative for each neuron in output layer
def derLoss(targets,a_output,loss_derivative):
    #to create properly sized output variable (pre-allocation) information of input(targets or a_output as they should have same shape) shape is needed
    shapeInput = a_output.shape
    #output variable is declared for pre-allocation
    dL_dy_all = np.empty(shapeInput)
    #iterating through neurons as each need to be calculated separetely
    for iNeuron in range(shapeInput[0]):
        #selecting input for current neuron (row vector for current neuron)
        y_row = targets[iNeuron,:]
        a_row = a_output[iNeuron,:]
        #calculating loss
        dL_dy = loss_derivative(y_row,a_row)
        #assigning loss of neuron across inputs(batch) to pre-allocated variable
        dL_dy_all[iNeuron,:] = dL_dy
    #returning the output
    return dL_dy_all
#this function calculates gradient of loss for subsequent layers
def getDelta(der_prev,z,activ_functions_list):
    #depending if neurons in layer have the same activation function or different gradient of loss would be calculated differently 
    if(all((activ_function.__code__.co_code == activ_functions_list[0].__code__.co_code) for activ_function in activ_functions_list)):#it is verified if all activation functions are the same. To get proper result codes needs to be compared as they would be the same for same function, even as different bjects. Simple comparision would yield false even two objects of same function were compared.
            #softmax function is a special case, as it can be only used with vector inputs, so all activation functions needs to be the same in the layer for it to function (makes no sense when calculated neuron by neuron as in different activ function in layer case). So, in practice function need to have different names for both cases (if for some reason user wants to use softmax in second case).
            if(activ_functions_list[0].__code__.co_code == softmax.__code__.co_code):
                activ_function = softmax_vec
            else:##everything except softmax doesn't need different function version for vector and scalar input
                activ_function = activ_functions_list[0]
            #getting derivative of activation function function
            activ_function_der = getDer(activ_function)
            #applying it for input
            a_der = activ_function_der(z)
            #calculating gradient of loss
            if(type(a_der) == np.ndarray):#some special case occurs in np arrays and for evaluation of it .ndim property is needed, but if derivative is value then such property does not exist. Thankfully situation does not occur (at least was not caught) when der is value so this can be used to avoid errors when der is value
                if(a_der.ndim == 2):#this is normal situation
                    delta = der_prev * a_der
                elif(a_der.ndim == 3):# in case of softmax, the derivative is 2D Jacobiam matrix, which results in following shape (num_examples,num_class,num_class) and loss derivative is in need of special method of calculation
                    #getting shape of output gradient of loss
                    delta = np.zeros((der_prev.shape[0],a_der.shape[0]))
                    #iterating through examples
                    for iExample in range(a_der.shape[2]):
                        delta[:,iExample] = der_prev[:,iExample] @ a_der[iExample]
            else:#if derivatives are values, then calculation is simple
                delta = der_prev * a_der
    else:#if neurons have different activation function in layer, then gradient loss proceeds neuron by neuron proceeds neuron by neuron
        #caling function that would perform gradient of loss calculation neuron by neuron
        delta = gradLoss(der_prev,z,activ_functions_list) 
    #returning the output
    return delta
#this function caluclates gradient of loss neuron by neuron (needed for case when there are different activation functions for neurons in the same layer)
def gradLoss(derPrev,z_output,activ_functions_list):
    #to create properly sized output variable (pre-allocation) information of input(z_output) shape is needed
    shapeInput = derPrev.shape
    #output variable is declared for pre-allocation
    delta = np.empty(shapeInput)
    #iterating through neurons as each need to be calculated separetely
    for iNeuron in range(shapeInput[0]):
        #selecting input for current neuron (row vector for current neuron)
        der_row = derPrev[iNeuron,:]
        z_row = z_output[iNeuron,:]
        #getting derivative of activation function for current neuron
        activ_function = activ_functions_list[iNeuron]
        activ_function_der = getDer(activ_function)
        #getting derivatives values for current neuron
        a_der = activ_function_der(z_row)
        #calculating delta of current neuron for inputs
        delta_neur = der_row * a_der
        #assigning gradient of loss to output variable for currently processed neuron
        delta[iNeuron,:] = delta_neur
    #returning the output
    return delta
#this function gets derivative of activation function based on activation function
der_map = {
    identity: der_identity,
    sigmoid: der_sigmoid,
    tanh: der_tanh,
    relu: der_relu,
    leaky_relu: der_leaky_relu,
    softmax: der_softmax,
    softmax_vec: der_softmax_vec
}#mapping all derivatives to their activation functions
#function itself
def getDer(func):
    #it needs ti be checked wheter derivative for given activation function is implemented and raise proper error if not. This is quite important part of error handling of the project as previously it was only possible to determine if given input is function (callable), now it is checked if it is actually one of implemented activation functions 
    if func in der_map:
        #if pair exist, then derivative can be returned
        return der_map[func]
    else:#if there is no derivative then proper error needs to be raised
        raise NotImplementedError(f"Derivative not implemented for {func.__name__}")
#this function clip gradients to avoid issue of exploding gradients
def clip_gradient(grad, threshold=1.0):
    norm = np.linalg.norm(grad)
    if norm > threshold:
        grad = grad * (threshold / norm)
    return grad
#this function test given network with given input transformed for different possible input data formats. Essentially output for all input data formats should be the same and this is verified.
def testInputFormat(base_list,network):
    #inputs
    inputTestList = base_list
    inputTestVect1D = np.array(base_list)
    inputTestVect2Drow = inputTestVect1D[np.newaxis,:] 
    inputTestVect2Dcol = inputTestVect2Drow.T
    inputTestArray = np.concatenate((inputTestVect2Dcol,inputTestVect2Dcol), axis = 1) 
    #forward pass
    o1 = network.forward(inputTestList)
    o2 = network.forward(inputTestVect1D)
    o3 = network.forward(inputTestVect2Drow)
    o4 = network.forward(inputTestVect2Dcol)
    o5 = network.forward(inputTestArray)
    if(type(o1[0]) == list):#test network
        #testlayer/neuron
        if((o1[-1][1].all() == o2[-1][1].all()) & (o1[-1][1].all() == o3[-1][1].all()) & (o1[-1][1].all() == o4[-1][1].all()) & (o1[-1][1].all() == o5[-1][1].all())):
            return True
        else:
            return False
    else:
        #testlayer/neuron
        if((o1[1].all() == o2[1].all()) & (o1[1].all() == o3[1].all()) & (o1[1].all() == o4[1].all()) & (o1[1].all() == o5[1].all())):#in case of ANN only output of last layer is compared
            return True
        else:
            return False
#this function one hot encodes classes in given file
def one_hot_encode(labels, num_classes):
    #one hot encoding work differently when classes are integers or strings(descriptive)
    if((labels.dtype == object)):#strings in np would be of dtype object
        #getting unique labels for encoding 
        unique = np.unique(labels)
        #first each label needs to be mapped to number(integer)
        maping = {des:val for val,des in enumerate(unique)}
        #getting shape of labels for further processing (it is used more than once, so it is bettere to assign it to variable)
        shapeLabels = labels.shape
        #declaring output variable for pre-allocation
        one_hot = np.zeros((num_classes,shapeLabels[1]))
        #loop iterating through each observation
        for iExample in range(shapeLabels[1]):
            #getting value(string) of current example
            value = labels[0,iExample]
            #getting value(of label) to know to which row (each row is different class) current example should be assigned
            index = maping[value]
            #assignment of class
            one_hot[index,iExample] = 1
    else:#if not object, then some kind of number, which can be integer
        #converting datatype of given classes to integers
        labels_int = labels.astype(int)
        #one hot encoding data
        one_hot = np.eye(num_classes)[labels_int].T #transpose so that shape is (num_classes, n_samples)
    #returning results
    return one_hot
#this function decodes one hot encoded input (with shape (num_classes, n_samples))
def one_hot_decode(one_hot):
    #input is decoded
    decoded = np.argmax(one_hot, axis = 0)#gets index of maximal value of rows in column. As highest value in rows corresponds to class with highest probability its index would be taken, which would give this specific class unique value for it
    #results are returned
    return decoded