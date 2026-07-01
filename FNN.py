import numpy as np
from Layer import *
from InitFunctions import  *
from SuppFunctions import  *
from ErrorClasses import *
from ActivFunctions import softmax, softmax_vec


class FNN:
#instance attributes
#self.weights_list - this variable holds list of arrays representing the weight values of layer neurons for each layer in the network. Weights are represented by 2d array where rows represent individual neurons and columns represent wieghts of neurons. The 0th column index is assumed to represent bias.
#self.activ_functions_list_list - this variable holds list of "list of activation function of neurons in layer" for all layers.

#constructor
    def __init__(self,weights_info,activ_functions_info,method_ini = "Zero", datatype_weights = "float64", random_lower_bound = -1.0, random_upper_bound = 1.0, random_mean = 0.0, random_std = 1.0):
        #if given weight information is list of neuron numbers in each dimension than it needs to be converted into np array (for code uniformity).
        if(type(weights_info) == list):#it first need to be checked if given input is list
            #to initialize weights some information needs to be given. If given list is empty (so it has no information) proper error should be thrown.
            if(len(weights_info) == 0):
                raise NotSupportedInputGiven("weights initialization","Given list is empty")
            #only integers can represent number of nurons in each layer. For this reason it needs to be verified if that's the case. If not proper error should be thrown
            if(all(isinstance(weight, int) for weight in weights_info)):#now it is checked if it is list of integer values (not np arrays like with other initialization method)
                try:
                    weights_info = np.array(weights_info)#compared to Neuron class, the list can only represent number of neurons (integers), so it has to be in format for dimensions not weights (floats). (dtype = datatype_weights is missing here)
                except Exception as error_caught:#if value error is caught, then no-standard error message should be given for clarity.
                    if(isinstance(error_caught,ValueError)):
                        raise NotSupportedInputGiven("weights initialization","Values given in list are not integer numbers and thus can not be used as neuron number.")
                    #if other error than value error is caight, then it should thrown with its message.
                    raise error_caught
        #weights assignment
        if(type(weights_info) == np.ndarray):#based on input types the weight assignment would process differently, so it is split by ifelse construct
            if(weights_info.ndim == 1):#if vector is given then it is assumed that it is for weight initialization by number of neurons in each layer information
                if(np.issubdtype(weights_info.dtype, np.integer)):#numbers in vector needs to be integers to represnt dimensions of layer and neurons in it
                    if(weights_info.size > 1):#you need at least 2 layers counting input to create smallest network by initialization. The reason for that is how each layer connects, i.e. the number of neurons in previous layer needs to match number of weights in next, so to create weight array for first hidden layer, the input layer infromation is needed (despite input layer not being represented in weight list in this implementation)
                        nWeights = weights_info[0]#first number is assumed to be for number of neurons in input layers
                        nLayers = (len(weights_info) - 1)#the number of layers to be initialized is smaller by 1. It is so, because input layer is not represented by weights in this implementation and thus it doesn't have to be initialized
                        #list to hold 2D weight arrays of layers is declared for proper assignment in loop.
                        weights_layers = []
                        #to initialize network, all layers (except input) needs to be initialized separately to ensure that it is done properly.
                        for iLayer in range(nLayers):
                            #getting number of neurons in current layer
                            nNeurons = weights_info[iLayer+1]#as [0] layer corresponds to inpout layer, the information of hidden layers starts from [1], so the information retrieval is shifted by 1
                            #weights are initializaed by basic method
                            shape_ini = [nNeurons,(nWeights+1)]#bias is accounted for by adding 1 to number of weights.
                            #for each subsequent layer, the number of weights depends on number of neurons in previous layer, so its assignment needs to be adjusted
                            nWeights = nNeurons#assigned value is used in next iteration. Of course it is unused for output layer as there is no next layer. So, this assignment is redundant for last iteration, but it is cheaper to leave as it is, than to create if statment.
                            #weight array is initialized according to choosen method
                            match method_ini:
                                case "Zero":
                                    weights_ini = zeroIni(shape_ini,datatype_weights)
                                case "RandomUni":
                                    weights_ini = randomIniUniform(shape_ini,datatype_weights, lower_bound= random_lower_bound, upper_bound= random_upper_bound)
                                case "RandomNor":
                                    weights_ini = randomIniNormal(shape_ini,datatype_weights, mean = random_mean, std = random_std)
                                case "XavUni":
                                    weights_ini = xavIniUniform(shape_ini,datatype_weights)
                                case "XavNor":
                                    weights_ini = xavIniNormal(shape_ini,datatype_weights)
                                case "HeUni":
                                    weights_ini = heIniUniform(shape_ini,datatype_weights)
                                case "HeNor":
                                    weights_ini = heIniNormal(shape_ini,datatype_weights)
                            #after initialization by chosen method, the layer is added to the list.
                            weights_layers.append(weights_ini)
                        #after initialization weights are assigned to object property
                        self.weights_list= weights_layers
                    else:
                        raise NotSupportedInputGiven("weights initialization","At least two numbers are needed for FNN creation: FNN needs at least input and output layer (single layer network).")
                else:#if in given vector the numbers are not integers, then they cannot be interpreted as number of neurons, so weight creation is impossible.
                    raise NotSupportedInputGiven("weights initialization","Given values in vector must be integers to properly represent number of neurons in layers")
            else:#array of incompatible size was given. Processing is not possible, so proper error should be thrown. 
                raise NotSupportedArrayDimGiven("1")
        elif(type(weights_info) == list):#in this case network is initialized by taking all weight values from given information.
            #first it is verified if in all elements of list there are arrays representing layers
            if(all(isinstance(weight, np.ndarray) for weight in weights_info)):#all elements of the list must be np arrays to represent weight arrays of each layer
                #list to hold 2D weight arrays of layers is declared for proper assignment in loop.
                weights_layers = []
                #to initialize network weight layers, all layers needs to be initialize separately to ensure that it is done properly (verification if dimensions match so that input propagation is possible).
                for iLayer in range(len(weights_info)):
                    #weight array is assigned to variable for code readability
                    weights_array = weights_info[iLayer]#contrary to previous initialization method, here there is no need to include information about input layer as number of weights is already known (although it doesn't mean thath it is correct, so it is verified in next steps). 
                    if(iLayer>0):#for all, but first layer, it needs to be checked if given weight arrays allow the input propagation through network. If not, then proper error should be thrown
                        #it is verified if number of neurons for previous layer matches number of weights for current layer
                        if(weights_array.shape[1] == (weights_layers[-1].shape[0] + 1)):#previous layer would always be last element in list. Number of rows shape[0] corresponds to number of neurons (+1 is added as accounting for bias term) and columns shape[1] corresponds to number of weights. There is no need to check empty scenario specifically here, as with first layer case, because there is no possibility at this point of zero weight shape in previous layer (first layer would throw error and possibility of proceeding (becoming previous layer) with zero weight shape is discarded here with shape comparision(can not be true if there is no possibility of 0 in previous layer)).
                            #if everything is correct, then array is added as next element of the list
                            weights_layers.append(weights_array)
                        else:#if there is missmatch which makes input propagation impossible proper error should be thrown
                            raise NotSupportedInputGiven("weights initialization",f"There is missmatch between number of neurons in layer {iLayer -1} and number of weights in layer {iLayer}")
                    elif(iLayer == 0):#the only plausible leftover case is when there is only one layer, so there is no need for verification
                        #to discard empty weight array scenario (zero weigth shape) for whole weight list, it needs to be checked if such situation does not occur in the first layer (more explanation in case above).
                        if((weights_array.shape[0] > 0) | (weights_array.shape[1] > 0)):
                            #if everything is correct, then array is added as next element of the list (in this case it is always first element)
                            weights_layers.append(weights_array)
                        else:
                            raise NotSupportedInputGiven("weights initialization","Weight array of first layer is empty.")
                    else:#there should be no option of this error occuring if program runs correctly, but if some problem connected with running (not code itself) occurs, then some unexpected outcome might occur. For this reason error with proper information and placing should be thrown.
                        raise NotSupportedInputGiven("weights initialization","Unexpected error. Some corruption occured.")
                #after all verification (if there was something wrong with the code the error would be raised) given list of weight array can be assigned as object property.
                self.weights_list = weights_layers
            else:
                raise NotSupportedInputGiven("weights initialization","Not all elements in list are np arrays, so they can not represent weight arrays of layer.")
        else:#input that is not compatible was given. Operation cannot proceed, so proper error should be thrown.
            raise NotSupportedInputGiven("weights initialization","Not supported data type given.")
        #activation function assignment 
        if (callable(activ_functions_info)):#if only single activation function is given, then it needs to be converted into single element list for code unification.
            #given value creates one element list with itself
            activ_functions_info = [activ_functions_info]
        if(type(activ_functions_info) == list):#activation function list initialization always starts from list. 
            #if there are only 2 elements some special case initialization can occur. So, such case must be verified and proceed properly.
            if(len(activ_functions_info) == 2):
                #if all elements of the list are lists (assumed to be lists of activation layers for 2 layer network), then standard initialization should be done.
                if(all(isinstance(activ_function_list, list) for activ_function_list in activ_functions_info)):
                    #setting proper value for flag variable
                    standardImplement = True
                elif(all(callable(activ_function) for activ_function in activ_functions_info)):#if two elements are functions (assumed to be activation functions), then special initialization occurs, where first activation function is used for all hidden layers (and neurons in them), but output layer, which uses second activation function
                    #setting proper value for flag variable (it is only done for principle as false values for this var are never used/compared)
                    standardImplement = False
                    #list to hold all lists of layer activation function is declared to hold initialization results for each layer and then be passed as whole.
                    activ_functions_list_list = []
                    #initialization goes through all layers, but last(output layer)
                    for iLayer in range(len(self.weights_list) - 1):
                        activ_functions_base = [activ_functions_info[0]] * self.weights_list[iLayer].shape[0]
                        activ_functions_list_list.append(activ_functions_base)
                    #activation function is assigned for last layer.
                    activ_functions_base = [activ_functions_info[1]] * self.weights_list[-1].shape[0]#as it is last layer last element of self.weights_list is taken 
                    activ_functions_list_list.append(activ_functions_base)
                    #ready activ function list is passed to instance atrribute
                    self.activ_functions_list_list = activ_functions_list_list
                else:#if none of the above occurs, then proper error should be thrown as incompatible input was given
                    NotSupportedInputGiven("activation functions initialization","Not supported data type given in list.")
            else:
                #setting proper value for flag variable
                standardImplement = True
            #if special case initialization did not occur than standard (assumed to be the most commonly used) would proceed
            if(standardImplement == True):
                #based on input varaible properties initialization would proceed differently
                if((len(activ_functions_info) == 1 )&(callable(activ_functions_info[0]))):#if given list has only one element and it is function, than it is assumed that the same activation function should be used by all neruons in the network
                    #list to hold all lists of layer activation function is declared to hold initialization results for each layer and then be passed as whole.
                    activ_functions_list_list = []
                    #initialization goes through all layers
                    for iLayer in range(len(self.weights_list)):
                        activ_functions_base = activ_functions_info * self.weights_list[iLayer].shape[0]
                        activ_functions_list_list.append(activ_functions_base)
                    #ready activ function list is passed to instance atrribute
                    self.activ_functions_list_list = activ_functions_list_list
                elif((len(activ_functions_info)) == (len(self.weights_list))):#it first needs to be verified if there is enough information for each layer
                    #based on informationt ype in given list, initialization proceeds differently, so it is verified how exactly it should proceed.
                    if(all(isinstance(activ_function_list, list) for activ_function_list in activ_functions_info)):#if all elements of the given list are lists (to be verified if lists of functions at later step) than initialization by assignment can proceed
                        #list to hold all lists of layer activation function is declared to hold initialization results for each layer and then be passed as whole.
                        activ_functions_list_list = []
                        #assignment goes through all layers
                        for iLayer in range(len(activ_functions_info)):
                            #gettign current activation function list into variable for code readability
                            activ_functions_list = activ_functions_info[iLayer]
                            #it needs to be verified if all elements of activation function list are activation functions and if number of activation function matches number of neurons in layer 
                            if((self.weights_list[iLayer].shape[0] == len(activ_functions_list)) & (all(callable(activ_function) for activ_function in activ_functions_info[iLayer]))):
                                #if everything is as it should be, then assignment proceeds
                                activ_functions_list_list.append(activ_functions_list)
                            else:#if given input does not fit criteria, then proper error should be thrown.
                                raise NotSupportedInputGiven("activation functions initialization","There is missmatch between number of neurons and activation functions in layer {iLayer} or one(or more) given activation function is not a function")
                        self.activ_functions_list_list = activ_functions_list_list
                    elif(all(callable(activ_function) for activ_function in activ_functions_info)):#if all elements of list are functions (assumed to be activation functions) than assigment by initialization (same activ function for all neurons in layer) proceeds
                        #list to hold all lists of layer activation function is declared to hold initialization results for each layer and then be passed as whole.
                        activ_functions_list_list = []
                        #initialization goes through all layers
                        for iLayer in range(len(self.weights_list)):
                            activ_functions_base = [activ_functions_info[iLayer]] * self.weights_list[iLayer].shape[0]
                            activ_functions_list_list.append(activ_functions_base)
                        #ready activ function list is passed to instance atrribute
                        self.activ_functions_list_list = activ_functions_list_list
                    else:
                        raise NotSupportedInputGiven("activation functions initialization","Not supported data type given in list.")
                else:
                    raise NotSupportedInputGiven("activation functions initialization","Not enough or too much information was given to proceed with initialization")
        else:
            raise NotSupportedInputGiven("activation functions initialization","Not supported data type given.")
#methods
    #simple input processing by network
    def forward(self,input):
        #getting instance attributes to separate variables for readability
        weights_list = self.weights_list
        activ_functions_list_list = self.activ_functions_list_list
        #conversion of given input if it is list instead of np array (not always possible as numbers are needed)
        if(type(input) == list):
            try:
                input = np.asanyarray(input, dtype = weights_list[0].dtype)#input should be in same format as weights to stay consistent with data types in calculations.
            except Exception as error_caught:
                if(isinstance(error_caught,ValueError)):
                    raise NotSupportedInputGiven("input propagation","Values given in list are not numbers and thus can not be used as input to neuron.")
                else:
                    raise error_caught
        #input needs to be in form of np array and that needs to be verified. 
        if(type(input) == np.ndarray):
            #input to Layer needs to be a rational number for the operations to be completed. Thus it needs to be verified.
            if(isRationalNumber(input)):
                #depending on the array dimensionality the input can be interpreted differently. Due to this it needs to be checked and passed to proper method of handling.
                number_dimensions = input.ndim
                if(number_dimensions == 1):#given input vector needs to be represented as a column vector in 2D array format for matrix operations. Proper transformations are done below to do so.
                    #the input needs to be represent as a 2D array with only one column(column vector) for matrix operations usage (in case of vector).
                    input_format = input[:,np.newaxis]
                elif(number_dimensions == 2):#given array might be not only array, but also row and column vector. If only vector is given, than it has to be in column vector form. To ensure proper form some verification and if necessery processing is done
                    input_format = getProperInputArray(input)
                else:#if data is of dimensions that handling is not implement proper error should be thrown. 
                    raise NotSupportedArrayDimGiven("1,2")
            else:#if given data is not proper, the error should be thrown.
                raise NotSupportedInputGiven("input propagation","Given values are not a rational numbers and thus can not be used to get output from the network.")
        else:#if given data is not in proper format, the error should be thrown. TO DO: proper error.
            raise NotSupportedInputGiven("input propagation","Not supported data type given.")
        #matix multiplications results and activation results needs to be saved for output. For this purpose variables with lists are created
        z_all = [input_format]#in this implementation a first value of z(matrix multi results) would be input values (can be interpreted as representation of input layer, which is not represented by any network attribute in this implementation) as for input layer it is input.
        a_all = [input_format]#in this implementation a first value of a(activation results) would be input values (can be interpreted as representation of input layer, which is not represented by any network attribute in this implementation) as for input layer it is identity value.
        #input needs to propagate through all layers in network and it is achieved by looping through them.
        for iLayer in range(len(weights_list)):
            #variables necessery for current layer propagations are assigned to local variables for readability
            weights_array = weights_list[iLayer]
            activ_functions_list = activ_functions_list_list[iLayer]
            input = a_all[-1]
            #bias input is added to input vector/array to account for bias, which is 0 weight
            input_ready = addBiasInput(input)
            #it needs to be verified if matrix multiplication can be performed. Theoretically it should be strictly necessery only for first input as rest should be accounted by architecture, but it is present for all if some undetected mistake was made in architecture creation.
            if(weights_array.shape[1] == input_ready.shape[0]):
                #input is multiplied by weights for forward pass (matrix multiplication)
                z = weights_array @ input_ready
                #in this implementatation neurons in layer can have different activation function. Activation is calculated differently when layer has the same activation function for all neurons and when they are different. For this reason the case should be distinguished and process differently.
                if(all((activ_function.__code__.co_code == activ_functions_list[0].__code__.co_code) for activ_function in activ_functions_list)):#it is distinguished if all neurons in layer has the same activation function. In this implementation for same activ function in layer there should be no problem (as all activ functions are copies of each other) with normal comparision, but if it is done differently (or purpose or not) than intended than it might cause problems. For this reason codes are compared just to be sure.
                    #softmax function is a special case, as it can be only used with vector inputs, so all activation functions needs to be the same in the layer for it to function (makes no sense when calculated neuron by neuron as in different activ function in layer case). So, in practice function need to have different names for both cases (if for some reason user wants to use softmax in second case).
                    if(activ_functions_list[0].__code__.co_code == softmax.__code__.co_code):#as activ_functions_list[0] and softmax would be different object, their comparision would yield FALSE. To get proper result codes needs to be compared as they would be the same for same function, but different objects.
                        activ_function = softmax_vec
                    else:#everything except softmax doesn't need different function version for vector and scalar input
                        activ_function = activ_functions_list[0]
                    #layer activation is done (whole layer)
                    a = activ_function(z)
                else:#if neurons have different activation function in layer, then activation proceeds neuron by neuron
                    #layer activation is done (neuron by neuron)
                    a = activationLayer(z,activ_functions_list)
                #calculated values are passed to lists storing them
                z_all.append(z)
                a_all.append(a)
            else:#if inproper input was given and operation cannot proceed proper error should be raised.
                raise NotSupportedInputGiven("input propagation","Input does not match network, i.e. matrix multiplication cannot be done due to mismatch of dimensions.")
        #results are joined together into list for proper output
        output = [z_all,a_all]
        #results are returned
        return output
    #backward propagation of error through the network
    def backward(self, z_values, a_values, targets, loss_derivative):
        #getting properties of FNN object into variables for code readability
        weights_list = self.weights_list
        activ_functions_list_list = self.activ_functions_list_list
        #getting number of layers to know how long loop should go
        num_layers = len(weights_list)
        #creating hold variable for weight gradients
        grad_W = []
        #start of backpropagation from output layer by getting its parameters
        a_output  = a_values[-1]
        z_output = z_values[-1]
        activ_functions_list = activ_functions_list_list[-1]
        #getting derivative of loss for output layer
        if(a_output.shape == targets.shape):
            #softmax + cross entropy might be calculated in special way (it should improve stability and effectivity of training). If such method is choosen ("SoftmaxCrossEntropyDerivative"), then loss proceeds in special way
            if(((activ_functions_list[0].__code__.co_code == softmax.__code__.co_code)) &  (loss_derivative.__name__ == "SoftmaxCrossEntropyDerivative")):
                delta = loss_derivative(targets,a_output)
                #optimizers divide by batch size once again, so here we rescale to avoid effective 1/B^2 
                batch_size = a_output.shape[1]
                delta = delta * batch_size
            else:#in any other case loss propagation proceeds normally
                #calculation of loss derivative
                dL_dy = loss_derivative(targets,a_output)#it should be always possible to calculate loss by vectors
                #calculation of gradient of the loss for output layer
                delta = getDelta(dL_dy,z_output,activ_functions_list) 
        else:#if shapes does not match, than given data is not correct, backpropagation can not proceed and proper error should be thrown.
            raise NotSupportedInputGiven("backpropagation","Network output and ground truth does not match")
        #propagating loss through network (hidden layers)
        for layer_index in reversed(range(num_layers)):
            #getting input to current layer
            a_in = a_values[layer_index]#due to starting in reverse from range the index of current layer is shifted as +1 (it would be normal if input to network was not added as first elements of z and a), so to reach previous layer 'layer_index' is used, not -1 as previous +1 is accounted for and they cancel each other
            #adjusting to bias
            a_in_adj = addBiasInput(a_in)
            #calculating gradient of the loss with the respect to the weights
            dW = delta @ a_in_adj.T#a_in_adj is transposed for result to have shape of weight array -> as dW should have
            #adding current layer dW to the output list 
            grad_W.append(dW)
            #in the case of last layer (first in network) there is no layer to backpropagate to, so next step is unneeded for it
            if(layer_index>0):
                #to proceed weight array of current network is needed
                weight_array = weights_list[layer_index]#this is a different case than 'a_in' as 'layer_index' corresponds to layer in weights_list, so there is no adjustment by definition
                weight_array_nobias = weight_array[:,1:]
                #additionally to proceed with backpropagation, activation functions of previous layer are needed
                activ_functions_list_prev = activ_functions_list_list[layer_index-1]#same as with 'weight_array', indexes matches, so to get parameter from previous layer -1 is applied in indexing
                #pre-activation
                z_prev = z_values[layer_index]#same case as with 'a_in'
                #first previous delta needs to be multiplied by weight matrix
                delta_prot = weight_array_nobias.T @ delta 
                #gradient of loss to the pre-activ calculation
                delta_prev = getDelta(delta_prot,z_prev,activ_functions_list_prev) 
                #new delta was calculated,so it can be assigned to variable, so it can be used in next iteration
                delta = delta_prev
        #reversing dW list for proper representation(last layer as last element)
        grad_W_out = grad_W[::-1]
        #returning output of backpropagation
        return grad_W_out
    #FNN deconstruction into Layer class objects
    def decomposeIntoLayers(self):
        #getting properties of FNN object into variables for code readability
        weights_list = self.weights_list
        activ_functions_list_list = self.activ_functions_list_list
        #getting number of layers to know how long loop should go
        num_layers = len(weights_list)
        #creating list variable to append results of each loop
        layers_list = []
        #iterating through layers of FNN to decompose each into Layer object
        for iLayer in range(num_layers):
            #getting current layer properties
            weights_array = weights_list[iLayer]
            activ_functions_list = activ_functions_list_list[iLayer]
            #creating new Layer object
            layer = Layer(weights_array,activ_functions_list)
            #adding results of current loop to the list
            layers_list.append(layer)
        #returning list with results
        return layers_list
    #FNN deconstruction into lists of Neuron class objects
    def decomposeIntoNeurons(self):
        #decomposing FNN first into Layer objects
        layers_list = self.decomposeIntoLayers()
        #getting number of layer objects to know how long loop should go
        num_layers = len(layers_list)
        #creating list variable to append results of each loop
        neurons_list_list = []
        #iterating through Layer objects to decompose each into Neuron objects
        for iLayer in range(num_layers):
            #getting current Layer object
            layer = layers_list[iLayer]
            #decomposing Layer object into Neuron objects list
            neurons_list = layer.decomposeIntoNeurons()
            #adding results of current loop to the list
            neurons_list_list.append(neurons_list)
        #returning list with results
        return neurons_list_list
    #this functions gets prediction of input by network (for regression network)
    def predictRegression(self,input):
        #propagating input through network
        out = self.forward(input)
        #getting activation values of forward pass
        activation = out[-1]
        #getting activation values of output layer
        output = activation[-1]
        #assigning results of output layer as predictions (as value was predicted)
        predict = output
        #returning predictions
        return predict
    #this functions gets prediction of input by network (for binary classification networks)
    def predictClassBinary(self,input):
        #propagating input through network
        out = self.forward(input)
        #getting activation values of forward pass
        activation = out[-1]
        #getting activation values of output layer
        output = activation[-1]
        #creating output variable for pre-allocation
        predict = np.zeros(output.shape)
        #if probability of results is over 0.5, than it belongs to True
        predict[output >= 0.5] = 1
        #returning predictions
        return predict
    #this functions gets prediction of input by network (for multi-class classification networks)
    def predictClassMulti(self,input):
        #propagating input through network
        out = self.forward(input)
        #getting activation values of forward pass
        activation = out[-1]
        #getting activation values of output layer
        output = activation[-1]
        #class with highest probability becomes prediction. Class values is given based on index.
        predict = np.argmax(output, axis = 0)
        #returning predictions
        return predict