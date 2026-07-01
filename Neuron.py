#
import numpy as np
from InitFunctions import  *
from SuppFunctions import  *
from ErrorClasses import *
#
class Neuron:
    #instance attributes
    #self.weight_vector - this variable holds vector representing the weight values of neuron. It is represented by 2d array with one row (to represent neuron as part of 1 neuron layer), so by row vector. The 0th index is assumed to represent bias.
    #self.activ_function - this variable holds activation function of neuron.
    #constructor
    def __init__(self,weights_info,activ_function_info, method_ini = "Zero", datatype_weights = "float64", random_lower_bound = 0.0, random_upper_bound = 1.0, random_mean = 0.0, random_std = 1.0):
        #activation function assignment (activ function is first to assign due to length of processing. It is shorter then for weight. So, if error occurs during it the weight processign would not be done unnecessarly)
        if (callable(activ_function_info)):#if given variable is callable then there is high chance that's the wanted activation function. It is not ideal solution, but it is optimal
            self.activ_function = activ_function_info
        else:#if given variable is not an activation function, then class object can not be initialized due to lack (no activ function) or too much (e.g. vector of activ functions) of information. 
            raise NotSupportedInputGiven("activation function initialization","Given variable is not a function and thus can not be used as activation function.")
        #conversion of given weight input if it is list instead of np array (not always possible as numbers are needed)
        if(type(weights_info) == list):
            try:
                weights_info = np.asanyarray(weights_info, dtype = datatype_weights)
            except Exception as error_caught:
                if(isinstance(error_caught,ValueError)):
                    raise NotSupportedInputGiven("weights initialization","Values given in list are not  numbers and thus can not be used as weight values.")
                else:
                    raise error_caught
        #based on the input (weights) the weight vector of the object would be assigned (or created, initialized). The choice of method is based on the data type of input
        type_weights = type(weights_info)
        #weights assignment
        if (type_weights == int):#this is case when weights values are not given by the user, only dimension(as it is for single neuron, e.g. one row of weights). In this case weights needs to be initialized. it is assumed that given value is without bias.
            #basic version (zero initialization) is always done as a base to ommit need to pass datatype information down to initialization methods, so shape of weight vector is adjusted
            shape_ini = [1,(weights_info+1)]#Given number of weights is assumed to be without bias, so it is added for creation.
            #weights are initialized based on selected method
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
            #after initialization weight are assigned to object property
            self.weights_vector = weights_ini
        elif (type_weights == np.ndarray):#this is the case when already initialized weights are passed for neuron object creation
            #not all data types can represent or can be converted to represent weights. They need to be rational number (at least in this implementation). At this point it is verified if passed data meets that criterium.
            if (isRationalNumber(weights_info)):
                #object weights must be of default dtype or as specified by user. It is checked if that is the case, if not then weights dtype is converted
                weights_dtype = ensureDtypeNpArray(weights_info, datatype_weights)
                #given variable dimensions are checked to identify if it can be passed as object weight vector and by which method.
                number_dimensions = weights_dtype.ndim
                if(number_dimensions == 1):
                    #to represent neuron weight vector as one row of layer weight array, the weight vector is a 2D array with only one row(row vector).
                    weights_vector_row = weights_dtype[np.newaxis,:]
                    #transformation are finished and results can be assigned as object property
                    self.weights_vector = weights_vector_row
                elif(number_dimensions == 2):
                    #for proper representation the weight_vector of neuron needs to be row vector. It is checked if that's the case and changed if not.
                    if(isRowVector(weights_dtype)):
                        self.weights_vector = weights_dtype
                    elif(isColVector(weights_dtype)):
                        self.weights_vector = weights_dtype.T
                    else:#if none of above true, then given variable is array and not vector. Appropriate error must be passed to user. 
                        raise NotSupportedInputGiven("weights initialization","Given variable is not an vector (not column or row vector in 2 dim), so it is unsuitable for neuron weight initialization.")
                else:#if given NumPy array is not vector than it is unappropriate for use and appropriate error should be thrown. 
                    raise NotSupportedInputGiven("weights initialization","Given variable is not an vector (due to more than 2 dim), so it is unsuitable for neuron weight initialization.")
            else:#given variable is of unappropriate datatype. Error should be thrown. 
                raise NotSupportedInputGiven("weights initialization","Given values are not a rational numbers and thus can not be used as weight values.")
        else:#given variable is of type for which object initialization is not implemented. Appropriate error sould be passed. 
            raise NotSupportedInputGiven("weights initialization","Not supported data type given.")
    #simple input processing by through neuron (single layer, single neuron ANN)
    def forward(self,input):
        #gettin instance attributes to separate variables for readability
        weights_vector = self.weights_vector
        activ_function = self.activ_function
        #conversion of given input if it is list instead of np array (not always possible as numbers are needed)
        if(type(input) == list):
            try:
                input = np.asanyarray(input, dtype = weights_vector.dtype)#input should be in same format as weights to stay consistent with data types in calculations.
            except Exception as error_caught:
                if(isinstance(error_caught,ValueError)):
                    raise NotSupportedInputGiven("input propagation","Values given in list are not numbers and thus can not be used as input to neuron.")
                else:
                    raise error_caught
        #input needs to be in form of np array. 
        if(type(input) == np.ndarray):
            #input to Neuron needs to be a rational number for the operations to be completed. Thus it needs to be verified.
            if(isRationalNumber(input)):
                #depending on the array dimensionality the input can be interpreted differently. Due to this it needs to be checked and passed to proper method of handling.
                number_dimensions = input.ndim
                if(number_dimensions == 1):#given input vector needs to be represented as a column vector in 2D array format for matrix operations. Proper transformations are done below to do so.
                    #the input needs to be represent as a 2D array with only one column(column vector) for matrix operations usage.
                    input_format = input[:,np.newaxis]
                elif(number_dimensions == 2):
                    input_format = getProperInputArray(input)
                else:#if data is of dimensions that handling is not implement proper error should be thrown. 
                    raise NotSupportedArrayDimGiven("1,2")
                #bias input is added to input vector
                input_ready = addBiasInput(input_format)
            else:#if given data doesn't have rational numbers data, then error should be thrown as such input cannot be put through neuron. 
                raise NotSupportedInputGiven("input propagation","Given values are not a rational numbers and thus can not be used to get output from neuron.")
        else:#if given data is not in proper format, the error should be thrown.
            raise NotSupportedInputGiven("input propagation","Not supported data type given.")
        #before proceeding with input propagation through neuron it needs to be verified if input is compatible.
        if(weights_vector.shape[1] == input_ready.shape[0]):#if they are compatible for matrix multiplication than operation can proceed
            #input is multiplied by weights for forward pass (matrix multiplication)
            matrix_multi = weights_vector @ input_ready
            #the results of input "passing through" weights needs to be put through activation function
            activation_out = activ_function(matrix_multi)
            #both matrix multiplication results (z) and activation results (a) are send out
            output = [matrix_multi,activation_out]
        else:#if inproper input was given and operation cannot proceed proper error should be raised. 
            raise NotSupportedInputGiven("input propagation","Input does not match network, i.e. matrix multiplication cannot be done due to mismatch of dimensions.")
        #results are returned
        return output
  