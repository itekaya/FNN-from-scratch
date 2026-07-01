import numpy as np
#this is .py file with weight initialization function


#weight initialization, where all weights are initialized as 0
def zeroIni(array_shape,datatype_weights):
    #initializing weight array
    array_ini = np.zeros(array_shape,dtype=datatype_weights)
    #passing output
    return array_ini
#weight initialization, where weights are random numbers from uniform distribution
def randomIniUniform(array_shape,datatype_weights, lower_bound= -1.0, upper_bound= 1.0):
    #initializing weight array
    array_ini = np.random.uniform(lower_bound,upper_bound,array_shape)
    #changing datatype to desired (previous uni function does not have property to set datatype so it has to be done in separate line)
    array_ini_dtype = np.asarray(array_ini,dtype=datatype_weights)
    #passing output
    return array_ini_dtype
#weight initialization, where weights are random numbers from normal distribution
def randomIniNormal(array_shape,datatype_weights, mean = 0.0, std = 1.0):
    #initializing weight array
    array_ini = np.random.normal(mean,std,array_shape)
    #changing datatype to desired (previous uni function does not have property to set datatype so it has to be done in separate line)
    array_ini_dtype = np.asarray(array_ini,dtype=datatype_weights)
    #passing output
    return array_ini_dtype
#weight initialization, where weights are initialized by Xavier/Glorot method for uniform distribution
def xavIniUniform(array_shape,datatype_weights):
    #information is separated into variables for code readability
    fan_in = array_shape[1]
    fan_out = array_shape[0]
    #getting the limit according to the method
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    #initializing weight array
    array_ini = np.random.uniform(-limit,limit,array_shape)
    #changing datatype to desired (previous uni function does not have property to set datatype so it has to be done in separate line)
    array_ini_dtype = np.asarray(array_ini,dtype=datatype_weights)
    #passing output
    return array_ini_dtype
#weight initialization, where weights are initialized by Xavier/Glorot method for normal distribution
def xavIniNormal(array_shape,datatype_weights):
    #information is separated into variables for code readability
    fan_in = array_shape[1]
    fan_out = array_shape[0]
    #getting std value according to method
    std = np.sqrt(2.0 / (fan_in + fan_out))
    #initializing weight array
    array_ini = np.random.normal(0,std,array_shape)#in this method mean is set to 0
    #changing datatype to desired (previous uni function does not have property to set datatype so it has to be done in separate line)
    array_ini_dtype = np.asarray(array_ini,dtype=datatype_weights)
    #passing output
    return array_ini_dtype
#weight initialization, where weights are initialized by He method for uniform distribution
def heIniUniform(array_shape,datatype_weights):
    #information is separated into variables for code readability
    fan_in = array_shape[1]
    fan_out = array_shape[0]
    #getting the limits according to the method
    lower_limit = (-(np.sqrt(6.0 / fan_in)))
    upper_limit = np.sqrt(6.0 / fan_out)
    #initializing weight array
    array_ini = np.random.uniform(lower_limit,upper_limit,array_shape)
    #changing datatype to desired (previous uni function does not have property to set datatype so it has to be done in separate line)
    array_ini_dtype = np.asarray(array_ini,dtype=datatype_weights)
    #passing output
    return array_ini_dtype
#weight initialization, where weights are initialized by He method for normal distribution
def heIniNormal(array_shape,datatype_weights):
    #information is separated into variables for code readability
    fan_in = array_shape[1]
    #getting std value according to method
    std = np.sqrt(2.0 / fan_in)
    #initializing weight array
    array_ini = np.random.normal(0,std,array_shape)#in this method mean is set to 0
    #changing datatype to desired (previous uni function does not have property to set datatype so it has to be done in separate line)
    array_ini_dtype = np.asarray(array_ini,dtype=datatype_weights)
    #passing output
    return array_ini_dtype