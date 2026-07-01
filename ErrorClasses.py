#this is .py file with errors used in this project


#this is error which would be thrown when given input some has problems
class NotSupportedInputGiven(Exception):
    def __init__(self,arg_to_ini, additional = ""):
        message =  f"Given input is not supported by implementation for {arg_to_ini}. {additional}"
        super().__init__(message)
#this is error which would be thrown when given input has proper data type, but it is of not supported dimensions
class NotSupportedArrayDimGiven(Exception):
    def __init__(self,supported):
        message =  f"Given array is of size not supported by implementation. Supported dimensions: {supported}."
        super().__init__(message)