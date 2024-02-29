import numpy as np

from Gate_File import Gate
from Q_Register_File import *
from Q_Register_File import Q_Register
from Tensor import TensorProduct
"""
class ErrorCorrection(object):

    def __init__(self, matrixType, initialRegister):

        self.register = initialRegister
        self.state = initialRegister.state
        
    def 

"""
Register = Q_Register(2)
Gate_test = Gate("Sparse", "spinX")

Register.apply_gate(Gate_test, [1])
print(Register)