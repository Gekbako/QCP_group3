import numpy as np
import time
from Sparse import SparseMatrix
from LazyMatrix_File import LazyMatrix, LinearOperator


class BitwiseGate(LazyMatrix):
    '''
    Class implementing a bitwise manipulation approach to quantum gate application. Inherits from LazyMatrix class.
    Class variables: 
        QBpos [array]               -   the positions of qubits the gate acts on. Right now this only works for inital 
                                        element in array, i.e. applying a single gate to a single qubit. 
        GateMatrix [SparseMatrix]   -   a sparse matrix representing the gate to be applied.
        GateDim [int]               -   dimension of the gate to be applied. Required as it is represented by a Sparse Matrix. 
        Dimension [int]             -   dimension of the Q_Register the gate is applied to.
    '''

    def __init__(self,QuBitPositions, gateDimension, registerDimension, gateElements = [(0,0,1), (1,1,1)]): 
        self.QBpos = QuBitPositions  
        self.GateMatrix = SparseMatrix(gateDimension, gateElements) 
        self.GateDim = gateDimension            
        self.Dimension = registerDimension


    def Gather(self, i):
        '''
        Function to identify and extract active qubits in register that are being acted upon by the gate. 
        '''
        j = 0
        for k in range(len(self.QBpos)):
            j = j | ((i >> self.QBpos[k]) & 1) << k
        return j


    def Scatter(self, j):
        '''
        Function to place given bits back to correct place in qubit register.
        '''
        i = 0
        for k in range(len(self.QBpos)): 
            i = i | ((j >> k) & 1) << self.QBpos[k]
        return i
    

    def Apply(self, register):
        '''
        Function to apply gate to register through bitwise manipulation methods, 
            making use of Gather() and Scatter() functions to apply gate only to the affected qubits in the register.

        Input
        ------
        register [cx. array]    -   the state vector of the qubit register to apply gate to.

        Returns
        -------
        newReg [cx. array]      -   the state vector of the qubit register after application.

        '''
        newReg = np.zeros(self.Dimension, dtype = complex)
        for i in range(self.Dimension):
            row = self.Gather(i)
            i_0 = i & ~self.Scatter(row)
            
            for col in range(self.GateDim):
                j = i_0 | self.Scatter(col)
                newReg[i] += self.GateMatrix[row, col] * register[j]

        return newReg         

