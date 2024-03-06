import numpy as np
import time
from Sparse import SparseMatrix
from LazyMatrix_File import LazyMatrix, LinearOperator


class BitwiseGate(LazyMatrix):

    '''
        Just plagiarising for now, will have to actually sit down and have a proper ol'
        think about how this thing works with the bitwise manipulation. Also how this all 
        ties into the Apply method for the LazyMatrix.
        At this point it appears to work for single qubit operations, but if multiple qubits are 
        input as QuBitPositions array, it will just apply the first one...?
    '''

    def __init__(self,QuBitPositions, gateDimension, registerDimension, gateElements = [(0,0,0)]): 
        self.QBpos = QuBitPositions  # positions in register of qubits to operate on..? 
        self.SquareMat = SparseMatrix(gateDimension, gateElements) # Matrix representing the gate to apply..?
        self.SMdim = gateDimension # Then the SMdim is the dimension of the gate and Dimension inherited from 
                                                # LazyMatrix is dimension of QuRegister..?
        self.Dimension = registerDimension


    def Gather(self, i):
        j = 0
        for k in range(len(self.QBpos)):
            j = j | ((i >> self.QBpos[k]) & 1) << k
        return j


    def Scatter(self, j):
        i = 0
        for k in range(len(self.QBpos)): 
            i = i | ((j >> k) & 1) << self.QBpos[k]
        return i
    

    def Apply(self, vector):
        w = np.zeros(self.Dimension, dtype = complex)
        for i in range(self.Dimension):
            row = self.Gather(i)
            i_0 = i & ~self.Scatter(row)
            for col in range(self.SMdim):
                j = i_0 | self.Scatter(col)
                w[i] += self.SquareMat[row, col] * vector[j]
                # print(i, col)

        return w         

