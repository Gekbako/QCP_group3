import numpy as np

from Gate_File import Gate
from Q_Register_File import *
from LazyMatrix_File import LazyMatrix
from Tensor import TensorProduct

class GroverCircuit(object):
    """
    Class representing a Grover Circuit

    Must contain: 
        - The type of computation
        - The number of Qubits
        - The "Target State" of the oracle
    """ 

    def __init__(self, matrixType, numberOfQubits, targetState):
        """
        This constructor initialises our register and some important variables for the circuit

        Input:
        ------
        numberOfQubits [int] - the number of qubits to tensor together to form the register 
        matrixType [string] - the type of matrix implemented in gates. "Dense", "Sparse" or "Lazy" 
        targetState [int] - the target of the oracle. (this is a number between 0 and 2^numberOfQubits - 1 inclusive)

        """
        self.Register = Q_Register(numberOfQubits)
        self.Dimension = 2**numberOfQubits 
        self.NumberOfQubits = numberOfQubits
        self.MatrixType = matrixType
        self.targetState = targetState

        '''
        Unused Visualization tracking
        
        self.GateArray = [ []*n for n in range(numberOfQubits)] # array to keep track of applied gates for visualisation
        
        if matrixType == "Dense":
            self.CircuitChain = DenseMatrix(np.identity(self.Dimension))
        elif matrixType == "Sparse":
            self.CircuitChain = SparseMatrix(np.identity(self.Dimension)) 
        elif matrixType == "Lazy":
            self.CircuitChain = LazyMatrix(np.identity(self.Dimension))
        '''
    
    def getHadamards(self):
        """
        Function to initialise the matrix for applying a hadamard to every quBit
        
        Input
        ------
        self
        """
        
        hadamardList = []
        
        for i in range(self.NumberOfQubits):
            if self.MatrixType != "Lazy":
                hadamardList.append(Gate(self.MatrixType, gateName = "hadamard").GateMatrix)
            else:
                hadamardList.append(Gate("Sparse", gateName = "hadamard").GateMatrix)      
            
        if self.MatrixType == "Dense":
            TensoredHadamards = TensorProduct(hadamardList).denseTensorProduct
            
            TensoredH = TensoredHadamards().denseTensorProduct()
            
        elif self.MatrixType == "Sparse" or self.MatrixType == "Lazy":
            TensoredHadamards = TensorProduct(hadamardList)
            
            TensoredH = TensoredHadamards.sparseTensorProduct()
        
        if self.MatrixType != "Lazy":
            self.TensoredH = TensoredH
        else:
            self.TensoredH = LazyMatrix(self.Dimension, TensoredH.SparseApply)

    def getOracle(self):
        """
        Function to create the oracle matrix/operator
        
        Input
        ------
        self
        
        Returns
        ------
        oracleMatrix [Dense/Sparse/Lazy Matrix] - The oracle matrix, which flips a target state
        """
        
        if self.MatrixType == "Dense":
            oracleMatrix = np.eye(self.Dimension)
            oracleMatrix[self.targetState][self.targetState] = -1
            
            oracleMatrix = DenseMatrix(oracleMatrix)
            
        elif self.MatrixType == "Sparse":
            
            oracleMatrixElements = []
            
            for i in range(self.Dimension):
                if i == self.targetState:
                    oracleMatrixElements.append([i,i,-1])
                else:
                    oracleMatrixElements.append([i,i,1])
                
                oracleMatrix = SparseMatrix(self.Dimension, oracleMatrixElements)
                
        elif self.MatrixType == "Lazy":
            
            oracleMatrixElements = []
            
            for i in range(self.Dimension):
                if i == self.targetState:
                    oracleMatrixElements.append([i,i,-1])
                else:
                    oracleMatrixElements.append([i,i,1])
                
                oracleSparse = SparseMatrix(self.Dimension, oracleMatrixElements)
                
                oracleMatrix = LazyMatrix(self.Dimension, oracleSparse.SparseApply)
                
        return oracleMatrix

    def getFullyMixedState(self):
        """
        Function to create the fully mixed state.
        
        Input
        ------
        self
        
        Returns
        ------
        fullyMixedState [vector] - The fully mixed state (which has equal probabilities of being measured in any state)
        """
        
        Register = Q_Register(self.NumberOfQubits)
        
        if self.MatrixType == "Dense":
            fullyMixedState = self.TensoredH.DenseApply(Register.state.inputArray)
        else:
            fullyMixedState = self.TensoredH.SparseApply(Register.state.inputArray)
        
        return fullyMixedState

    def getGrover(self):
        """
        Function to create the grover matrix/operator
        
        Input
        ------
        self
        
        Returns
        ------
        groverMatrix [list of Dense/Sparse/Lazy Matrix] - The grover matrix, which inverts qubit values around the mean
            Note:
                groverMatrix is returned as a list of matrices, so we can avoid performing expensive matrix matrix multiplication.
                We use that grover is given by H diag(1,-1,-1,...,-1) H with H being hadamards applied to every quBit.
        """
        
        groverState = self.getFullyMixedState()
        
        if self.MatrixType == "Dense":
            groverTemp = -np.eye(self.Dimension)
            
            groverTemp[0][0] = 1
                
            groverMatrix = [self.TensoredH, DenseMatrix(groverTemp), self.TensoredH]
                
        elif self.MatrixType == "Sparse":
            #groverTemp is the diag(1,-1,-1,-1,...,-1,-1) matrix
            groverTemp = []
                
            for i in range(self.Dimension):
                groverTemp.append([i,i,-1])
                
            groverTemp[0] = [0,0,1] 
            
            groverMatrix = [self.TensoredH, SparseMatrix(self.Dimension, groverTemp), self.TensoredH]
            
        elif self.MatrixType == "Lazy":
            groverTemp = []
            
            hadamardList = []
        
            for i in range(self.NumberOfQubits):
                hadamardList.append(Gate("Sparse", gateName = "hadamard").GateMatrix)
                
            for i in range(self.Dimension):
                groverTemp.append([i,i,-1])
                
            groverTemp[0] = [0,0,1] 
            
            groverTemp = SparseMatrix(self.Dimension, groverTemp)
            
            groverLazy = LazyMatrix(self.Dimension, groverTemp.SparseApply)
                
            TensoredH = TensorProduct(hadamardList)
            
            TensoredHadamards = TensoredH.sparseTensorProduct()
            
            LazyHadamards = LazyMatrix(self.Dimension, TensoredHadamards.SparseApply)
            
            groverMatrix = LazyHadamards.multiply(groverLazy)
            
            groverMatrix = groverMatrix.multiply(LazyHadamards)
            
        return groverMatrix
            
    def iterateGrover(self):
        """
        Function to perform one grover iteration
        
        Input
        ------
        self
        """
            
        if self.MatrixType == "Dense":
                
            #Apply the Oracle
            self.Register.state = self.oracle.DenseApply(self.Register.state)
                
            #Apply the Grover
            self.Register.state = self.grover.DenseApply(self.Register.state)
                
        elif self.MatrixType == "Sparse":
            
            print("State before Oracle")
            print(self.Register.state)
            
            #Apply the Oracle
            self.Register.state = self.oracle.SparseApply(self.Register.state)  
            
            print("State after Oracle")
            print(self.Register.state)
            
            #Apply the Grover
            self.Register.state = self.grover[0].SparseApply(self.Register.state)
            self.Register.state = self.grover[1].SparseApply(self.Register.state)
            self.Register.state = self.grover[2].SparseApply(self.Register.state)
            
            print("State after Grover")
            print(self.Register.state)
            
            

    def runGrover(self):
        """
        Function to run the full grover circuit
        
        Input
        ------
        self
        
        Returns
        ------
        finalState [vector] - The register state after applying the grover iteration ~ pi/4 * root(2^numOfQuBits) times
        measuredState [vector] - The state collapsed to upon measurement
        result [int] - The number represented by measuredState. This is most likely to equal targetState
        """
        
        #Initialise the register and the oracle/grover operators.
        self.getHadamards()
        
        if self.MatrixType != "Lazy":
            self.Register.state = self.getFullyMixedState()
        else:
            self.fullGrover = LazyMatrix(self.Dimension, self.TensoredH.SparseApply)
            
        self.grover = self.getGrover()
        self.oracle = self.getOracle()
        
        numIterations = int(np.pi / 4 * np.sqrt(self.Dimension))
        
        if self.MatrixType != "Lazy":
            for i in range(numIterations):
                self.iterateGrover()
        else:
            for i in range(numIterations):
                self.fullGrover = self.oracle.multiply(self.fullGrover)
                self.fullGrover = self.grover.multiply(self.fullGrover)
                 
            self.fullGrover.Apply(self.Register.state)
            
        #Extract the result
        finalState = self.Register.state
        result = self.Register.measure()
        measuredState = self.Register.state
        
        return finalState, measuredState, result

#Demonstration of the circuit
circuit = GroverCircuit("Sparse", 8, 200)
state, measuredState, result = circuit.runGrover()

print(state)
print(measuredState)
print(result)

