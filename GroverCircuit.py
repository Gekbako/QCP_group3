import numpy as np

from Gate_File import Gate
from Q_Register_File import *
from Tensor import TensorProduct

class GroverCircuit(object):
    '''
    Class representing an entire quantum circuit.

    Must contain: 
        - The full circuit, consisting of: 
            * Gates, tensored together to account for which qubits are being operated on
                and then multiplied together (eiter with matmul or Lazy mult.)  
            * Quantum register being operated on 
            * Measurement options at the end?
        - A function to add gates to certain qubits 
            * This must be optimalised in some way and not just applying one and one gate with
                a ton of identity matrices tensored to them. 
        - Visualisation?
    ''' 

    def __init__(self, matrixType, numberOfQubits, targetState):
        '''
        Notes:
            This constructor must initialise a Qu Register and a "space to apply gates to". 

        Input:
        ------
        numberOfQubits [int] - the number of qubits to tensor together to form the register 
        matrixType [string] - the type of matrix implemented in gates. "dense", "sparse" or "lazy" 
            --> Should matrixType be global for the class or specific to the gates? I.e. should we be able to mix types? 

        '''
        self.Register = Q_Register(numberOfQubits)
        self.Dimension = 2**numberOfQubits 
        self.NumberOfQubits = numberOfQubits
        self.GateArray = [ []*n for n in range(numberOfQubits)] # array to keep track of applied gates for visualisation
        self.MatrixType = matrixType
        self.targetState = targetState


        if matrixType == "dense":
            self.CircuitChain = DenseMatrix(np.identity(self.Dimension))
        elif matrixType == "sparse":
            self.CircuitChain = SparseMatrix(np.identity(self.Dimension)) 
        elif matrixType == "lazy":
            #???
            pass


         # maybe this needs to be initialised to 1 (identity) such 
                                                                # that we can multiply gates into it from the start? 
        

    ###########################################################################################
        # Alternative way to add gates to circuit:
        #I modififed these functions to use construct for creating tensor products (then just gonna use apply)

    def AddGate(self, gate : str, qubit):
        '''
        Function to add Gate to Circuit by use of Apply method from Q_Register. 
        Also keeps track of which gates have been added to circuit for visualisation.   
        '''
        addedGate = Gate(self.MatrixType, gate)
        self.Register = self.Register.apply_gate(addedGate, qubit)
        
        #Visualization Step
        if gate == "cNot" or gate == "cV":
            self.GateArray[qubit[0]].insert(0, gate + "-control")
            self.GateArray[qubit[1]].insert(0, gate + "-target")

        else: self.GateArray[qubit].insert(0, gate)

    def getOracle(self, targetState):
        """
        Function to create the oracle matrix/operator
        """
        
        if self.MatrixType == "Dense":
            oracleMatrix = np.eye(self.Dimension)
            oracleMatrix[targetState][targetState] = -1
            
            oracleMatrix = DenseMatrix(oracleMatrix)
            
        elif self.MatrixType == "Sparse":
            
            oracleMatrixElements = []
            
            for i in range(self.Dimension):
                if i == targetState:
                    oracleMatrixElements.append([i,i,-1])
                else:
                    oracleMatrixElements.append([i,i,1])
                
                oracleMatrix = SparseMatrix(self.Dimension, oracleMatrixElements)
                
        return oracleMatrix

    def getFullyMixedState(self):
        
        Register = Q_Register(self.NumberOfQubits)
        
        hadamardList = []
        
        for i in range(self.NumberOfQubits):
            hadamardList.append(Gate(self.MatrixType, gateName = "hadamard").GateMatrix)
            
        if self.MatrixType == "Dense":
            TensoredHadamards = TensorProduct(hadamardList).denseTensorProduct
            
            fullyMixedState = TensoredHadamards.DenseApply(Register.state)
            
        elif self.MatrixType == "Sparse":
            TensoredH = TensorProduct(hadamardList)
            
            TensoredHadamards = TensoredH.sparseTensorProduct()
            
            fullyMixedState = TensoredHadamards.SparseApply(Register.state.inputArray)
        
        return fullyMixedState

    def getGrover(self):
        
        groverState = self.getFullyMixedState()
        
        if self.MatrixType == "Dense":
            groverMatrix = -np.eye(self.Dimension)
            
            for i in range(self.Dimension):
                groverMatrix[i][i] += 2 * groverState[i]
                
            groverMatrix = DenseMatrix(groverMatrix)
                
        elif self.MatrixType == "Sparse":
            groverTemp = []
            
            hadamardList = []
        
            for i in range(self.NumberOfQubits):
                hadamardList.append(Gate(self.MatrixType, gateName = "hadamard").GateMatrix)
                
            for i in range(self.Dimension):
                groverTemp.append([i,i,-1])
                
            groverTemp[0] = [0,0,1] 
            
            groverTemp = SparseMatrix(self.Dimension, groverTemp)
                
            TensoredH = TensorProduct(hadamardList)
            
            TensoredHadamards = TensoredH.sparseTensorProduct()
            
            ExtraTemp = TensoredHadamards
            
            groverMatrix = ExtraTemp.Multiply(groverTemp)
            
            groverMatrix = groverMatrix.Multiply(TensoredHadamards)
            
            """
            This implementation is super hacky since I was just trying to check that it works. (Also I've only implemented sparse rn)
            Essentially, I use that the grover operator is given by H diag(1,-1,-1,...,-1) H, with H being the fully tensored hadamards
            """
            
        return groverMatrix
            
    def iterateGrover(self):
            
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
            self.Register.state = self.grover.SparseApply(self.Register.state)
            
            print("State after Grover")
            print(self.Register.state)
            
            

    ###########################################################################################
        # WAIT! THE QU REGISTER HAS THE APPLY METHOD DEFINED...USE THIS

    ###########################################################################################


    def visualiseCircuit(self):
        '''
        Function to make graphic visualisation of what's going on. Most useful to make sense of what is applied where 
        in the circuit chain. 
        No idea how to implement.
        Function should be run each time a gate is added to show the updated circuit? 
        '''
        pass

    def runGrover(self):
        '''
        Function to run the circuit. Returns either endstate or result of measurement? Or both? 
        '''
        
        #Initialise the register and the oracle/grover operators.
        self.Register.state = self.getFullyMixedState()
        self.grover = self.getGrover()
        self.oracle = self.getOracle(self.targetState)
        
        numIterations = int(np.pi / 4 * np.sqrt(self.Dimension))
        
        for i in range(numIterations):
            self.iterateGrover()
            
        finalState = self.Register.state
        result = self.Register.measure()
        measuredState = self.Register.state
        
        return finalState, measuredState, result
    
circuit = GroverCircuit("Sparse", 3, 6)
state, measuredState, result = circuit.runGrover()

print(state)
print(measuredState)
print(result)

