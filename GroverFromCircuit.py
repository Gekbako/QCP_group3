from Circuit import *
import time

class GroverCircuit(Circuit):
    '''
    Attempt to understand better and optimise through reworking.
    '''
    def __init__(self, matrixType, numberOfQubits, targetState):
        super().__init__(matrixType, numberOfQubits)
        assert targetState < 2**numberOfQubits, f"Target state {targetState} out of reach for quantum system of {numberOfQubits} qubits. Must have targetState < {2**numberOfQubits}."
        self.TargetState = targetState
        
        if matrixType == "Dense":

            hadamardIndices = np.arange(numberOfQubits) 
            self.AddGate("hadamard", hadamardIndices) 
            self.AddLayer() # prepare quantum register in H^tensor_n state for Grover circuit    
            self.GroverMatrix = -np.identity(self.Dimension)

            self.Oracle = -self.GroverMatrix
            self.Oracle[targetState, targetState] = -1
            self.Oracle = DenseMatrix(self.Oracle)

            self.GroverMatrix[0, 0] = 1
            self.GroverMatrix = DenseMatrix(self.GroverMatrix)

        elif matrixType == "Sparse":

            hadamardIndices = np.arange(numberOfQubits) 
            self.AddGate("hadamard", hadamardIndices) 
            self.AddLayer() # prepare quantum register in H^tensor_n state for Grover circuit 

            indices = np.arange(self.Dimension, dtype = int)
            values = -np.ones(self.Dimension, dtype = int)
            values[0] = 1
            self.GroverMatrix = np.column_stack((indices, indices, values))
            self.GroverMatrix = SparseMatrix(self.Dimension, self.GroverMatrix)

            values[0] = -1
            values = -values 
            values[targetState] = -1
            self.Oracle = np.column_stack((indices, indices, values))
            self.Oracle = SparseMatrix(self.Dimension, self.Oracle)

        elif matrixType == "Lazy":

            self.hadamardGate = Gate("Sparse", "hadamard").GateMatrix 
            
            # create n-dim Hadamard tensor bitwise and lazily 
            hadamardBitwise = BitwiseGate([0], self.hadamardGate.Dimension, (self.Dimension), self.hadamardGate.Elements)
            self.hadamardBitwiseLazy = LazyMatrix(self.Dimension, hadamardBitwise.Apply)
            for i in range(1, self.NumberOfQubits):
                hadamardBitwise = BitwiseGate([i], self.hadamardGate.Dimension, (self.Dimension), self.hadamardGate.Elements)
                self.hadamardBitwiseLazy = self.hadamardBitwiseLazy.multiply(LazyMatrix(self.Dimension, hadamardBitwise.Apply))
            
            # prepare quantum register in H^tensor_n state for Grover circuit
            self.Register.state = self.hadamardBitwiseLazy.Apply(self.Register.state)

            # create Grover matrix to be applied Sparsely
            indices = np.arange(self.Dimension, dtype = int)
            values = -np.ones(self.Dimension, dtype = int)
            values[0] = 1
            self.GroverMatrix = np.column_stack((indices, indices, values))
            self.GroverMatrix = SparseMatrix(self.Dimension, self.GroverMatrix)

            # create Oracle matrix to be applied Sparsely
            values[0] = -1
            values = -values 
            values[targetState] = -1
            self.Oracle = np.column_stack((indices, indices, values))
            self.Oracle = SparseMatrix(self.Dimension, self.Oracle)

            # Lazify the matrices, create composite gate GroverDiffusionOperator 
            self.GroverMatrix = LazyMatrix(self.Dimension, self.GroverMatrix.SparseApply)
            self.Oracle = LazyMatrix(self.Dimension, self.Oracle.SparseApply)
            self.GroverDiffusionOperator = self.Oracle.multiply(self.hadamardBitwiseLazy.multiply(self.GroverMatrix.multiply(self.hadamardBitwiseLazy)))
            


    def groverIteration(self):
        '''
        Function to apply a single Grover diffusion operator to register. 
        '''
        if self.MatrixType == 'Dense':
            #Apply the Oracle
            self.Register.state = self.Oracle.DenseApply(self.Register.state)
                
            #Apply the Grover
            self.Register.state = self.SavedTensor.DenseApply(self.Register.state)
            self.Register.state = self.GroverMatrix.DenseApply(self.Register.state)
            self.Register.state = self.SavedTensor.DenseApply(self.Register.state)

                
        elif self.MatrixType == "Sparse":
            
            #Apply the Oracle
            self.Register.state = self.Oracle.SparseApply(self.Register.state)  

            #Apply the Grover
            self.Register.state = self.SavedTensor.SparseApply(self.Register.state)
            self.Register.state = self.GroverMatrix.SparseApply(self.Register.state)
            self.Register.state = self.SavedTensor.SparseApply(self.Register.state)
 

        
        elif self.MatrixType == "Lazy":

            self.Register.state = self.GroverDiffusionOperator.Apply(self.Register.state)

            
    def run(self):

        numIterations = int(np.pi / 4 * np.sqrt(self.Dimension))
        for i in range(numIterations):
            self.groverIteration()

        finalState = self.Register.state
        result = self.Register.measure()
        measuredState = self.Register.state

        return finalState, result, measuredState


startTime = time.time()

#Demonstration of the circuit
circuit = GroverCircuit("Lazy", 11, 98)
state, result, measuredState = circuit.run()

# print(state)
# print(measuredState)
print(result)

endTime = time.time()
timeElapsed = endTime - startTime
print(f"Time elapsed: {timeElapsed}s")
