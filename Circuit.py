import numpy as np

from Gate_File import *
from Q_Register_File import *
from BitwiseApply import BitwiseGate


class Circuit(object):
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

    def __init__(self, matrixType, numberOfQubits):
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

        self.TensorArray = np.zeros(numberOfQubits, dtype = Gate)


        if matrixType == "dense":
            self.CircuitChain = DenseMatrix(np.identity(self.Dimension))
        elif matrixType == "sparse":
            self.CircuitChain = DenseMatrix(np.identity(self.Dimension)).Sparse()
        elif matrixType == "lazy":
            #???
            pass


         # maybe this needs to be initialised to 1 (identity) such 
                                                                # that we can multiply gates into it from the start? 
        
                                  

    def AddGate(self, gate : str, qubit):
        '''
        Function to add Gate to Circuit without use of apply method from Q_Register. CNOT and CV are applied 
        directly to the register by use of the apply_gate from Q_Register. 
        Also keeps track of which gates have been added to circuit for visualisation. 
        '''

        addedGate = Gate(self.MatrixType, gate)


        if gate == "cNot" or gate == "cV":
            # check special case of 2-qubit gates, these are applied straight to register due to need for Swap-matrices.
            self.Register = self.Register.apply_gate(addedGate, qubit)
            
            self.GateArray[qubit[0]].append(gate + "-control")
            self.GateArray[qubit[1]].append(gate + "-target")

        else: 
            if isinstance(qubit, int):
                self.TensorArray[qubit] = addedGate
                self.GateArray[qubit].append(gate)
            else:
                for Qbit in qubit:

                    self.TensorArray[Qbit] = addedGate
                    self.GateArray[Qbit].append(gate)



    def AddLayer(self):
        '''
        Function to apply the tensor product gate constructed by a series of AddGate to quantum register. 
        Also saves the tensor it applies as self.SavedTensor, to allow for use outside of program.
        '''
        
        identityGate = Gate(self.MatrixType, "custom", np.array([[1, 0], [0, 1]]))

        self.TensorArray = np.where(self.TensorArray == 0, identityGate, self.TensorArray) 
        # Now all elements are Gates, TensorProduct needs Dense-/Sparse matrices
        
        # if self.MatrixType == "Sparse":
        #     tensorArray = np.zeros(self.NumberOfQubits, dtype = SparseMatrix)
        # elif self.MatrixType == "Dense":
        #     tensorArray = np.zeros(self.NumberOfQubits, dtype = DenseMatrix)

        tensorList = []

        for i in range(self.NumberOfQubits):
            # tensorArray[i] = self.TensorArray[i].GateMatrix
            tensorList.insert(0, (self.TensorArray[i].GateMatrix))
        
        if self.MatrixType == "Sparse":
            tensor = TensorProduct(tensorList).sparseTensorProduct()
            nuReg = tensor.SparseApply(self.Register.state)
            self.Register.state = nuReg
            self.SavedTensor = tensor


        elif self.MatrixType == "Dense":
            tensor = TensorProduct(tensorList).denseTensorProduct()
            nuReg = tensor.DenseApply(self.Register.state)
            self.Register.state = nuReg
            self.SavedTensor = tensor


        
        self.TensorArray = np.zeros(self.NumberOfQubits, dtype = Gate) # Clear TensorArray to be ready for new layer
        # self.visualiseCircuit()
        


    def visualiseCircuit(self):
        '''
        Function to make graphic visualisation of what's going on. Most useful to make sense of what is applied where 
        in the circuit chain. 
        No idea how to implement.
        Function should be run each time a gate is added to show the updated circuit? 
        '''
        
        for i in range(len(self.GateArray)):
            gateString = f"|0>_{len(self.GateArray) - 1 - i} --"
            for element in self.GateArray[len(self.GateArray) - 1 - i]:
                gateString += f"-- {element} --"
            print(f"{gateString}-- END\n")


    def runCircuit(self):
        '''
        Function to run the circuit. Returns either endstate or result of measurement? Or both? 
        '''
        assert self.TensorArray.any() == 0, "There are still gates that haven't been applied. \nUse AddLayer() to apply them, then proceed."

        measurements = self.Register.measure() 

        return measurements





