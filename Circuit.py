import numpy as np

from Gate_File import *
from Q_Register_File import *
from BitwiseApply import BitwiseGate


class Circuit(object):
    '''
    Class representing an entire quantum circuit.

    Class variables: 
    Register [Q_Register]   -   a qubit register initialised to |0>, keeping track of the quantum state of the qubits in 
                                the circuit. 
    Dimension [int]         -   dimensionality of the qubit register. 2^{number of qubits}.
    NumberOfQubits [int]    -   the number of qubits in the circuit.
    GateArray [2D list]     -   a list to keep track of which gates have been applied to which qubits in the register.
    MatrixType [str]        -   a string denoting the type of matrix implementation to be used by the circuit.
    TensorArray [np.array]  -   an array to construct composite gates with different gates applied to different qubits 
                                in the register. The heart of the class' functionality. 
    SavedTensor [np.array]  -   array containing the latest composite gate constructed by the AddLayer() function. Helpful
                                when extending class to more specific cases, specifically GroverCircuit. 
    ''' 

    def __init__(self, matrixType, numberOfQubits):
        '''
        Notes:
            This constructor must initialise a Qu Register and a "space to apply gates to". 

        Input:
        ------
        numberOfQubits [int] - the number of qubits to tensor together to form the register 
        matrixType [string] - the type of matrix implemented in gates. "dense", "sparse" or "lazy" 
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

        Input
        ------
        gate [str]     -   the name of the gate to be applied.
        qubit [int]    -   the number of the qubit to apply gate to. 

        Returns
        -------
        Nothing, adds gates of given type to the specific indices in the TensorArray. 
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
        Function to make graphic visualisation of what's going on. Very simple implementation, mostly useful to 
        make sense of what is applied where in the circuit chain. 
        '''
        
        for i in range(len(self.GateArray)):
            gateString = f"|0>_{len(self.GateArray) - 1 - i} --"
            for element in self.GateArray[len(self.GateArray) - 1 - i]:
                gateString += f"-- {element} --"
            print(f"{gateString}-- END\n")


    def runCircuit(self):
        '''
        Function to run the circuit. 
        Returns measurement of the quantum register after having had all gates applied. 
        '''
        assert self.TensorArray.any() == 0, "There are still gates that haven't been applied. \nUse AddLayer() to apply them, then proceed."

        measurements = self.Register.measure() 

        return measurements





