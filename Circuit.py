import numpy as np

from Gate_File import Gate
from Q_Register_File import *

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


        if matrixType == "dense":
            self.CircuitChain = DenseMatrix(np.identity(self.Dimension))
        elif matrixType == "sparse":
            self.CircuitChain = SparseMatrix(np.identity(self.Dimension)) 
        elif matrixType == "lazy":
            #???
            pass


         # maybe this needs to be initialised to 1 (identity) such 
                                                                # that we can multiply gates into it from the start? 
        
                                  

    def addGate(self, gateTuple):
        '''
        This function should line up a gate to be applied to a certain qubit at a certain point in the chain. 
        It should also: 
            - Somehow be applied to the correct qubit
                --> I.e. it should be tensored at the correct place. 
            - Somehow be optimised wrt memory and not just be naively tensored with Identities in the other vector spaces
                --> That is, it should - assuming we are applying other gates to other qubits at the same point in the 
                        chain - be tensored to those other gates to save computation. 

       For now assuming gateTuple on form [(gate, qubit acting on), (gate, qubit acting on), ...].                 

        '''



                # This below gets us into trouble in the TensorProduct...
        compositeGate = Gate(self.MatrixType, "custom", np.array([1]))                # Naive implementation:
                                            # this way assumes you add all gates that go at a certain point, at the same time
        # this way also assumes we are talking single qubit gates... Two qubit gates are a hassle. 

        # Two qubit gate implementation is going to depened heavily on how they are implemented in the Gate class, so wait

        qubitNumber = np.array(gateTuple)[:, 1]
        
        
        # initialise an Idenity gate to tensor the gates with to account for unaffected qubits. 
        # Initialise it outside for loop to avoid computational loading. '
        IdentityGate = Gate(self.MatrixType, "identity")

        for n in range(self.NumberOfQubits):
            
            if n in qubitNumber: 
                index = np.where(qubitNumber == str(n))[0][0]  # this ugly as hell
                 
                # tensor together the compositeGate and the gate to be added 
                # OBS: if densematrix, the tensor product now takes just the arrays and return just the arrays
                #        the way to extract the arrays from Gate is not quite implemented yet

                nthGate = Gate(self.MatrixType, gateTuple[n][0])
                compositeGate = TensorProduct([compositeGate.GateMatrix.inputArray, 
                                                    nthGate.GateMatrix.inputArray]).denseTensorProduct()
                                        # add some way to determine whether denseTP or sparseTP
                compositeGate = Gate(self.MatrixType, "custom", compositeGate.inputArray)

                self.GateArray[index].append(gateTuple[n][0])

                continue
            
            print(type(compositeGate.GateMatrix))
            print(type(IdentityGate.GateMatrix))
            #tensor together the compositeGate and Identity gate 
            compositeGate = TensorProduct([compositeGate.GateMatrix.inputArray, 
                                                IdentityGate.GateMatrix.inputArray]).denseTensorProduct()
            compositeGate = Gate(self.MatrixType, "custom", compositeGate.inputArray) # this might prove problematic since 
                                    # the Gate class creates a Sparse Gate through conversion from Dense, meaning if we 
                                    # end up with a huge tensored compositeGate, the whole thing will need to be put on 
                                    # Dense form before being Sparse'd for efficiency...

        self.CircuitChain.Multiply(compositeGate.GateMatrix)
        self.visualiseCircuit()
        

    ###########################################################################################
        # Alternative way to add gates to circuit:

    def AddGate(self, gate : str, qubit):
        '''
        Function to add Gate to Circuit by use of Apply method from Q_Register. 
        Also keeps track of which gates have been added to circuit for visualisation.   
        '''
        addedGate = Gate(self.MatrixType, gate)
        self.Register = self.Register.apply_gate(addedGate, qubit)
        if gate == "cNot" or gate == "cV":
            self.GateArray[qubit[0]].insert(0, gate + "-control")
            self.GateArray[qubit[1]].insert(0, gate + "-target")

        else: self.GateArray[qubit].insert(0, gate)



    def Construct(self):
        '''
        Function to apply the tensor product gate constructed by a series of AddGate to quantum register. 
        '''
        
        
        pass



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

    def runCircuit(self):
        '''
        Function to run the circuit. Returns either endstate or result of measurement? Or both? 
        '''
        pass

