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
base_states = []
base_states_matrices = []
for i in range(8):
    temp = np.zeros(8, dtype=complex)
    temp[i] = 1
    base_states.append(temp)
    base_states_matrices.append(DenseMatrix(np.outer(temp, temp)))
# print(base_states, base_states_matrices, sep="\n")


def trace_register(register):
    assert (isinstance(register.inputArray, np.ndarray) and
            len(register.inputArray) == 2**10), "A register of 10 qubits is neccessary -> register of 7 qubits + 3 qubit control"

    out = np.zeros((2**7, 2**7), dtype=complex)

    for i in range(2**7):
        for j in range(2**7):

            old = register.inputArray[i*2**3:i*2**3+8, j*2**3:j*2**3+8]

            for t in range(3):
                new = np.zeros((2**(2-t), 2**(2-t)))
                for a in range(2**(2-t)):
                    for b in range(2**(2-t)):
                        new[a, b] = trace_single(old[a*2:a*2+2, b*2:b*2+2])
                old = new.copy()
            out[i, j] = old[0, 0]
    return DenseMatrix(out)


def trace_single(matrix):
    assert (isinstance(matrix, np.ndarray) and
            len(matrix) == 2), "Need 2*2 matrix that is then traced to 0 or 1"

    if matrix[0, 0] == 0 and matrix[1][1] == 0:
        return 0
    else:
        return matrix[0, 0] + matrix[1][1]


"""
 Register = Q_Register(2)
Gate_test = Gate("Sparse", "spinX")

Register.apply_gate(Gate_test, [1])
print(Register)
"""
"""m = [[0 for ___ in range(2**10)] for __ in range(2**10)]
for _ in range(2**10):
    m[_][_] = 1
input = np.array(m)
output = trace_register(input)"""


Register = Q_Register(10)
XGate = Gate("Sparse", "spinX")
CGate = Gate("Sparse","cNot")

Register.apply_gate(XGate, [1,2,3,4,5]) # Error at position with index 5 (qubit 6)

# State is |0111110000>

# First parity check
paritylist1 = [0,2,4,6] # check qubits in positions 1,3,5,7
for i in paritylist1:
    Register.apply_gate(CGate, [i,9])

# Second parity check
paritylist2 = [1,2,5,6] # check qubits in positions 2,3,6,7
for i in paritylist2:
    Register.apply_gate(CGate, [i,8])

# Third parity check
paritylist3 = [3,4,5,6] # check qubits in positions 4,5,6,7
for i in paritylist3:
    Register.apply_gate(CGate, [i,7])

# State should now be in the state |0111110110>

densitymat = DenseMatrix(np.outer(Register.state,Register.state))
Id = DenseMatrix(np.eye(128))
errorpos = 0

for i in range(8):
    TensorProd = TensorProduct([Id,base_states_matrices[i]]).denseTensorProduct()
    matrix = TensorProd.Multiply(densitymat)
    if np.any(matrix.DenseApply(np.ones(2**10))) == False:
        pass
    else:
        errorpos += i
        break

if errorpos == 0:
    print("There is no error.")
else:
    print("The error is in qubit " + str(errorpos))

Register.apply_gate(XGate, [errorpos-1])
newdensitymat = DenseMatrix(np.outer(Register.state,Register.state))

FinalCleanDensityMat = trace_register(TensorProduct([Id,base_states_matrices[errorpos]]).denseTensorProduct().Multiply(newdensitymat))
FinalCleanStateVec = FinalCleanDensityMat.DenseApply(np.ones(2**7))
print(FinalCleanStateVec)





