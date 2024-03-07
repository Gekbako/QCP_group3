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
    
def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0: 
       return vector
    return vector / norm

# State preparation
Register = Q_Register(10)
indices = [0,840,816,120,720,408,480,680]
Register.state[0] = 0
for i in indices:
    Register.state[i] = (1/np.sqrt(8))

HGate = Gate("Sparse", "hadamard")
XGate = Gate("Sparse", "spinX")
ZGate = Gate("Sparse", "spinZ")
CGate = Gate("Sparse","cNot")

Register.apply_gate(XGate, [1]) # Bit error at position with index 1 (qubit 2)
Register.apply_gate(ZGate, [5]) # Phase error at position with index 5 (qubit 6)


# Test for bit error first

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

densitymat = DenseMatrix(np.outer(Register.state,Register.state))
Id = DenseMatrix(np.eye(128))
errorpos = 0

for i in range(8):
    TensorProd = TensorProduct([Id,base_states_matrices[i]]).denseTensorProduct()
    matrix = TensorProd.Multiply(densitymat)
    if not np.any(matrix.inputArray) == True:
        pass
    else:
        errorpos += i
        break

if errorpos == 0:
    print("There is no bit error.")
else:
    print("The bit error is in qubit " + str(errorpos))

Register.apply_gate(XGate, [errorpos-1])

if errorpos == 0:
    pass
elif errorpos == 1:
    Register.apply_gate(XGate, [9])
elif errorpos == 2:
    Register.apply_gate(XGate, [8])
elif errorpos == 3:
    Register.apply_gate(XGate, [8,9])
elif errorpos == 4:
    Register.apply_gate(XGate, [7])
elif errorpos == 5:
    Register.apply_gate(XGate, [7,9])
elif errorpos == 6:
    Register.apply_gate(XGate, [7,8])
elif errorpos == 7:
    Register.apply_gate(XGate, [7,8,9])

# Return ancilla qubits to |000>

Register.apply_gate(HGate, [0,1,2,3,4,5,6])

# Change basis of the system to change phase errors into bit flip errors

# First parity check
for i in paritylist1:
    Register.apply_gate(CGate, [i,9])

# Second parity check
for i in paritylist2:
    Register.apply_gate(CGate, [i,8])

# Third parity check
for i in paritylist3:
    Register.apply_gate(CGate, [i,7])

densitymat2 = DenseMatrix(np.outer(Register.state,Register.state))
errorpos2 = 0

for i in range(8):
    TensorProd = TensorProduct([Id,base_states_matrices[i]]).denseTensorProduct()
    matrix2 = TensorProd.Multiply(densitymat2)
    if not np.any(matrix2.inputArray) == True:
        pass
    else:
        errorpos2 += i
        break

if errorpos2 == 0:
    print("There is no phase error.")
else:
    print("The phase error is in qubit " + str(errorpos2))

# Fix error
Register.apply_gate(XGate, [errorpos2-1])
# Return to initial basis
Register.apply_gate(HGate, [0,1,2,3,4,5,6])

newdensitymat = DenseMatrix(np.outer(Register.state,Register.state))

FinalCleanDensityMat = trace_register(TensorProduct([Id,base_states_matrices[errorpos2]]).denseTensorProduct().Multiply(newdensitymat))
FinalCleanStateVec = np.diagonal(FinalCleanDensityMat.inputArray).copy()
for i in range(len(FinalCleanStateVec)):
    if FinalCleanStateVec[i] < 0.0001:
        FinalCleanStateVec[i] = 0
FinalCleanStateVec = normalize(FinalCleanStateVec)

print(FinalCleanStateVec)





