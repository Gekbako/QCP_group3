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
    base_states_matrices.append(np.outer(temp, temp))
# print(base_states, base_states_matrices, sep="\n")


def trace_register(register):
    assert (isinstance(register, np.ndarray) and
            len(register) == 2**10), "A register of 10 qubits is neccessary -> register of 7 qubits + 3 qubit control"

    out = np.zeros((2**7, 2**7), dtype=complex)

    for i in range(2**7):
        for j in range(2**7):

            old = register[i*2**3:i*2**3+8, j*2**3:j*2**3+8]

            for t in range(3):
                new = np.zeros((2**(2-t), 2**(2-t)))
                for a in range(2**(2-t)):
                    for b in range(2**(2-t)):
                        new[a, b] = trace_single(old[a*2:a*2+2, b*2:b*2+2])
                old = new.copy()
            out[i, j] = old[0, 0]
    return out


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
