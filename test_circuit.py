import numpy as np
from Sparse import SparseMatrix
from Dense import DenseMatrix
from Apply_File import Apply
from Gate_File import Gate
from Q_Register_File import Q_Register
from Tensor import TensorProduct


"""# initialize a 2 qbit register 
register1 = Q_Register(3)
"""
gateH = Gate("Sparse", "hadamard")
gate1 = Gate("Sparse", "cNot") 
gate2 = Gate("Sparse", "spinX")
gate3 = Gate("Sparse", "spinZ")

register1 = Q_Register(10)
indices = [0,840,816,120,720,408,480,680]
register1.state[0] = 0
for i in indices:
    register1.state[i] = (1/np.sqrt(8))

# # apply the gates to register1
register1.apply_gate(gate3, [0])
print(register1.state)
register1.apply_gate(gateH, [0,1,2,3,4,5,6])

print(f"Qubits after the gates are {register1.state}.")

"""matrix = DenseMatrix(np.outer(register1.state,register1.state))
matrix2 = TensorProduct([DenseMatrix(np.eye(4)),DenseMatrix(np.outer([0,1],[0,1]))]).denseTensorProduct()

print(matrix2.Multiply(matrix))
all_zeros = not np.any(matrix2.Multiply(matrix).inputArray)
print(all_zeros)
"""