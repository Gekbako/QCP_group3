import numpy as np
from Sparse import SparseMatrix
from Dense import DenseMatrix
from Apply_File import Apply
from Gate_File import Gate
from Q_Register_File import Q_Register
from Tensor import TensorProduct


# initialize a 2 qbit register 
register1 = Q_Register(2)

print(f"Initial qubits are {register1}.")

# create a hadamard gate to act on register1
gate1 = Gate("Dense", "hadamard") 

# # apply the hadamard gate to register1
register1.apply_gate(gate1,[0,1]) 
print(f"Qubits after the gate are {register1}.")

# measure the register
register1.measure()
print(f"Collapsed qubits are {register1}.")