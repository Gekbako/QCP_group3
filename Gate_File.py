import numpy as np
from Sparse import SparseMatrix
from Dense import DenseMatrix
from Tensor import TensorProduct
from LazyMatrix_File import LazyMatrix


def SwapMatrix1a(dim, a):
    """
    Defines a sparse matrix that swaps qubit in index 0 with qubit in index a in a qunatum register with dim qubits.

    Input
    ------
    dim = Number of qubits in register
    a = Index of qubit to swap with first qubit

    Output
    ------
    SwapMatrix = SparseMatrix that swaps qubit in index a with qubit in index 0
    """
    SwapSmall = SparseMatrix(4, [[0, 0, 1], [1, 2, 1], [2, 1, 1], [3, 3, 1]])
    if dim == 2:
        SwapMatrix = SwapSmall
    else:
        IdentityStart = DenseMatrix(np.eye(2**(dim-2))).Sparse()
        SwapMatrix = TensorProduct(
            [SwapSmall, IdentityStart]).sparseTensorProduct()
        for i in range(1, a):
            Step = TensorProduct([DenseMatrix(np.eye(2**i)).Sparse(), SwapSmall,
                                 DenseMatrix(np.eye(2**(dim-2-i))).Sparse()]).sparseTensorProduct()
            SwapMatrix = SwapMatrix.Multiply(Step)
        for j in range(a-2, -1, -1):
            if j == 0:
                Step = TensorProduct(
                    [SwapSmall, IdentityStart]).sparseTensorProduct()
            else:
                Step = TensorProduct([DenseMatrix(np.eye(2**j)).Sparse(), SwapSmall,
                                     DenseMatrix(np.eye(2**(dim-2-j))).Sparse()]).sparseTensorProduct()
            SwapMatrix = SwapMatrix.Multiply(Step)

    return SwapMatrix


class Gate(object):

    def __init__(self, matrixType, gateName=None, customInput=None):
        # TODO: let's say the matrixTypes are "Sparse", "Dense", ???"Lazy"???
        self.matrixType = matrixType
        self.gateName = gateName
        self.customInput = customInput
        self.GateMatrix = self.gateMethod()

    def gateMethod(self):
        if self.gateName == "hadamard":
            gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        elif self.gateName == "cNot":
            gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                            [0, 0, 0, 1], [0, 0, 1, 0]])
        elif self.gateName == "cV":
            gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1j]])
        elif self.gateName == "spinX":
            gate = np.array([[0, 1], [1, 0]])
        elif self.gateName == "spinY":
            gate = np.array([[0, -1j], [1j, 0]])
        elif self.gateName == "spinZ":
            gate = np.array([[1, 0], [0, -1]])
        elif self.gateName == "custom":
            gate = self.customInput

        if self.matrixType == "Dense":
            return DenseMatrix(gate)
        if self.matrixType == "Sparse":
            return DenseMatrix(gate).Sparse()


"""
Swap = SwapMatrix1a(3, 1)
Swap2 = TensorProduct([SparseMatrix(2, [[0, 0, 1], [1, 1, 1]]),
                      SwapMatrix1a(2, 1)]).sparseTensorProduct()
SwapTogether = Swap2.Multiply(Swap)
u = np.array([0, 0, 0, 1, 0, 0, 0, 0])
v = Swap.SparseApply(u)
Final = Swap2.SparseApply(v)
Final2 = SwapTogether.SparseApply(u)
print(Final)
print(Final2)
"""
