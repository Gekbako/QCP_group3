import numpy as np
from Apply_File import Apply
from Gate_File import Gate, SwapMatrix1a
from Tensor import TensorProduct
from Dense import DenseMatrix
from Sparse import SparseMatrix
H_gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])


class Qubit:
    def __init__(self, state=np.array([1, 0], dtype=complex)) -> None:
        self.state = np.array(self.normalize(state), dtype=complex)

    def normalize(self, vec_to_norm):
        """
        Normalizes a complex vector to magnitude 1
        Args:
        vec_to_norm (1D array) : 2*1 vector holding the linear coefficient of "unormalized" state
        """
        assert (isinstance(vec_to_norm, np.ndarray) and len(
            vec_to_norm) == 2), "Supply complex vector (2*1 numpy array)"
        factor = np.sqrt((vec_to_norm*np.conj(vec_to_norm)).sum())
        return vec_to_norm / factor

    def apply(self, gate):
        """
        Applies a gate to a qubit -> performs inner profuct on the 
        matrix and vector representation of the state of the qubit
        Args:
        gate (2D array) : square matrix repesentig a quantum gate (lin operator)
        """
        temp = Apply([gate, self.state])
        self.state = temp.DenseApply()

    def measure(self):
        """
        A qubit is of form |state> = a|+> + b|->, so when measured,
        based on its amplitudes (a,b) it will collapse either to |+> or |->
        """
        P = np.abs(self.state) ** 2
        pos = np.random.choice([0, 1], p=P)
        self.state[pos] = 1
        self.state[pos-1] = 0

    def __str__(self) -> str:
        out = f"state = | {self.state[0]}|+> + {self.state[1]}|-> >"
        return out


class Q_Register:
    def __init__(self, n: int, states=None) -> None:
        """
        Initializes the Q_Register with n |0> state Qubits
        Args:
        n (int) : number of qubits
        """
        self.n = n
        self.state = np.zeros(2**n, dtype=complex)
        temp = []
        # TODO: is it a problem that the quibits are normalized individually when they are initialized?
        if np.all(states) == None:

            self.state[0] = 1
            self.state = DenseMatrix(self.state).inputArray
        else:
            to_tens_prod = []
            for i in range(n):
                temp.append(Qubit(states[2*i: 2*(i+1)]))
                to_tens_prod.append(DenseMatrix(temp[i].state))
            self.state = np.squeeze(TensorProduct(
                to_tens_prod).denseTensorProduct().inputArray)

    def apply_gate(self, gate: Gate, index):
        """
        Applies gate to qubit/qubits in index and return the new state of the Q_Register
        Arg:
        gate : matrix representation of the gate
        index (list) : qubit index that the gate should be applied to (NOTE If gate is cNot/cV, first index in list is control, second is target)
        Returns:
        The register with the modified state
        """
        # TODO: we assume the gate is compatible with the register
        # -> qRegState is of size 2**n * 1 and gate 2**n * 2**n
        QubitNum = self.n
        State = self.state
        TensorList = []
        if gate.gateName != "cNot" and gate.gateName != "cV":
            if gate.matrixType == "Sparse":
                Identity = SparseMatrix(2, [[0, 0, 1], [1, 1, 1]])
                for i in range(QubitNum):
                    TensorList.append(Identity)
                for num in index:
                    TensorList[num] = gate.GateMatrix
                TensorGate = TensorProduct(TensorList).sparseTensorProduct()
                NewState = TensorGate.SparseApply(State)
                self.state = NewState
                return NewState
            elif gate.matrixType == "Dense":
                Identity = DenseMatrix(np.array([[1, 0], [0, 1]]))
                for i in range(QubitNum):
                    TensorList.append(Identity)
                for num in index:
                    TensorList[num] = gate.GateMatrix
                TensorGate = TensorProduct(TensorList).denseTensorProduct()

                NewState = TensorGate.DenseApply(State.inputArray)
                NewState = DenseMatrix(NewState)
                self.state = NewState
                return NewState

            else:  
                pass
        else:
            Control = index[0]
            Target = index[1]
            Identity = SparseMatrix(2, [[0, 0, 1], [1, 1, 1]])
            if Control == 0:
                SwapMatrixControl = DenseMatrix(np.eye(2**QubitNum)).Sparse()
            else:
                SwapMatrixControl = SwapMatrix1a(QubitNum, Control)
            if Target == 1:
                SwapMatrixTarget = DenseMatrix(np.eye(2**QubitNum)).Sparse()
            else:
                SwapMatrixTarget = TensorProduct(
                    [Identity, SwapMatrix1a(QubitNum-1, Target-1)]).sparseTensorProduct()
            SwapMatrixForward = SwapMatrixTarget.Multiply(SwapMatrixControl)
            SwapMatrixBackward = SwapMatrixControl.Multiply(SwapMatrixTarget)
            if gate.matrixType == "Sparse":
                for i in range(QubitNum-1):
                    TensorList.append(Identity)
                TensorList[0] = gate.GateMatrix
                TensorGate = TensorProduct(TensorList).sparseTensorProduct()
                NewState1 = SwapMatrixForward.SparseApply(State)
                NewState2 = TensorGate.SparseApply(NewState1)
                NewState = SwapMatrixBackward.SparseApply(NewState2)
                self.state = NewState
                return NewState
            elif gate.matrixType == "Dense":
                DenseSwapForward = DenseMatrix(SwapMatrixForward.Dense())
                DenseSwapBackward = DenseMatrix(SwapMatrixBackward.Dense())
                Identity = DenseMatrix(np.array([[1, 0], [0, 1]]))
                for i in range(QubitNum-1):
                    TensorList.append(Identity)
                TensorList[0] = gate.GateMatrix
                TensorGate = TensorProduct(TensorList).denseTensorProduct()
                NewState1 = DenseSwapForward.DenseApply(State)
                NewState2 = TensorGate.DenseApply(NewState1)
                NewState = DenseSwapBackward.DenseApply(NewState2)
                self.state = DenseMatrix(NewState)
                return NewState
            else:  
                pass

    def measure(self):
        """
        Measurement of the possibly entagled state of Q_Register,
        according to the amplitudes, leaving the register in a basis state 
        between 0 and (2**n)-1       
        """
        # callculate prob. amplituded
        P = np.array([abs(qb)**2 for qb in self.state])
        # get a basis state bases on the prob. amplitudes
        result = np.random.choice(np.arange(len(self.state)), p=P)

        self.state = self.state*0
        self.state[result] = 1

        return result

    def __str__(self) -> str:
        # prints state of the register

        return str(self.state)


a = np.array([1+1j, 2+2j], dtype=complex)
b = np.array([3+3j, 4+4j], dtype=complex)
# , 1/np.sqrt(2)*np.array([1+0j, 1+0j, 1+0j, 1+0j, 1+0j, 1+0j]))
q = Q_Register(7)


"""
print(q)
q.measure()
print(q)

HGate = Gate("Sparse", "spinX")
q.apply_gate(HGate, [0])

densityMat = np.outer(q.state, q.state)
# print(densityMat)

test1 = DenseMatrix(np.outer(np.array([1, 0]), np.array([1, 0])))
test2 = DenseMatrix(np.outer(np.array([0, 1]), np.array([0, 1])))

test = [test1, test2]
Id = DenseMatrix(np.eye(2))
TProd = TensorProduct([Id, test1]).denseTensorProduct()
"""


"""TestFinal = TProd.inputArray*densityMat*TProd.inputArray
print(TestFinal)"""
