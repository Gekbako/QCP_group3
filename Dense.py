# written by Simon
import numpy as np
from Sparse import SparseMatrix


class DenseMatrix():

    def __init__(self, inputArray, shape=None):
        """
        Input
        ------
        A numpy array. If input is not a numpy array it is converted and a warning message appears. 
        """
        if not isinstance(inputArray, np.ndarray):
            print(
                f"Warning, had to convert DenseMatrix primary input from a {type(inputArray)} into a numpy array.")
            inputArray = np.array(inputArray)

        self.inputArray = inputArray
        self.shape = np.shape(inputArray)

    def Scale(self, factor):
        """
        Input
        ------
        Scalar as an integer or floating point number. 

        Returns
        -------
        Nothing, modifies the original matrix. 
        """

        assert isinstance(
            factor, (int, float)), "DenseMatrix scale method expects an int or float input scalar."

        self.inputArray = np.asmatrix(self.inputArray) * factor

    def Multiply(self, matrix2):
        """
        Input
        ------
        Another DenseMatrix 

        Returns
        -------
        The product of the two input matrices as a DenseMatrix. 
        """

        assert isinstance(
            matrix2, DenseMatrix), "DenseMatrix multiply method expects DenseMatrix input."

        product = np.asmatrix(self.inputArray) * \
            np.asmatrix(matrix2.inputArray)
        array = np.asarray(product)
        return DenseMatrix(array)

    def DenseApply(self, u):
        """
        Input
        ------
        Vector u

        Implementation for dense matrices is row multiplication over the vector elements, as usual.
        Function assumes the matrix is square, which should be fine as all matrices we use will be square.
        For matrix M applied to vector u, the element v_1 of the new vector is given by: v_1 = M_11u_1+M_12u_2+...+M_1nu_n.

        Returns
        -------
        NewVector : The matrix applied to the target vector u.
        The output will be a vector itself.

        """
        # Start product

        NewVector = []

        # Loop over matrix elements and vector elements to find new vector elements and append them

        for i in self.inputArray:
            NewVectorelem = 0
            for j in range(0, self.shape[1]):
                NewVectorelem += i[j]*u[j]
            NewVector.append(NewVectorelem)
        NewVector = np.asarray(NewVector, dtype=complex)
        return (NewVector)

    def Sparse(self):
        """
        Output
        ------
        The matrix in sparse format
        """
        n = self.shape[0]
        elements = []
        rownum = -1

        for i in self.inputArray:
            rownum += 1
            for j in range(0, n):
                if i[j] != 0:
                    elem = [rownum, j, i[j]]
                    elements.append(elem)
        return (SparseMatrix(n, elements))

    def __getitem__(self, index):
        row, col = index
        return self.inputArray[row, col]
    
    def __setitem__(self, index, value):
        row, col = index
        self.inputArray[row, col] = value
        
    def __str__(self):
        return str(np.asmatrix(self.inputArray))



if __name__ == "__main__":
    a = np.array([[3, 0, 1], [1, 0, 0], [0, 0, 6]])
    b = 2
    c = [[3, 0, 1], [1, 0, 0], [0, 0, 6]]
    d = (3, 1, 2)

    matrixA = DenseMatrix(a)
    matrixA.Scale(b)
    print(matrixA)

    matrixC = DenseMatrix(c)
    product = matrixA.Multiply(matrixC)
    apply = matrixC.DenseApply(d)

    print(product)
    print(apply)
    print(matrixC.Sparse())
    print(matrixC)
    print(matrixC[1, 0])
    matrixC[1,0] = 5
    print(matrixC[1, 0])
