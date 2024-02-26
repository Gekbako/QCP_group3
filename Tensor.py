from Sparse import SparseMatrix
from Dense import DenseMatrix
import numpy as np


class TensorProduct(object):

    def __init__(self, thingsToBeTensored):

        assert isinstance(
            thingsToBeTensored, list), "The primary input for the tensor product method should be passed as a list."

        # Check that we're inputting only vectors or only matrices
        if (all(isinstance(matrix, DenseMatrix) for matrix in thingsToBeTensored)):
            self.tensorProduct = self.denseTensorProduct
        elif (all(isinstance(matrix, SparseMatrix) for matrix in thingsToBeTensored)):
            self.tensorProduct = self.sparseTensorProduct
        else:
            raise Exception(
                "The inputs for a tensor product should ALL be numpy arrays or lists.")

        self.thingsToBeTensored = thingsToBeTensored

    def denseTensorProduct(self):
        """
        Initial naive implementation
        Based on the definition that for 2 matrices, A (n x m) and B (p x q), the tensor product is:

        [[A_11 B, A_12 B, ... , A_1m B],
         [A_21 B, A_22 B, ... , A_2m B],
          ...
         [A_n1 B, A_n2 B, ... , A_nm B]]

        Thus the output is a np x mq matrix.

        The ij'th component of the product is therefore:
        A_kl B_xy
        With k/l = floor(i/j / p/q) and x/y = i/j mod p/q

        Note for vectors, which have shape n x 1, m x 1, the tensor product has shape nm x 1.
        Strictly, for 2 vectors, one has the tensor product of shape n x m.
        The elements relate as the n x m, ij'th entry, is the i*n +j'th element of the nm x 1 vector.
        That is, this is just a representation of the same thing.
        I based this code on the kronecker product, which is essentially a tensor product specialised to matrices.

        Returns
        -------
        Product : The tensor product of the list thingsToBeTensored. 

        Output is of type operator or vector depending what is being tensored.

        """

        # Initialise the product
        Product = np.array(self.thingsToBeTensored[0].inputArray)

        for productNumber in range(1, len(self.thingsToBeTensored)):

            if len(self.thingsToBeTensored[productNumber].inputArray.shape) == 1:
                yLengthA = 1
            else:
                yLengthA = Product.shape[1]

            xLengthA = Product.shape[0]

            xLengthB = self.thingsToBeTensored[productNumber].inputArray.shape[0]
            if len(self.thingsToBeTensored[productNumber].inputArray.shape) == 1:
                yLengthB = 1
            else:
                yLengthB = self.thingsToBeTensored[productNumber].inputArray.shape[1]

            # Following the notation of the docstring, Product is A, and self.thingsToBeTensored[productNumber] is B
            BMatrix = self.thingsToBeTensored[productNumber].inputArray

            # Find the shape of the product
            newShape = (xLengthA * xLengthB, yLengthA * yLengthB)

            newProduct = np.zeros(newShape, dtype=complex)

            # Loop over entries of the tensor product
            for i in range(newShape[0]):
                for j in range(newShape[1]):

                    n = int(i / xLengthB)
                    m = int(j / yLengthB)

                    p = i % xLengthB
                    q = j % yLengthB

                    if newShape[1] == 1:
                        newProduct[i] = Product[n] * BMatrix[p]
                    else:
                        newProduct[i][j] = Product[n][m] * BMatrix[p][q]

            Product = newProduct

        return DenseMatrix(Product)

    def sparseTensorProduct(self):
        """
        Implemenetation using sparse matrices.
        Still based on the kronecker product.

        Assumes sparse matrices are input as lists of tuples [(i,j,M_ij)]

        Returns
        -------
        Product : The tensor product of the list thingsToBeTensored. 

        Output is of type operator or vector depending what is being tensored.


        """

        # Initialise the product
        Product = self.thingsToBeTensored[0].Elements
        ProductDim = self.thingsToBeTensored[0].Dimension

        for productNumber in range(1, len(self.thingsToBeTensored)):

            BMatrix = self.thingsToBeTensored[productNumber]
            BElements = BMatrix.Elements

            newProduct = []

            for elementA in Product:
                for elementB in BElements:

                    i = BMatrix.Dimension * elementA[0] + elementB[0]
                    j = BMatrix.Dimension * elementA[1] + elementB[1]

                    val = elementA[2] * elementB[2]

                    newProduct.append((i, j, val))

            Product = newProduct
            ProductDim = ProductDim * BMatrix.Dimension

        return SparseMatrix(ProductDim, Product)


# Test Code
"""
A = SparseMatrix(2, [[0,0,1],[0,1,1],[1,0,1],[1,1,-1]])
B = SparseMatrix(2, [[0,0,1],[0,1,1],[1,0,1],[1,1,-1]])
C = SparseMatrix(2, [[0,0,1],[1,1,1]])

ATensorB = TensorProduct([A,C,B,C])

print(ATensorB.tensorProduct())
print("------------------")

A = np.array([[1,1],[1,-1]])
B = np.array([[1,1],[1,-1]]) 
C = np.eye(2)

print(np.kron(np.kron(np.kron(A,C),B),C))
"""
"""

A = DenseMatrix(np.sqrt(1/2)*np.array([1,1]))
B = DenseMatrix(np.array([1,0]))
ATensorB = TensorProduct([A,B,B,B])
print(np.squeeze(ATensorB.denseTensorProduct().inputArray))
"""
