from Sparse import SparseMatrix
from Dense import DenseMatrix
import numpy as np

class TensorProduct(object):
    
    def __init__(self, thingsToBeTensored):
        
        assert isinstance(thingsToBeTensored, list), "The primary input for the tensor product method should be passed as a list."
        
        #Check that we're inputting only vectors or only matrices
        if(all(isinstance(matrix, np.ndarray) for matrix in thingsToBeTensored)):
           self.tensorProduct = self.denseTensorProduct
        elif(all(isinstance(matrix, list) for matrix in thingsToBeTensored)):
           self.tensorProduct = self.sparseTensorProduct
        else:
           raise Exception("The inputs for a tensor product should ALL be numpy arrays or lists.")

        
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
        
        #Initialise the product 
        Product = self.thingsToBeTensored[0]
        
        for productNumber in range(1, len(self.thingsToBeTensored)):
            
            xLengthA = Product.shape[0]
            yLengthA = Product.shape[1]
            
            xLengthB = self.thingsToBeTensored[productNumber].shape[0]
            yLengthB = self.thingsToBeTensored[productNumber].shape[1]
            
            #Following the notation of the docstring, Product is A, and self.thingsToBeTensored[productNumber] is B
            BMatrix = self.thingsToBeTensored[productNumber]
            
            #Find the shape of the product
            newShape = (xLengthA * xLengthB, yLengthA * yLengthB)
            
            newProduct = np.zeros(newShape)
            
            #Loop over entries of the tensor product
            for i in range(newShape[0]):
                for j in range(newShape[1]):
                    
                    n = int(i / xLengthB)
                    m = int(j / yLengthB)
                    
                    p = i % xLengthB
                    q = j % yLengthB
                    
                    newProduct[i][j] = Product[n][m] * BMatrix[p][q]
                    
            
            Product = newProduct
            
        return Product
    
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
        
        #Initialise the product 
        Product = self.thingsToBeTensored[0]
        
        for productNumber in range(1, len(self.thingsToBeTensored)):
            
            BMatrix = self.thingsToBeTensored[productNumber]
            
            newShape = (len(Product[0]) * len(BMatrix[0]), len(Product[1]) * len(BMatrix[1]))
            
            newProduct = []
            
            for elementA in Product:                
                for elementB in BMatrix:
                    
                    i = len(BMatrix[0]) * elementA[0] + elementB[0]
                    j = len(BMatrix[1]) * elementA[1] + elementB[1]
                    
                    val = elementA[2] * elementB[2]
                    
                    newProduct.append((i,j,val))
            
            Product = newProduct
          
        return Product
    
#Test Code
"""
A = [[0,0,1],[0,1,1],[1,0,1],[1,1,-1]]
B = [[0,0,1],[0,1,1],[1,0,1],[1,1,-1]]
C = [[0,0,1],[1,1,1]]

ATensorB = TensorProduct([A,C,B])

print(ATensorB.tensorProduct())
print("------------------")

A = np.array([[1,1],[1,-1]])
B = np.array([[1,1],[1,-1]]) 
C = np.eye(2)

print(np.kron(np.kron(A,C),B))
"""
