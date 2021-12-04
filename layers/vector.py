import numpy as np
from layers import Layer


class Vector(Layer):
    
    def forward(self, inputs):
        """
        Converts tensor inputs into vectors.
        Caches input shape for recovery in the backward pass.

        Inputs:
            -inputs: Array with shape (N, D1, ..., Dk)

        Returns:
            -outputs: Array with shape (N, D)

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        outputs = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return outputs
    
    def backward(self, out_grad):
        """
        Converts vector inputs into tensors.
        Restores the shape of the array received in the forward pass.

        Inputs:
            - out_grad: Array with shape (N, D)
        
        Returns:
            - in_grad: Array with shape (N, D1, ..., Dk)

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad
