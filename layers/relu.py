import numpy as np
from layers import Layer


class ReLU(Layer):
    
    def forward(self, inputs):
        """
        Apply ReLU activation function to inputs.
        The function is applied elementwise such that positive
        values are left unchanged but negative values are set to
        zero. Thus the output has the same shape as the input.
        The input is stored for gradient computation.

        Inputs:
            - inputs: Array as generated from linear or convolutional layers.

        Returns:
            - outputs: Array with same shape as input.

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
        Compute the gradient with respect to the inputs.
        For each input value, the local gradient is one if the value
        is positive and zero otherwise.

        Input:
            - out_grad: Gradient with respect to layer output.

        Returns:
            - in_grad: Gradient with respect to layer input.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad
        