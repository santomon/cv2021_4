import numpy as np
from layers import Layer


class Linear(Layer):

    def __init__(self, in_features, out_features, bias=True):
        """
        Create linear or affine transformation layer.
        Initializes weights and bias randomly and stores the arrays
        in the `param` dictionary created by the base class, using
        the keys `weights` and `bias`, respectively.

        Inputs:
            - in_features: Dimension of inputs.
            - out_features: Dimension of outputs.
            - bias: Optional hint indicating if bias should be used or not.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################



        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################

    def forward(self, inputs):
        """
        Compute linear or affine transformation of the inputs.
        Stores inputs for computation of the loss gradient with respect
        to the weights in the backward pass.

        Inputs:
            - inputs: Array with shape (num_samples, in_features)

        Returns:
            - outputs: Array with shape (num_samples, out_features)

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
        Compute gradient with respect to parameters and inputs.
        The gradient with respect to the weights and bias is stored
        in the `grad` dictionary created by the base class, with
        the keys `weights` and `bias`, respectively.

        Inputs:
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
