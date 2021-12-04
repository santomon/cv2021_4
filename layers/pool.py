import numpy as np
from layers import Layer


class MaxPool(Layer):
    
    def __init__(self, kernel_size, stride=None):
        """
        Create max pooling layer with given kernel size and stride.
        If no stride is provided, the stride is set to the kernel
        size, such that non-overlapping areas are filtered for
        the maximum.

        Inputs:
            - kernel_size: Size of the pooling region which is assumed to be square.
            - stride: Step size of the filter operation.

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
        Apply max pooling to the given inputs.
        The pooling is applied to each channel separately. For each
        filter position, the maximum activation value is selected.
        The inputs are stored for gradient computation.

        Inputs:
            - inputs: Array with shape (num_samples, num_channels, in_height, in_width)
        
        Returns:
            - outputs: Array with shape (num_samples, num_channels, out_height, out_width)

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
        Compute the gradient with respect to the layer input.
        The gradient of the layer output with respect to the layer
        input is one for the selected maximum values in each filter
        position and zero for all other values.

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
