import numpy as np
from layers import Layer


class Conv2D(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        """
        Create a convolutional layer using the given parameters.
        Each filter has the same number of channels as the input and
        the number of filters is equal to the number of output channels.
        If requested, a bias is added to each output unit, with one bias
        value per output channel. Parameters are stored in the `param`
        dictionary with keys `weights` and `bias`, respectively.

        Inputs:
            - in_channels: Number of input channels.
            - out_channels: Number of output channels.
            - kernel_size: Size of filter kernel which is assumed to be square.
            - padding: Number of zeros added to borders of input.
            - stride: Step size for the filter.

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
        Convolve the inputs with the stored weights and add a bias
        if requested. The method should work for an arbitrary number
        of input and output channels and allow the use of padding
        and non-unit stride.

        Inputs:
            - inputs: Array with shape (num_samples, in_channels, in_height, in_width)

        Returns:
            - outputs: Array with shape (num_samples, out_channels, out_height, out_width)

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
