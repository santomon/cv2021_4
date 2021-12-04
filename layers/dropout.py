import numpy as np
from layers import Layer


class Dropout(Layer):

    def __init__(self, p=0.5):
        """
        Creates a dropout layer where units are kept with the
        given probability p and dropped with the probability 1-p,
        preventing co-adaptation of neurons.

        Inputs:
            - p: Probability that neuron is kept.

        """
        super().__init__()

        self.p = p
        self.training = True

    def forward(self, inputs):
        """
        Apply dropout when called during training, otherwise just
        pass the input. Implements inversed dropout, so during
        training, the activations are scaled with 1/p.
        
        Inputs:
            - inputs: Array with shape (num_samples, num_features)

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
        Compute the backward pass through the dropout layer.
        Since the inputs are multiplied elementwise with the dropout
        mask, the local gradient with respect to the input is just
        the random mask scaled by 1/p.
        
        Inputs:
            - out_grad: Gradient with respect to layer outputs.

        Returns:
            - in_grad: Gradient with respect to layer inputs.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad
