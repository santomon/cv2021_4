import numpy as np


class CrossEntropyLoss():

    def __init__(self, model):
        """
        Create a cross-entropy loss for the given model.
        Minimizing the loss is equivalent to minimizing the
        cross-entropy between the normalized scores and
        the correct class labels.

        Inputs:
            - model: Model to compute the loss for.

        """
        self.model = model

    def forward(self, inputs, labels):
        """
        Compute the loss for given inputs and labels.
        Stores the probabilities obtained from applying the
        softmax function to the inputs for computing the
        gradient in the backward pass.

        Inputs:
            - inputs: Scores generated from the model.
            - labels: Array with correct classes

        Returns:
            - loss: Loss averaged over inputs.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        loss = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return loss

    def backward(self):
        """
        Compute gradient with respect to model parameters.
        Uses probabilities stored in the forward pass to compute
        the local gradient with respect to the inputs, then
        backpropagates the gradient through the model.

        Returns:
            - in_grad: Gradient with respect to the inputs.

        """
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        in_grad = None

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################
        return in_grad

    def __call__(self, inputs, labels):
        """
        Make instances callable for convenience.
        Arguments are passed to the forward method of the class.
        """
        return self.forward(inputs, labels)
