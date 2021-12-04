import numpy as np
from layers import Layer


class SGD:
    
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        """
        Create optimizer for gradient descent with momentum.
        Extends standard SGD with momentum, so that in each step
        a weighted average of the past gradients is used for the
        update. Setting momentum to zero is equivalent to use
        standard SGD.

        Inputs:
            - model: Object with layers stored as attributes.
            - learning_rate: Step size to use for parameter updates.
            - momentum: Contribution of past gradients.

        """
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Store only layers with parameters.
        self.layers = {}

        for key, layer in model.__dict__.items():

            # All layers are subclasses of Layer and have a possibly empty param dictionary.
            if isinstance(layer, Layer) and layer.param:
                self.layers[key] = layer

                # Create a new dictionary for velocity if necessary.
                if not hasattr(layer, 'velocity'):
                    layer.velocity = {}

                    # Accumulated gradients still have the same shape as the parameter.
                    for name, array in layer.param.items():
                        layer.velocity[name] = np.zeros_like(array)

    def step(self):
        """
        Perform updates for all parameters of the model.
        Stores the velocity for the current time step, computed from
        the most recent gradient and the gradient history weighted
        with the momentum.
        """
        for layer in self.layers.values():
            for name in layer.param:
                ############################################################
                ###                  START OF YOUR CODE                  ###
                ############################################################

                layer.param[name] = None

                ############################################################
                ###                   END OF YOUR CODE                   ###
                ############################################################
