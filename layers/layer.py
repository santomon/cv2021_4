class Layer:
    
    def __init__(self):
        """
        Base class for network layers.
        Creates dictionaries for parameters and gradients.
        """
        self.param = {}
        self.grad = {}

    def forward(self, inputs):
        """
        Every layer should implement a method for the forward pass.
        Raises an exception if implementation is missing.
        """
        raise NotImplementedError('Subclass should implement forward method!')

    def backward(self, out_grad):
        """
        Every layer should implement a method for the backward pass.
        Raises an exception if implementation is missing.
        """
        raise NotImplementedError('Subclass should implement backward method!')
