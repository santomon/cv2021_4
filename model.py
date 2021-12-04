from layers import Dropout, Layer


class Model:

    def _get_layers(self):
        """
        Create an ordered list of the layers of the model.
        Assumes that layers are just added as attributes in the
        constructor of the subclass. In newer Python versions
        the order should be preserved.

        Returns:
            - layers: List of all layers.

        """
        layers = []

        # All of our layers are subclasses of the Layer class.
        for value in self.__dict__.values():
            if isinstance(value, Layer):
                layers.append(value)

        return layers

    def forward(self, inputs, training=True):
        """
        Evaluate complete network for given inputs.
        Inputs are passed to the first layer and then each
        output is in turn passed to the following layer.
        The last output is the prediction.

        Inputs:
            - inputs: Input for the first layer.
            - training: Flag for training or testing.

        Returns:
            - outputs: Output of the last layer.

        """
        layers = self._get_layers()

        # Compute forward pass through network.
        outputs = inputs        

        for layer in layers:
            
            # We want to keep all neurons active during inference.
            if isinstance(layer, Dropout):
                layer.training = training

            # Compute forward pass through layer.
            outputs = layer.forward(outputs)

        return outputs

    def backward(self, out_grad):
        """
        Compute gradients of the loss with respect to
        the model parameters using backpropagation, that is,
        gradients are computed locally in each layer and
        combined with the gradient received from the
        following layer.

        Inputs:
            - out_grad: Gradient with respect to the model output.

        Returns:
            - in_grad: Gradient with respect to the model input.
        """
        layers = self._get_layers()

        # Compute backward pass through network.
        in_grad = out_grad

        for layer in reversed(layers):
            in_grad = layer.backward(in_grad)

        return in_grad

    def __call__(self, inputs, training=True):
        """
        Make instances callable for convenience.
        Arguments are passed to the forward method of the class.
        """
        return self.forward(inputs, training=True)
