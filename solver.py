import numpy as np
from loss import CrossEntropyLoss
from optim import SGD


class Solver:

    def __init__(self, model, data, **kwargs):
        """
        Create a solver for training the classifier.
        This object encapsulates the training logic and provides
        some default values for the hyperparameters of the model.
        During training the loss and accuracies are stored
        for further analysis.

        Inputs:
            - model: Network to train.
            - data: Dictionary with training and validation
                    sets stored with keys X_train, y_train,
                    X_val and y_val, respectively.
            - learning_rate: Step size for parameter updates.
            - learning_rate_decay: Factor multiplied to the learning
                                   rate after each epoch.
            - momentum: Hyperparameter for optimizer.
            - num_epochs: Number of epochs to train the model.
            - num_train_samples: Number of samples from the training
                                 set used to compute accuracy.
            - num_val_samples: Number of samples from the validation
                               set used to compute accuracy. Default
                               is to use the complete set.
            - print_every: Interval of training iterations to
                           print the current loss.
            - verbose: Indicates if intermediate results
                       should be printed.

        """
        self.model = model

        # Store training and validation data.
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Define default values for parameters.
        defaults = {
            'batch_size': 100,
            'learning_rate': 0.01,
            'learning_rate_decay': 1.0,
            'momentum': 0.9,
            'num_epochs': 10,
            'num_train_samples': 1000,
            'num_val_samples': None,
            'print_every': 10,
            'verbose': False
        }

        for key, value in defaults.items():
            self.__dict__[key] = kwargs.pop(key, value)

        # Some attributes for bookkeeping.
        self.epoch = 0
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def __step(self, loss, optimizer):
        """
        Perform one parameter update on the model.
        Samples a random batch of inputs and labels from the
        training data with replacement, computes the loss for
        the inputs and computes the gradient with respect to
        the parameters.

        Inputs:
            - loss: Function to compute data loss.
            - optimizer: Update rule to use.

        """
        num_train = len(self.y_train)

        # Sample random minibatch with replacement.
        indices = np.random.choice(num_train, self.batch_size)

        inputs = self.X_train[indices]
        labels = self.y_train[indices]

        # Compute forward pass.
        outputs = self.model(inputs)

        # Compute and store loss.
        current_loss = loss(outputs, labels)
        self.loss_history.append(current_loss)

        # Compute backward pass.
        loss.backward()

        # Update model parameters.
        optimizer.step()

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Checks accuracy on the given dataset.
        If the number of samples is given and less than
        the length of the dataset, the set is subsampled
        before evaluation.
        
        Inputs:
            - X: Set of inputs to evaluate.
            - y: Corresponding labels.
            - num_samples: Number of samples used to check accuracy.
            - batch_size: Number of samples per batch.

        Returns:
            - accuracy: Average of correct predictions.

        """
        N = X.shape[0]

        if num_samples is not None and N > num_samples:

            # Subsample dataset if necessary.
            indices = np.random.choice(N, num_samples)

            X = X[indices]
            y = y[indices]

            N = num_samples

        # Compute number of batches for testing.
        num_batches = N // batch_size

        if N % batch_size != 0:
            num_batches += 1
        
        # Create list for predictions.
        y_pred = []
        
        for i in range(num_batches):
            
            # Compute index range for batch.
            start = i * batch_size
            end = (i + 1) * batch_size

            # Evaluate model and store predictions.
            scores = self.model(X[start:end], training=False)
            y_pred.append(np.argmax(scores, axis=1))

        # Compute and return accuracy.
        return np.mean(np.hstack(y_pred) == y)

    def train(self):
        """
        Train the model using the given settings.
        If requested, prints out the current loss in specified
        intervals. Training and validation accuracy are checked
        at the start and end of training and once per epoch,
        using the specified number of samples.
        
        Returns:
            - history: Loss and accuracies from training.

        """
        num_train = len(self.y_train)

        # Compute number of iterations per epoch and total number of iterations.
        num_iter_per_epoch = max(num_train // self.batch_size, 1)
        num_iter = num_iter_per_epoch * self.num_epochs
        
        # Create optimizer and loss function.
        optimizer = SGD(self.model, self.learning_rate, self.momentum)
        loss = CrossEntropyLoss(self.model)

        for i in range(num_iter):

            # Compute one training step with parameter update.
            self.__step(loss, optimizer)

            # Display current loss if requested.
            if self.verbose and i % self.print_every == 0:
                print(f'Iteration: {i+1}/{num_iter} Loss: {self.loss_history[-1]}')

            # Apply learning rate decay and increment epoch counter.
            epoch_end = (i + 1) % num_iter_per_epoch == 0

            if epoch_end:
                optimizer.learning_rate *= self.learning_rate_decay
                self.epoch += 1

            # Check accuracy at first and last iteration and end of epoch.
            first_iter, last_iter = (i == 0), (i == num_iter - 1)

            if first_iter or last_iter or epoch_end:
                
                # Check and store training accuracy.
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=self.num_train_samples)
                self.train_acc_history.append(train_acc)

                # Check and store validation accuracy.
                val_acc = self.check_accuracy(self.X_val, self.y_val, num_samples=self.num_val_samples)
                self.val_acc_history.append(val_acc)

                # Display accuracy per epoch.
                if self.verbose:
                    print(
                        f'Epoch: {self.epoch}/{self.num_epochs}',
                        f'Train accuracy: {train_acc}',
                        f'Validation accuracy: {val_acc}'
                    )

        return {
            'loss': self.loss_history,
            'train_accuracy': self.train_acc_history,
            'val_accuracy': self.val_acc_history
        }
