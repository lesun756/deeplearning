import numpy as np


class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        #########################################################################
        # TODO: 20 points                                                       #
        # - Generate a random NN weight matrix to use to compute loss.          #
        # - By using dictionary (self.params) to store value                    #
        #   with standard normal distribution and Standard deviation = 0.0001.  #
        #########################################################################


        self.params['w1'] = 0.0001 * np.random.randn(inputDim, hiddenDim)
        self.params['w2'] = 0.0001 * np.random.randn(hiddenDim, outputDim)
        self.params['b1'] = np.zeros(hiddenDim)
        self.params['b2'] = np.zeros(outputDim)

        pass
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None
        #############################################################################
        # TODO: 40 points                                                           #
        # - Compute the NN loss and store to loss variable.                         #
        # - Compute gradient for each parameter and store to grads variable.        #
        # - Use Leaky RELU Activation at hidden and output neurons                  #
        # - Use Softmax loss
        # Note:                                                                     #
        # - Use L2 regularization                                                   #
        # Hint:                                                                     #
        # - Do forward pass and calculate loss value                                #
        # - Do backward pass and calculate derivatives for each weight and bias     #
        #############################################################################

        w1 = self.params['w1']
        w2 = self.params['w2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        # number of samples
        N = x.shape[0]

        h = x.dot(w1) + b1
        # Leaky ReLU
        h = np.maximum(0.01*h, h)

        s = h.dot(w2) + b2
        # Leaky ReLU
        s = np.maximum(0.01*s, s)

        s = s - np.max(s, axis=1, keepdims=True)
        sum_s = np.sum(np.exp(s), axis=1, keepdims=True)
        prob = np.exp(s) / sum_s

        # compute softmax loss
        loss_i = -np.log(prob[np.arange(N), y])
        loss = np.sum(loss_i)
        # loss with L2 regularization
        loss += reg * (np.sum(w1 * w1) + np.sum(w2 * w2))

        # compute gradients
        ind = np.zeros(prob.shape)
        ind[np.arange(N), y] = 1
        ds = prob - ind
        dw2 = (1/N) * (h.T).dot(ds)
        db2 = (1/N) * np.sum(ds, axis=0)
        dreg2 = 2 * reg * w2
        dw2 = dw2 + dreg2

        dh = ds.dot(w2.T)
        dh_ = (h > 0) * dh
        dw1 = (1/N) * (x.T).dot(dh_)
        db1 = (1/N) * np.sum(dh_, axis=0)
        dreg1 = 2 * reg * w1
        dw1 = dw1 + dreg1

        grads['w1'] = dw1
        grads['w2'] = dw2
        grads['b1'] = db1
        grads['b2'] = db2




        pass
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, grads

    def train (self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iterations):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################

            num_train = np.random.choice(x.shape[0], batchSize)
            xBatch = x[num_train]
            yBatch = y[num_train]

            # get loss and gradients
            loss, grads = self.calLoss(xBatch, yBatch, reg)
            # update weight
            self.params['w1'] = self.params['w1'] - lr * grads['w1']
            self.params['w2'] = self.params['w2'] - lr * grads['w2']
            self.params['b1'] = self.params['b1'] - lr * grads['b1']
            self.params['b2'] = self.params['b2'] - lr * grads['b2']

            lossHistory.append(loss)


            pass
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # Decay learning rate
            lr *= decay
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Store the predict output in yPred                                    #
        ###########################################################################

        w1 = self.params['w1']
        w2 = self.params['w2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        # score matrix
        h = np.maximum(0, x.dot(w1)) + b1
        s = h.dot(w2) + b2

        # predict output
        yPred = np.argmax(s, axis=1)


        pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################

        yPred = self.predict(x)
        acc = np.mean(y == yPred) * 100

        pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



