# -*- python -*-
#
# license: GPLv3
# author: Attila Gabor (based on code from Avlant Nilsson)
# date: 2022-08-03
# version: 0.1



from typing import Callable
import pandas as pd
import numpy as np
import scipy as sp
import jax.numpy as jnp
import jax
from jax.experimental import sparse
import equinox as eqx
import matplotlib.pyplot as plt
import optax

import os


__all__ = ["bioNetwork"]


class bioNetwork():
    """ recurrent neural network model of genome scale signaling following Nilsson et al 2022 Nat. Com. """

    def __init__(self, networkFile, nodeAnnotationFile, banList=[], batchSize = 1,inputAmplitude=1, projectionAmplitude=1):
        """ initialize the model 
        
        
        Args:
            networkFile (str): path to the network model file
            nodeAnnotationFile (str): path to the node annotation file
            banList (list): list of nodes to be excluded from the model
            batchSize (int): number of samples in a batch

            
        """

        # assure batchSize is an integer
        if not isinstance(batchSize, int):
            raise ValueError("batchSize must be an integer")
        
        # set batch size
        self.batchSize = batchSize

        self.network = temp_network(networkFile=networkFile, banList=banList, nodeAnnotationFile = nodeAnnotationFile)
        self.trainingParameters = trainingParameters()

        # setup the neural net model
        self.model = BionetworkModel(self.network.networkList,
                self.network.nodeNames,
                self.network.modeOfAction,
                inputAmplitude,
                projectionAmplitude,
                self.network.inName,
                self.network.outName,
                self.trainingParameters)

    def loadParams(self,fileName):
        dictionary = dict(zip(self.network.nodeNames, list(range(len(self.network.nodeNames)))))
        data = pd.read_csv(fileName, delimiter = '\t')
        
        
        # check that the network file has the correct format
        if not set(['Source', 'Target', 'Type', 'Value']).issubset(set(data.columns)):
            raise ValueError("Parameter file must have columns 'Source', 'Target', 'Type', 'Value'")

        inputLayer_weights = np.zeros(self.model.layers[0].weights.shape)
        outputLayer_weights = np.zeros(self.model.layers[2].weights.shape)
        rNN_weights = np.zeros(self.model.layers[1].weights.shape)
        rNN_biases = np.zeros(self.model.layers[1].biases.shape)


        inputLookup = self.model.inputNodeOrder
        networkLookup = self.network.networkList
        projectionLookup = self.model.outputNodeOrder

        for i in range(data.shape[0]):
            curRow = data.iloc[i,:]
            source = dictionary[curRow['Source']]
            value = curRow['Value']
            if curRow['Type'] == 'Weight':
                target = dictionary[curRow['Target']]
                weightNr = np.argwhere(np.logical_and(networkLookup[1,:] == source, networkLookup[0,:] == target))
                rNN_weights[weightNr] = value
            elif curRow['Type'] == 'Bias':
                rNN_biases[source] = value
            elif curRow['Type'] == 'Projection':
                outputLayer_weights[projectionLookup == source] = value
            elif curRow['Type'] == 'Input':
                inputLayer_weights[inputLookup == source] = value
        
        # modifying pyTree: https://github.com/patrick-kidger/equinox/issues/94
        self.model.layers[0] = eqx.tree_at(lambda l: l.weights, self.model.layers[0], replace=jnp.asarray(inputLayer_weights))
        self.model.layers[1] = eqx.tree_at(lambda l: l.weights, self.model.layers[1], replace=jnp.asarray(rNN_weights))
        self.model.layers[1] = eqx.tree_at(lambda l: l.biases, self.model.layers[1], replace=jnp.asarray(rNN_biases))
        self.model.layers[2] = eqx.tree_at(lambda l: l.weights, self.model.layers[2], replace=jnp.asarray(outputLayer_weights))
        


class BionetworkModel(eqx.Module):
    layers: list
    inputNodeOrder: np.ndarray = eqx.static_field()
    outputNodeOrder: np.ndarray = eqx.static_field()

    def __init__(self, sparseNetwork, nodeList, modeOfAction, inputAmplitude, projectionAmplitude,inputNames, outputNames, bionetParams):
        super(BionetworkModel, self).__init__()

        nInputs = len(inputNames)
        nOutputs = len(outputNames)
        nNodes = len(nodeList)
        
        # determine what are the indices of the inputs and outputs in the nodeList

        dictionary = dict(zip(nodeList, list(range(len(nodeList)))))
        self.inputNodeOrder = np.array([dictionary[x] for x in inputNames])
        self.outputNodeOrder = np.array([dictionary[x] for x in outputNames])

        self.layers = [inputProjectionLayer(size_in=nInputs,
                                            size_out=nNodes,
                                            inOutIndices=self.inputNodeOrder,
                                            weight = inputAmplitude),
                       recurrentLayer(nStates=nNodes,
                                    modeOfAction=modeOfAction,
                                    networkList= sparseNetwork,
                                    iterations = bionetParams.iterations,
                                    leak=bionetParams.leak),
                       outputProjectionLayer(size_in=nNodes,
                                            size_out=nOutputs,
                                            inOutIndices=self.outputNodeOrder,
                                            weight = projectionAmplitude)
                       ]
        
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class inputProjectionLayer(eqx.Module):
    """Maps the inputs to the nodes."""

    weights: jnp.ndarray
    size_in: int = eqx.static_field()
    size_out: int = eqx.static_field()
    inOutIndices: np.ndarray = eqx.static_field()

    def __init__(
        self,
        size_in: int,
        size_out: int,
        inOutIndices: np.ndarray,
        weight: float = 1.0
    ):
        """**Arguments:**
        - `size_in`: The input size.
        - `size_out`: The output size.
        - `inOutIndices`: indices of output mapped to the inputs.
        - `weight`: The weight of the input projection.
        """
        super().__init__()
        
        self.size_in = size_in
        self.size_out = size_out
        self.inOutIndices = inOutIndices
        self.weights = weight*jnp.ones(size_in)

    def __call__(self, x):
        # TODO: 
        # check if there are faster ways, e.g. via jax.where or matrix multiplication (note that the matrix might be huge).
        # this might not work, because x becomes numpy in the next step.
        # probably we dont need to address batch dimensions, because we could just use jnp.vmap for that in the loss. 
        
        # FIX: this leads to jax.errors.TracerArrayConversionError
        # y = np.zeros([x.shape[0],  self.size_out])
        # y[:, self.inOutIndices] = self.weights * x

        y = jnp.zeros([x.shape[0],  self.size_out])
        return y.at[:, self.inOutIndices].set(self.weights * x)

        

class outputProjectionLayer(eqx.Module):
    """Maps the nodes to the output."""

    weights: jnp.ndarray
    size_in: int = eqx.static_field()
    size_out: int = eqx.static_field()
    inOutIndices: np.ndarray = eqx.static_field()

    def __init__(
        self,
        size_in: int,
        size_out: int,
        inOutIndices: np.ndarray,
        weight: float = 1.0
    ):
        """**Arguments:**
        - `size_in`: The input size.
        - `size_out`: The output size.
        - `inOutIndices`: indices of output mapped to the inputs.
        - `weight`: The weight of the input projection.
        """
        super().__init__()
        
        self.size_in = size_in
        self.size_out = size_out
        self.inOutIndices = inOutIndices
        self.weights = weight*jnp.ones(size_out)

    def __call__(self, x):
           # TODO:
           # probably we dont need to address batch dimensions, because we could just use jnp.vmap for that in the loss. 
        return self.weights * x[:, self.inOutIndices]

class recurrentLayer(eqx.Module):
    """Recurrent layer simulating the difference equation."""

    weights: jnp.ndarray
    biases: jnp.ndarray
    nStates: int = eqx.static_field()
    nReactions: int = eqx.static_field()
    modeOfAction: np.ndarray = eqx.static_field()
    networkList: np.ndarray = eqx.static_field()
    iterations: int = eqx.static_field()
    leak: float = eqx.static_field()
    A: sparse.BCOO = eqx.static_field()
    activation: Callable = eqx.static_field()

    def __init__(
        self,
        nStates: int,
        modeOfAction: np.ndarray,
        networkList: list,
        iterations: int = 150,
        leak:float = 0.01
    ):
        """**Arguments:**
        - `nStates`: The number of states.
        
        - `modeOfAction`: vector of signs of the interactions.
        - `networkList`: adjacency matrix of the network.
        """
        super().__init__()
        
        self.nStates = nStates
        self.nReactions = modeOfAction.shape[1]
        self.modeOfAction = modeOfAction
        self.networkList = networkList
        
        self.iterations = iterations
        self.leak = leak

        weights, biases = self.initializeWeights()
        self.weights = weights
        self.biases = biases
        
        # https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html
        scipyA =  sp.sparse.csr_matrix((weights, networkList), shape=(nStates, nStates))
        self.A = sparse.BCOO.from_scipy_sparse(scipyA)
        # Activition function
        # TODO: add the others
        self.activation = self.get_activation_function()

    def __call__(self, x):
        
        self.A.data = self.weights
        
        # x comes in as a row vector, but we need a column vector.
        bIn = x.T + self.biases

        # xhat is also a column vector
        xhat = jnp.zeros(bIn.shape)

        xhatBefore = xhat.copy()

        for i in range(self.iterations):
            #if i>40: #normally takes around 40 iterations to reach steady state
            #    if i>41:
            #        if jnp.sum(jnp.abs(xhat-xhatBefore))<1e-6:
            #            break            
            #    xhatBefore = xhat.copy()            
            # this should be a matrix times a column vector: 
            xhat = self.A @ xhat + bIn
            # activation should preserve the dimensionality of the input
            xhat = self.activation(xhat)

        # we should return a row vector to be consistent with the input and other layers. 
        return xhat.T
    
 

        

    def initializeWeights(self):
        """ initialize the weights of the recurrent layer """
        
        weights = 0.1 + 0.1 * np.random.uniform(size=(self.nReactions,))
        weights[self.modeOfAction[1,:]] = -weights[self.modeOfAction[1,:]]
        weights = jnp.asarray(weights)

        # the bias is a column vector. 
        bias = 1e-3 * np.ones((self.nStates, 1))
        
        # This is from Avlant, but not sure about its function yet
        for i in range(self.nStates):
            affectedIn = self.networkList[0,:] == i
            if jnp.any(affectedIn):
                if jnp.all(weights[affectedIn]<0):
                    bias[i] = 1 #only affected by inhibition, relies on bias for signal
        bias = jnp.asarray(bias)
        return weights, bias

    def get_activation_function(self):
        def MMLactivation(x):
            
            x = jnp.where(x < 0, x * self.leak, x)
            x = jnp.where(x > 0.5, 1 - 0.25/x, x) #Pyhton will display division by zero warning since it evaluates both before selecting
            return x
        return MMLactivation


class temp_network():
    """ temporary class to read network model from file """

    def __init__(self, networkFile, banList = [], nodeAnnotationFile = None) -> None:
        """ initialize the network model from files """

        self._importNetwork(networkFile, banList)
        self._importNodeAnnotation(nodeAnnotationFile)

        self.nStates = len(self.nodeNames)
        self.nReactions = len(self.networkList[0])
        self.nInputs = len(self.inName)
        self.nOutputs = len(self.outName)
        

    def _importNodeAnnotation(self, nodeAnnotationFile):
        # assure networkAnnotationFile is an existing file:
        if not os.path.isfile(nodeAnnotationFile):
            raise ValueError("networkAnnotationFile must be an existing file")

        annotation = pd.read_csv(nodeAnnotationFile, sep='\t', low_memory=False)
        inName = annotation.loc[annotation.ligand, 'name'].values
        outName = annotation.loc[annotation.TF, 'name'].values
        
        self._annotation = annotation
        self.inName = inName
        self.outName = outName


    def _importNetwork(self, networkFile, banList= []):

         # assure networkModelFile is an existing file:
        if not os.path.isfile(networkFile):
            raise ValueError("networkFile must be an existing file")

        # read network model from file
        net = pd.read_csv(networkFile, sep='\t', index_col=False)

        # check that the network file has the correct format
        if not set(['source', 'target', 'stimulation', 'inhibition']).issubset(set(net.columns)):
            raise ValueError("network file must have columns 'source', 'target', 'weight'")

        # filter out banned nodes
        net = net[~ net["source"].isin(banList)]
        net = net[~ net["target"].isin(banList)]

        sources = list(net["source"])
        targets = list(net["target"])
        stimulation = np.array(net["stimulation"])
        inhibition = np.array(net["inhibition"])
        modeOfAction = 0.1 * np.ones(len(sources))
        modeOfAction[stimulation==1] = 1
        modeOfAction[inhibition==1] = -1

        networkList, nodeNames, weights = self.makeNetworkList(sources, targets, modeOfAction)  #0 == Target 1 == Source due to np sparse matrix structure
        modeOfAction = np.array([[weights==1],[weights==-1]]).squeeze()

        self.net_df = net
        self.networkList = networkList
        self.nodeNames = nodeNames
        self.modeOfAction = modeOfAction
    
    def plot(self):
        """ plot the adjacency matrix of the network with spy."""
        plt.spy(self.A)
        plt.xticks(range(len(self.nodeNames)), self.nodeNames, size='small', rotation=90)
        plt.yticks(range(len(self.nodeNames)), self.nodeNames, size='small')
        plt.xlabel("Target")
        # plot xtick labels to bottom
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.ylabel("Source")
        plt.show()

    def makeNetworkList(self, sources, targets, weights):
        nodeNames = list(np.unique(sources + targets))
        dictionary = dict(zip(nodeNames, list(range(len(nodeNames)))))
        sourceNr = np.array([dictionary[x] for x in sources]) #colums
        targetNr = np.array([dictionary[x] for x in targets]) #rows
        size = len(nodeNames)
        A = sp.sparse.csr_matrix((weights, (sourceNr, targetNr)), shape=(size, size))
        self.A = A

        networkList = sp.sparse.find(A)
        weights = networkList[2]
        networkList = np.array((networkList[1], networkList[0]))  #0 == Target 1 == Source due to numpy sparse matrix structure
        return networkList, nodeNames, weights

class trainingParameters():

    def __init__(self,iterations = 150, clipping=1, leak=0.01,targetPrecision = 1.0e-4, spectralTarget=None) -> None:
      
        #set defaults
        self.iterations = iterations
        self.clipping = clipping
        self.leak = leak
        self.targetPrecision = targetPrecision
        
        if spectralTarget is None:
            self.spectralTarget = np.exp(np.log(self.targetPrecision)/self.iterations)

        