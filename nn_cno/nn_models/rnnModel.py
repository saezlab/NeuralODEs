# -*- python -*-
#
# license: GPLv3
# author: Attila Gabor (based on code from Avlant Nilsson)
# date: 2022-08-03
# version: 0.1



import pandas as pd
import numpy as np
import scipy as sp
import jax.numpy as jnp
import jax
import equinox
import matplotlib.pyplot as plt
import optax

import os


__all__ = ["bioNetwork"]


class bioNetwork():
    """ recurrent neural network model of genome scale signaling following Nilsson et al 2022 Nat. Com. """

    def __init__(self, networkFile, nodeAnnotationFile, banList=[], batchSize = 1):
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
        
    
class temp_network():
    """ temporary class to read network model from file """

    def __init__(self, networkFile, banList = [], nodeAnnotationFile = None) -> None:
        """ initialize the network model from files """

        self._importNetwork(networkFile, banList)
        self._importNodeAnnotation(nodeAnnotationFile)

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

