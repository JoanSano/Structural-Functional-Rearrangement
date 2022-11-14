from matplotlib.pyplot import savefig
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
import torch

from .paths import get_info, get_subjects, check_path

class GraphFromTensor:
    def __init__(self, graph, name, base_dir='', rois=170):
        """
        WARNING: If you pass a flat graph be sure that the size is [1,Features] !!
        """
        self.features = int(rois*(rois-1)/2)
        self.conns = np.array(graph)  # Connections
        self.graph_size = self.conns.shape
        if self.graph_size[0] == 1: # Flat graph
            self.conns = self.conns[0, :self.features].reshape((1,self.features))
        else: # 2D graph
            self.conns = self.conns[:rois, :rois]
        self.name = name
        if base_dir == '':
            self.dir = os.getcwd()+'/'
        else:
            self.dir = base_dir
        check_path(self.dir)
        self.originals = self.conns
        self.graph = self.dir + self.name + '.csv'

    def __revert(self):
        """
        Reorders AAL3 regions by hemispheres.
        Odd indices correspond to Left hemisphere regions.
        Even indices correspond to rigth hemisphere regions.
        Stores a dictionary with the reodering of indices.
        """

        odd_odd = self.conns[::2, ::2]
        odd_even = self.conns[::2, 1::2]
        first = np.vstack((odd_odd, odd_even))
        even_odd = self.conns[1::2, ::2]
        even_even= self.conns[1::2, 1::2]
        second = np.vstack((even_odd, even_even))
        self.conns = np.hstack((first,second))

        # To map actual labels with original ones
        labels = np.array([x for x in range(0, self.graph_size[0])])
        left = np.array([x for x in range(1, self.graph_size[0], 2)])
        rigth = np.array([x for x in range(0, self.graph_size[0], 2)])
        self.hemis = dict(zip(labels, np.concatenate((left, rigth), axis=0)))

    def __take_log(self):
        """
        Takes the natural logarithm of the connections. Enhances visualisation of the matrix.
        """

        self.conns = np.log1p(self.conns)

    def __plot_graph(self, save=True, show=False, fig_size=(20,15), bar_label='Connection Strength'):
        """
        Plot a graph. It assumes that the adjancency matrix is a csv file.
        """

        plt.figure(figsize=fig_size)
        plt.imshow(self.conns)
        cbar = plt.colorbar()
        cbar.set_label(bar_label, rotation=270)
        plt.tight_layout()
        if save:
            plt.savefig(self.dir+self.name+'.png')      
        if show:
            plt.show()     
        plt.close('all')

    def process_graph(self, log=True, reshuffle=True, save=True, show=False, fig_size=(20,15), bar_label='Connection Strength'):
        """
        Applies default operations to the graph to work with it.
        """

        self.processed = True # The object has been processed
        if self.conns.shape[0] <= 1:
            raise ValueError("You are trying to process a flat graph. Can't do it. Unflatten your graph and set it to default.")
        else:
            if log:
                self.__take_log()
            if reshuffle:
                self.__revert()
            self.__plot_graph(save=save, show=show, fig_size=fig_size, bar_label=bar_label)
    
    def get_connections(self, ini=False):
        """ 
        Get values of the connectivites of either the original or processed graph. 
        """
        if not ini:
            return self.conns 
        else:
            return self.originals
    
    def flatten_graph(self, save=True):
        """
        Flatten the lower triangular adjancency matrix of the graph. 
        The flattened graph becomes available after applying this method.
        """
        
        x = self.conns.shape[0] # Dimensions of the graph 
        if x <= 1:
            raise ValueError("Dimension of the graph is 1 (or lower). You can't flattened an already flattened graph")
        else:
            dims = int(self.conns.shape[0]*(self.conns.shape[0]-1)/2)
            self.flat_conns = np.zeros((1,dims))
            k = 0
            for i in range(x):
                for j in range(i):
                    self.flat_conns[0,k] = self.conns[i,j]
                    k += 1
            if save:
                np.savetxt(self.dir+self.name+'_flatCM.csv', self.flat_conns, delimiter=',')
            return self.flat_conns

    def unflatten_graph(self, to_default=False, save_flat=True):
        """
        Unflatten a graph and transform it to a square symmetric matrix. 
        The unflattened graph becomes available after applying this method.
        to_default: bool - The unflattened matrix becomes the default graph and replaces 
            the initial flat graph. As a checkpoint, the flattened graph is saved in the directory(default: False)
        """

        x = self.conns.shape[0] # First dimension of the flattened graph 
        flat_dim = self.conns.shape[1]
        if x > 1:
            raise ValueError("Dimension of the graph greater than 1. You can't unflattened an already unflattened graph")
        else:
            dims = int(1+np.sqrt(1+8*flat_dim)/2) # Dimensions of the squared graph
            self.unflat_conns = np.zeros((dims,dims))
            k = 0
            for i in range(dims):
                for j in range(i):
                    self.unflat_conns[i, j] = self.conns[0, k]
                    self.unflat_conns[j, i] = self.conns[0, k]
                    k += 1
            if to_default:
                if save_flat:
                    # We save the flat graph with another name
                    np.savetxt(self.dir+self.name+'_flatCM.csv', self.conns, delimiter=',')
                # We re-initialize the graph with the unflattened graph and both the same name and directory
                self.__init__(self.unflat_conns, self.name, self.dir)
            return self.unflat_conns

    def save(self):
        np.savetxt(self.graph, self.conns, delimiter=',')

class GraphFromCSV:
    """
    It creates an object with different properties from a csv. Everything related to it, will be
        saved with the provided 'name' followed by the proper extensions.
    """

    def __init__(self, graph, name, base_dir='/', rois=170):
        self.features = int(rois*(rois-1)/2)
        self.graph = graph  # Graph file name
        self.conns = pd.read_csv(graph, delimiter=',', header=None).values # Graph connections
        self.graph_size = self.conns.shape
        if self.graph_size[0] == 1: # Flat graph
            self.conns = self.conns[0, :self.features].reshape((1,self.features))
        else: # 2D graph
            self.conns = self.conns[:rois, :rois]
        self.name = name
        if base_dir == '/':
            self.dir = os.getcwd()+'/'
        else:
            self.dir = base_dir
        self.originals = self.conns
        # TODO: Add check path for base_dir!

    def __revert(self):
        """
        Reorders AAL3 regions by hemispheres.
        Odd indices correspond to Left hemisphere regions.
        Even indices correspond to rigth hemisphere regions.
        Stores a dictionary with the reodering of indices.
        """

        odd_odd = self.conns[::2, ::2]
        odd_even = self.conns[::2, 1::2]
        first = np.vstack((odd_odd, odd_even))
        even_odd = self.conns[1::2, ::2]
        even_even= self.conns[1::2, 1::2]
        second = np.vstack((even_odd, even_even))
        self.conns = np.hstack((first,second))

        # To map actual labels with original ones
        labels = np.array([x for x in range(0, self.graph_size[0])])
        left = np.array([x for x in range(1, self.graph_size[0], 2)])
        rigth = np.array([x for x in range(0, self.graph_size[0], 2)])
        self.hemis = dict(zip(labels, np.concatenate((left, rigth), axis=0)))

    def __take_log(self):
        """
        Takes the natural logarithm of the connections. Enhances visualisation of the matrix.
        """

        self.conns = np.log1p(self.conns)

    def __plot_graph(self, save=True, show=False, fig_size=(20,15), bar_label='Connection Strength'):
        """
        Plot a graph. It assumes that the adjancency matrix is a csv file.
        """

        plt.figure(figsize=fig_size)
        plt.imshow(self.conns)
        cbar = plt.colorbar()
        cbar.set_label(bar_label, rotation=270)
        plt.tight_layout()
        if save:
            plt.savefig(self.dir+self.name+'.png')      
        if show:
            plt.show()     

    def process_graph(self, log=True, reshuffle=True, save=True, show=False, fig_size=(20,15), bar_label='Connection Strength'):
        """
        Applies default operations to the graph to work with it.
        """

        self.processed = True # The object has been processed
        if self.conns.shape[0] <= 1:
            raise ValueError("You are trying to process a flat graph. Can't do it. Unflatten your graph and set it to default.")
        else:
            if log:
                self.__take_log()
            if reshuffle:
                self.__revert()
            self.__plot_graph(save=True, show=False, fig_size=(20,15), bar_label=bar_label)
    
    def get_connections(self, ini=False):
        """ 
        Get values of the connevtivites of either the original or processed graph. 
        """
        if not ini:
            return self.conns 
        else:
            return self.originals
    
    def flatten_graph(self, save=True):
        """
        Flatten the lower triangular adjancency matrix of the graph. 
        The flattened graph becomes available after applying this method.
        """
        
        x = self.conns.shape[0] # Dimensions of the graph 
        if x <= 1:
            raise ValueError("Dimension of the graph is 1 (or lower). You can't flattened an already flattened graph")
        else:
            dims = int(self.conns.shape[0]*(self.conns.shape[0]-1)/2)
            self.flat_conns = np.zeros((1,dims))
            k = 0
            for i in range(x):
                for j in range(i):
                    self.flat_conns[0,k] = self.conns[i,j]
                    k += 1
            if save:
                np.savetxt(self.dir+self.name+'_flatCM.csv', self.flat_conns, delimiter=',')
            return self.flat_conns

    def unflatten_graph(self, to_default=False, save_flat=True):
        """
        Unflatten a graph and transform it to a square symmetric matrix. 
        The unflattened graph becomes available after applying this method.
        to_default: bool - The unflattened matrix becomes the default graph and replaces 
            the initial flat graph. As a checkpoint, the flattened graph is saved in the directory(default: False)
        """

        x = self.conns.shape[0] # First dimension of the flattened graph 
        flat_dim = self.conns.shape[1]
        if x > 1:
            raise ValueError("Dimension of the graph greater than 1. You can't unflattened an already unflattened graph")
        else:
            dims = int(1+np.sqrt(1+8*flat_dim)/2) # Dimensions of the squared graph
            self.unflat_conns = np.zeros((dims,dims))
            k = 0
            for i in range(dims):
                for j in range(i):
                    self.unflat_conns[i, j] = self.conns[0, k]
                    self.unflat_conns[j, i] = self.conns[0, k]
                    k += 1
            if to_default:
                if save_flat:
                    # We save the flat graph with another name
                    np.savetxt(self.dir+self.name+'_flatCM.csv', self.conns, delimiter=',')
                # We replace the original file with the unflattend graph
                np.savetxt(self.graph, self.unflat_conns, delimiter=',')
                # We re-initialize the graph with the unflattened graph and both the same name and directory
                self.__init__(self.graph, self.name, self.dir)
            return self.unflat_conns

    def save(self):
        np.savetxt(self.graph, self.conns, delimiter=',')

def create_anat_prior(CONTROL, out_path, threshold=.2, save=False):
    """ Creates a prior distribution of connections between rois """
    control = np.array(CONTROL, dtype=float)
    prior = torch.mean(torch.tensor(np.where(control>threshold, 1, 0), dtype=float), dim=0)
    mean_connections = torch.mean(torch.tensor(control), dim=0)
    if save:
        np.savetxt(out_path+'prior.csv', prior.view(1, -1), delimiter=',')
        np.savetxt(out_path+'mean_conn.csv', mean_connections.view(1, -1), delimiter=',')
    return prior, mean_connections

def load_anat_prior(path, rois=170):
    """ Loads an available anatomical prior distributions of connections between rois"""
    prior = torch.tensor(np.log1p(GraphFromCSV(path+'prior.csv', 'ignore', rois=rois).get_connections()))
    mean_connections = torch.tensor(np.log1p(GraphFromCSV(path+'mean_conn.csv', 'ignore', rois=rois).get_connections()))
    return prior, mean_connections

if __name__ == '__main__':
    pass