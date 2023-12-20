import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import wandb
import warnings
import numpy as np
from scipy.stats import t as t_dist
import logging

from models.networks import LinearRegres, NonLinearRegres
from models.metrics import BayesianWeightedLoss

class Model(nn.Module):
    """ Generic object with methods common to different networks. 
    Shoud be usable for other frameworks. 
    """
    def __init__(self, network, optimizer, criterion, data, args):
        super().__init__()
        self.network = network.to(args.device)
        self.optimizer= optimizer
        self.criterion = criterion
        self.data = data 
        self.args = args

        # Data split between train and validation
        if args.validation:
            try:
                self.train_data, self.val_data = self.__split()
                self.val_step = True
            except:
                self.train_data = data
                self.val_step = False
                warnings.warn('The split percentatge does not allow for validations steps')
        else:
            self.train_data = data
            self.val_step = False

    def __split(self):
        """ Prepare data to feed the network.
        Inputs:
            None. The data is a list containing (input, target) with each 'domain' being a tensor
            of size (N, Features).
        Output:
            train_data: tuple with train timepoints
            val_data: tuple with validation timepoints 
        """
        N = self.data[0].shape[0]
        tr_N = (100-self.args.split)*0.01
        tr_indices, val_indices = train_test_split(range(N),train_size=tr_N)
        train_data, val_data = list(), list()
        for p in range(len(self.data)):
            train_data.append(torch.index_select(self.data[p], 0, torch.tensor(tr_indices)))
            val_data.append(torch.index_select(self.data[p], 0, torch.tensor(val_indices)))
        return tuple(train_data), tuple(val_data) 
        
    def __epoch(self, loader, backprop):
        epoch_loss, num_batches = [0, 0]
        for input_batch, target_batch in zip(loader[0], loader[1]):
            input_batch = input_batch.double().to(self.args.device)
            target_batch = target_batch.double().to(self.args.device)
            
            prediction = self.network(input_batch)
            loss = self.criterion(prediction, target_batch)

            if backprop: # If loss backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        return epoch_loss/num_batches

    def train(self):
        """ Trains the model. If specified and if possible also doing validation."""
        if self.args.wandb: 
            wandb.init(project="Plasticity-Regressor", entity="joansano")
            wandb.config.update(self.args)
        tr_loader, val_loader = list(), list()
        for ep in range(1, self.args.epochs+1):
            # Loading training and validation data
            for domain in range(len(self.train_data)):
                tr_loader.append(DataLoader(self.train_data[domain], batch_size=self.args.batch))
                if self.val_step:
                    val_loader.append(DataLoader(self.val_data[domain], batch_size=self.args.batch))
            # Training   
            with torch.enable_grad():
                loss_tr = self.__epoch(tr_loader, backprop=True)
                if self.args.wandb: 
                    wandb.log({"Batch Training Loss": loss_tr}, step=ep)
                else:
                    logging.info("Epoch {}/{}: Training loss: {}".format(ep, self.args.epochs, loss_tr))
            # Validation
            if self.val_step and (ep%self.args.val_freq==0):
                with torch.no_grad():
                    loss_val = self.__epoch(val_loader, backprop=False)
                    if self.args.wandb: 
                        wandb.log({"Batch validation Loss": loss_val}, step=ep)
                    else:
                        logging.info("Validation loss: {}".format(loss_val))
            # Live updating
            if self.args.wandb: 
                wandb.watch(self.network)

    def test(self, x, prior=None):
        """ Generates a prediction of a given batch """
        with torch.no_grad():
            output = self.network(x.double().to(self.args.device))
            if prior is None:
                return output
            else:
                prior = prior.to(self.args.device)
                posterior = 0*output 
                for t in range(x.shape[0]):
                    posterior[t] = torch.mul(output[t], prior.to(self.args.device))
                return posterior

def return_specs(args, prior=None):
    """
    Returns the object necessary to build the model
    Inputs:
        args: argparser containing the input arguments
        prior: (optional) Necessary to build the bayesian weighted loss function
    Returns:
        regres: torch network to train (python object)
        loss: torch loss function used to train the network (python object)
        sgd: torch object used to train train the network (python object)
    """
    if args.regressor == 'linear': 
        regres = LinearRegres(args.rois)
    elif args.regressor == 'nonlinear':
        regres = NonLinearRegres(args.rois)
    else:
        raise ValueError("Regressor not implemented")

    loss = BayesianWeightedLoss(prior, type=args.loss)

    if args.optimizer == 'sgd': 
        optimizer = optim.SGD(regres.parameters(), lr=args.learning_rate) # Leave default parameters
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(regres.parameters(), lr=args.learning_rate) # Leave default parameters
    else:
        raise ValueError("Optimizer not implemented")

    return regres, loss, optimizer

def grubbs_test(x, alpha=0.05):
    """
    Uses two-tailed Grubb's test to check if the maximum value of the data is an outlier. 
    Refs: [1] https://support.minitab.com/en-us/minitab/20/help-and-how-to/statistics/basic-statistics/how-to/outlier-test/methods-and-formulas/methods-and-formulas/#p-values-for-grubbs-test-statistic
          [2] https://www.originlab.com/doc/origin-help/grubbs-test-dialog
          [3] https://en.wikipedia.org/wiki/Grubbs%27s_test
    Inputs:
        x: data (numpy array)
        alpha: (optional) significance for the test
    Returns:
        h: if there is an outlier (bool)
        p: p-value of the test (float)
        arg_outlier: if h is True, index of the outlier (int)
    """
    # Two-tailed Grubbs statistic
    n, mean_x, sd_x = len(x), np.mean(x), np.std(x)
    g_calculated = max(abs(x-mean_x))/sd_x
    # Two-tailed Grubbs critical value
    t_value = t_dist.ppf(1 - alpha / (2 * n), n - 2)
    g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
    # p-value: Ref [1]
    t_stat = np.sqrt(n*(n-2)*(g_calculated**2))/np.sqrt((n-1)**2 - n*(g_calculated**2))
    p = 2*n*(1 - t_dist.cdf(t_stat, n - 2)) 
    # Outlier
    if g_critical > g_calculated:
        h = False
        arg_outlier = None
    else:
        h = True
        arg_outlier = np.argmax(abs(x-mean_x))
    return h, p, arg_outlier

def f_test(x, y):
    """
    Two-sided F-test for variance of two samples
    """
    from scipy.stats import f as F
    f = np.var(x)/np.var(y)
    nun = len(x)-1
    dun = len(y)-1
    p_value = 1-F.cdf(f, nun, dun)
    return f, p_value

def to_array(dict, dtype=np.float64):
    return np.array(list(dict.values()), dtype=dtype)

if __name__ == '__main__':
    pass