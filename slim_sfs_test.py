import torch
import re
import numpy as np
import pandas as pd
import glob
import pathlib
from tqdm import tqdm
import pickle
import h5py
from modules import *
from torch.autograd import Variable
from Sylvester import *
import pytorch_lightning as pl

from absl import app, flags

FLAGS = flags.FLAGS

# Data params
#flags.DEFINE_string('do_set', 0, 'Create Set Based Encoder')
#flags.DEFINE_string('pop_size', 500, 'Population Size')
flags.DEFINE_integer('Latent_Dim', 10, 'Number latent dimensions for selection coefficient')
flags.DEFINE_integer('Hidden_Dim', 64, 'Hidden Dims of layer')
flags.DEFINE_float('cuda', 1, "Use CUDA")
flags.DEFINE_string('dataset_path',"Slim_Run_Experiment_1_75.h5", "Path to dataset")
flags.DEFINE_string('create_dataset_path',"data_paths.txt", "Path to files that are convered to a dataset")


def create_data_set(the_path: str, slim_run_name: str="test.mixture", create_effect: bool=False, omega: float=0, sel_coef: float=0, dom_coef: float=0, mut_rate: float=0, pop_size: float=500):
    """Creates a dataset for the model to run on based on SLIM simulation runs, the slim runs are assumed to be 
        saved as test.mixture in @arg: slim_run_name

    Args:
        the_path (str): where simulations are saved
        slim_run_name (str): the name of each simulatoin
        sel_coef: selection coefficient or distribution of selection coefficients

    """    
    
    path = pathlib.Path(the_path) 
    np_files = path.rglob(slim_run_name) 
    np_dicts = [a.as_posix() for a in np_files] # Get file name and path
    data = np.empty(shape=(len(np_dicts)),dtype=object)
    dataset = np.empty(shape=(len(np_dicts),int((2*pop_size)),2))
    dataset[:,:,0] = np.arange(0,2*pop_size)
    min_sites = 1e8
    if '/' in the_path:
        save_path = (the_path.split('/')[-1])[:-4]
    else:
        save_path = the_path[:-4]
    if create_effect:
        save_path = save_path + '_with_effect'
    print("Save path: {}".format(save_path))
    # load data of frequencies and selection coefficeints
    for i, a_file in enumerate(tqdm(np_dicts)):
        temp = np.loadtxt(a_file,dtype='float',delimiter='\t',skiprows=1) # skips first row as it is the header (tag, position, selection, frequency)
        data[i]=temp # Just for storage later in case we need access
        # Now get SFS for that simulated population
        _,ind,counts = np.unique(temp[:,-1],return_index=True, return_counts=True) # Get unique frequncies
        vals = temp[ind,-1].astype('float')*2*pop_size
        vals = np.expand_dims(vals, 1)
        counts = np.expand_dims(counts, 1)
        dataset[i,vals.astype('int'),1]=counts # sfs[dataset #, frequency, 1] = #number of times allele occurs at frequency
        # if i > 5: # debugging statement
        #    break
        # find smallest size of segregating sites in simulations for creating effect sizes if needed
        if min_sites > temp.shape[0]:
            min_sites = temp.shape[0] 
    if create_effect:
        effect_sizes = np.random.normal(loc=0.0, scale=2*pop_size*np.abs(sel_coef)*omega,size=(dataset.shape[0], min_sites))
        hf = h5py.File("{}.h5".format(save_path), 'w')
        dset1 = hf.create_dataset("Frequencies",dataset.shape,dtype='f',data=dataset)
        dset1.attrs['sel_coef']=sel_coef
        dset1.attrs['mut_rate']=mut_rate
        dset1.attrs['dom_coef']=dom_coef
        dset1.attrs['pop_size']=pop_size
        dset2 = hf.create_dataset("Effects",effect_sizes.shape,dtype='f',data=effect_sizes)    
        dset2.attrs['sel_coef']=sel_coef
        dset2.attrs['omega']=omega
    else:
        hf = h5py.File("{}.h5".format(save_path), 'w')
        dset1 = hf.create_dataset("Frequencies",dataset.shape,dtype='f',data=dataset)
        dset1.attrs['sel_coef']=sel_coef
        dset1.attrs['mut_rate']=mut_rate
        dset1.attrs['dom_coef']=dom_coef
        dset1.attrs['pop_size']=pop_size    
    print("Finished processing and now saving")
    #np.save("{}.npy".format(save_path),data)
    print("Finished with {}".format(save_path))
    hf.close()

   
'''
class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))
'''

class integralDecoder(nn.Module):
    def __init__(self):
        super(integralDecoder, self).__init__()
    def forward(self, input, N, u):
        
        s = input
        points = torch.linspace(0.001, 1, 1000)
        probs = torch.zeros((N))
        scaled_pop_mut = 4*N*u
        f_i = lambda i: (1 - torch.exp(-2*N*s*(1-points))) * (torch.pow(points,i-1))*torch.pow((1-points),(N-i-1))
        for i in range(1,N):
            probs[i] = torch.trapz(values=f_i(i), x=points)
        return probs

class VAE(nn.Module):
    """
    The base VAE class containing linear encoder and decoder architecture.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, args):
        """
        Arguments:
            dim_in: an integer, input dimension.
            num_inds: an integer, number of inducing points.
            num_heads: an integer, number of heads.
            ln: boolean to normalize layers
        """
        super(VAE, self).__init__()
        

        # extract model settings from args
        self.z_size = args.z_size # Latent dimensions
        self.input_size = args.input_size # Number of sites
        #self.num_heads = args.num_heads # Multi-attention head
        #self.num_inds = args.num_inds
        self.dim_hidden = args.dim_hidden
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn = self.create_decoder()
        self.pop_size = args.pop_size
        self.mut_rate = args.mut_rate
        #self.ln = args.ln
        #self.num_outputs = args.num_outputs # number of samples
        self.set = args.set # If using permutation invariant layers (SET) 
        #self.q_z_nn_output_dim = 256

        # auxiliary
        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        # log-det-jacobian = 0 without flows
        self.log_det_j = Variable(self.FloatTensor(1).zero_())

    def create_encoder(self):
        """
        Helper function to create the elemental blocks for the encoder. Creates a set based encoder
        as the counts of sites and independant with each other.  
        """
        if self.set:
            q_z_nn = nn.Sequential(ISAB(self.input_size, self.dim_hidden, self.num_heads, self.num_inds, self.ln), 
                                            ISAB(self.dim_hidden, self.dim_hidden, self.num_heads, self.num_inds, self.ln))
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(256, self.z_size),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=7.))
        else:
            q_z_nn = nn.Sequential(nn.Linear(self.input_size, self.dim_hidden),nn.Linear(self.dim_hidden, self.dim_hidden),
                                   nn.Linear(self.dim_hidden, self.dim_hidden))  # 3 Layers 
            q_z_mean = nn.Linear(self.dim_hidden, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(self.dim_hidden, self.z_size),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=7.))
        return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        """
        Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
        """
        if args.set:
            p_x_nn= nn.Sequential(
                    PMA(self.dim_hidden, self.num_heads, self.num_outputs, ln=self.ln),
                    SAB(self.dim_hidden, self.dim_hidden, self.num_heads, ln=self.ln),
                    SAB(self.dim_hidden, self.dim_hidden, self.num_heads, ln=self.ln),
                    nn.Linear(self.dim_hidden, self.input_size))
            return p_x_nn
        else:
            return integralDecoder()
            # map the selection coefficients to the poisson random field F(i)
   
    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
            reparameterization trick.
        """

        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)

        return z

    def encode(self, x):
        """
        Encoder expects following data shapes as input: shape = (batch_size, num_channels, width, height)
        """

        h = self.q_z_nn(x)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)

        return mean, var

    def decode(self, z):
        """

        """
        x_mean = self.p_x_nn(torch.mean(z), self.pop_size, self.mut_rate)
        
        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """

        # mean and variance of z
        z_mu, z_var = self.encode(x)
        # sample z
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)

        return x_mean, z_mu, z_var, self.log_det_j, z
'''
class slimExperiment(pl.LightningModule):
    
    def __init__(self,
                 the_model: VAE,
                 params: dict, dataset: LightningDataModule) -> None:
        super(slimExperiment, self).__init__()

        self.model = the_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.dataset = dataset
'''       
    

def main(argv):

    if FLAGS.create_dataset_path:
        with open(FLAGS.create_dataset_path, "r") as f:  
            for info in f:
                a_path, sel_coef, omega, dom_coef, mut_rate, pop_size = info.split(',')
                print("Started to process data in {}".format(a_path))
                create_data_set(a_path, sel_coef=float(sel_coef), create_effect=True, omega=float(omega), dom_coef=float(dom_coef), mut_rate=float(mut_rate), pop_size=float(pop_size))
                print("Finished processing data in {}".format(a_path))
    else:
        print("Need a path to a dataset or to create one.")
        
                
    
    
    
                
if __name__=="__main__":
    app.run(main)


