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
from Sylvester import *
import pytorch_lightning as pl
from hdf5_dataset import GwasH5Dataset
import torch.optim as optim
import matplotlib.pyplot as plt

from absl import app, flags

FLAGS = flags.FLAGS

# Data params
#flags.DEFINE_string('do_set', 0, 'Create Set Based Encoder')
#flags.DEFINE_string('pop_size', 500, 'Population Size')
flags.DEFINE_integer('Latent_Dim', 10, 'Number latent dimensions for selection coefficient')
flags.DEFINE_integer('Hidden_Dim', 64, 'Hidden Dims of layer')
flags.DEFINE_integer('Epochs', 1, 'Number of Epochs')
flags.DEFINE_bool('cuda', True, "Use CUDA")
flags.DEFINE_string('path_to_binomial', "binomial_pop_size_500.npy", "Location of pre-created binomial coefficients")
flags.DEFINE_string('dataset_path',"Slim_Experiments_with_effects_h5/Slim_Run_Experiment_1_75_with_effect.h5", "Path to dataset if it does not need to be created")
flags.DEFINE_string('create_dataset_path', "", "A file that contains the path to files that are converted to a dataset")


def create_data_set(the_path: str, slim_run_name: str="test.mixture", create_effect: bool=False, omega: float=0, 
                    sel_coef: float=0, dom_coef: float=0, mut_rate: float=0, pop_size: float=500):
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
    min_sites = 1e8 # for creating a simulated true effect size 
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
        # data[i]=temp # Just for storage later in case we need access
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
    def forward(self, input, N, u, binomial_coef):
        """ Forward module for neural network to integrate the selection coefficient inferred and re-generate the site-frequency spectrum

        Args:
            input (_type_): the input latent dimensions from the last layer
            N (int): population size
            u (float): mutation rate

        Returns:
            _type_: _description_
        """        
        s = input
        if s.is_cuda:
            points = torch.linspace(0.001, 1, int(2*N)).cuda()
            probs = torch.zeros(int((2*N))).cuda()
        else:
            points = torch.linspace(0.001, 1, int(2*N))
            probs = torch.zeros(int((2*N)))  
        
        scaled_pop_mut = 4*N*u
        #f_i = lambda i: binomial_coef[i] * ((1 - torch.exp(-2*N*s*(1-points))) * (torch.pow(points,i-1))*torch.pow((1-points),(N-i-1)) * torch.pow((1 - torch.exp(-2*N*s)),-1))
        #f_i = lambda i: (1 - torch.exp(-2*N*s*(1-points))) * (torch.pow(points,i-1))*torch.pow((1-points),(N-i-1)) * torch.pow((1 - torch.exp(-2*N*s)),-1)
        for i in range(1,int(2*N)):
            #f_i_result = f_i(i)
            f_i_result = self._calc_mean(points, binomial_coef, N, s, i)
            int_result = torch.trapz(y=f_i_result, x=points)
            if int_result.isnan():
                print("i: {}, sel: {}".format(i,s.item()))
                
            probs[i] = scaled_pop_mut*int_result
        return probs
    
    def _calc_mean(self, points, binomal_coef, N, s, i):
        """_summary_

        Args:
            points (_type_): _description_
            binomal_coef (_type_): _description_
            N (_type_): _description_
            s (_type_): _description_
            i: 

        Returns:
            _type_: _description_
        """ 
        numerator_1 = 1 - torch.exp(-2*N*s*(1-points))  # 1 - exp(2ns(1-x))
        numerator_2 = (torch.pow(points,i-1))*torch.pow((1-points),(N-i-1))  # x^i(1-x)
        denominator = torch.pow((1 - torch.exp(-2*N*s)),-1)
        
        result = binomal_coef[i] * numerator_1 * numerator_2 * denominator
        if result.sum().isnan():
            print("result is nan")
            print("i: {}, sel: {}".format(i,s.item()))
        return result                                                       

class gwasVAE(nn.Module):
    """
    The base VAE class containing linear encoder and decoder architecture.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, latent_dims, pop_size, mut_rate, dim_hidden, use_set, use_cuda, coef):
        """
        Arguments:
            dim_in: an integer, input dimension.
            num_inds: an integer, number of inducing points.
            num_heads: an integer, number of heads.
            ln: boolean to normalize layers
        """
        super(gwasVAE, self).__init__()
        self.z_size = latent_dims # Latent dimensions
        self.input_size = int(pop_size*2) # Population size
        #self.num_heads = args.num_heads # Multi-attention head
        #self.num_inds = args.num_inds
        self.dim_hidden = dim_hidden
        self.set = use_set # If using permutation invariant layers (SET) 
        #self.q_z_nn_output_dim = 256
        
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn = self.create_decoder()
        
        #self.ln = args.ln
        #self.num_outputs = args.num_outputs # number of samples
        

        # auxiliary
        if use_cuda:
            self.FloatTensor = torch.cuda.FloatTensor
            self.binomial_coef = torch.from_numpy(coef).cuda()
        else:
            self.FloatTensor = torch.FloatTensor
            self.binomial_coef = torch.from_numpy(coef)

        # log-det-jacobian = 0 without flows
        self.log_det_j = self.FloatTensor(1).zero_()

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
            #q_z_mean = nn.Linear(self.dim_hidden, self.z_size)
            q_z_mean = nn.Sequential(
                nn.Linear(self.dim_hidden, self.z_size),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=7.))
            q_z_var = nn.Sequential(
                nn.Linear(self.dim_hidden, self.z_size),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=7.))
        return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        """
        Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
        """
        if self.set:
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
        x_mean = self.p_x_nn(torch.mean(z), self.pop_size, self.mut_rate, self.binomial_coef)
        
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

        return x_mean, z_mu, z_var, self.log_det_j, z, z
    
    def calculate_loss(self, recon_x, x, z_mu, z_var, z_0, z_k, ldj):
        """
        Computes the binary loss function while summing over batch dimension, not averaged!
        :param recon_x: shape: (batch_size, num_channels, pixel_width, pixel_height), bernoulli parameters p(x=1)
        :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
        :param z_mu: mean of z_0
        :param z_var: variance of z_0
        :param z_0: first stochastic latent variable
        :param z_k: last stochastic latent variable
        :param ldj: log det jacobian
        :param beta: beta for kl loss
        :return: loss, ce, kl
        """
        
        beta=1.

        #reconstruction_function = nn.BCELoss(reduction='sum')
        reconstruction_function = nn.PoissonNLLLoss(full=True, reduction='mean')

        batch_size = x.size(0)

        # - N E_q0 [ ln p(x|z_k) ]
        bce = reconstruction_function(recon_x, x)

        # ln p(z_k)  (not averaged)
        log_p_zk = self.log_normal_standard(z_k, dim=1)
        # ln q(z_0)  (not averaged)
        log_q_z0 = self.log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)
        # N E_q0[ ln q(z_0) - ln p(z_k) ]
        summed_logs = torch.sum(log_q_z0 - log_p_zk)

        # sum over batches
        summed_ldj = torch.sum(ldj)

        # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
        kl = (summed_logs - summed_ldj)
        loss = bce + beta * kl

        loss = loss / float(batch_size)
        bce = bce / float(batch_size)
        kl = kl / float(batch_size)

        return loss, bce, kl
    
    def log_normal_standard(self, x, average=False, reduce=True, dim=None):
        log_norm = -0.5 * x * x

        if reduce:
            if average:
                return torch.mean(log_norm, dim)
            else:
                return torch.sum(log_norm, dim)
        else:
            return log_norm
    
    def log_normal_diag(self, x, mean, log_var, average=False, reduce=True, dim=None):
        log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
        if reduce:
            if average:
                return torch.mean(log_norm, dim)
            else:
                return torch.sum(log_norm, dim)
        else:
            return log_norm
        
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
    elif FLAGS.dataset_path:
        num_data = 0
        h5_paths = [FLAGS.dataset_path] # need to change so input can be list of h5
        datasets = GwasH5Dataset(h5_paths)
        data_loader = torch.utils.data.DataLoader(datasets, num_workers=2, batch_size = 1, shuffle=False)
        device = torch.device("cuda" if FLAGS.cuda else "cpu")
        train_loss = np.zeros(len(data_loader))
        train_z= np.zeros(len(data_loader))
        print("Start Training")
        for i in range(FLAGS.Epochs):
            for step, the_data in enumerate(data_loader):
                if step == 0:
                    binom_coef = np.load(FLAGS.path_to_binomial)
                    #create a predefined binomial sample
                    # initalize model
                    #(self, latent_dims, pop_size, mut_rate, dim_hidden, use_set, use_cuda):
                    model = gwasVAE(FLAGS.Latent_Dim, pop_size=the_data['Labels']['pop_size'][0].item(), 
                                    mut_rate=the_data['Labels']['mut_rate'][0], dim_hidden=FLAGS.Hidden_Dim, use_set=False, use_cuda=FLAGS.cuda, coef = binom_coef).to(device)
                    true_sel_coef = the_data['Labels']['sel_coef'][0]
                    optimizer = optim.Adamax(model.parameters(), lr=0.0005,  eps=1.e-7)
                    if FLAGS.cuda:
                        #data = torch.mul(the_data['Freq'][:,:,1].cuda(), (1/2*the_data['Labels']['pop_size'][0]))
                        data = the_data['Freq'][:,:,1].cuda()
                    else:
                        #data = torch.mul(the_data['Freq'][:,:,1], (1/2*the_data['Labels']['pop_size'][0]))
                        data = the_data['Freq'][:,:,1]
                    
                    x_mean, z_mu, z_var, ldj, z0, zk = model(data)

                    #loss, rec, kl, bpd = model.calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj)
                    loss, rec, kl = model.calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj)
                    loss.backward()
                    train_loss[step] = loss.item()
                    train_z[step] = torch.mean(zk).detach().cpu().numpy()
                    optimizer.step()
                    num_data += 4 # hard-coded temporarily
                else:
                    optimizer.zero_grad()
                    x_mean, z_mu, z_var, ldj, z0, zk = model(data)
                    #loss, rec, kl, bpd = model.calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj)
                    loss, rec, kl = model.calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj)
                    loss.backward()
                    optimizer.step()
                    train_loss[step] = loss.item()
                    train_z[step] = torch.mean(zk).detach().cpu().numpy()
                    num_data += 4 # hard-coded temporarily
                
                if step % 10 == 0:
                
                    print('Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]  \tLoss: {:11.6f}\trec: {:11.6f}\tkl: {:11.6f}'.format(
                        i, num_data, len(data_loader.sampler), 100. * step / len(data_loader),
                        loss.item(), rec, kl))
                    #break
                    
    else:
        print("Need a path to a dataset or to create one.")
    
    plt.axhline(y=np.log(np.abs(true_sel_coef.numpy())), color='r', linestyle='-')
    plt.plot(np.log(np.abs(train_z)))
    plt.show()

    
                
if __name__=="__main__":
    app.run(main)


