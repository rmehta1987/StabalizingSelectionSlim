import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class GwasH5Dataset(Dataset):
    """Represents an abstract HDF5 dataset for GWAS data.  

    Args:
        h5_paths: List of files containing the dataset (one or multiple HDF5 files), each hdf5 file has 
            one dataset for allle frequencies (site frequency spectrum) and another for effects.  Each dataset also
            has attributes of selection, mutation coefficient, population size, and/or dominance coefficient.
        transform: PyTorch transform to apply to every data instance (default=None).
    """    
    def __init__(self, h5_paths, load_data, transform=None):
        self.h5_paths = h5_paths
        self.indices = [None]*len(h5_paths)
        self.dataset_length = 0
        if len(h5_paths) > 1:
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
            for a, archive in enumerate(self.archives):
                the_keys = list(archive.keys())
                the_keys.sort()
                assert archive[the_keys[0]].shape[0] == archive[the_keys[1]].shape[0], "Effect sizes and frequency datasets should be of the same size (same number of samples"
                self.dataset_length = self.dataset_length + archive[the_keys[0]].shape[0]
                self.indices[a] = (a,the_keys[0],the_keys[1])
        else:
            self._archives = h5py.File(h5_paths[0], "r")
            the_keys = list(self._archives.keys())  # Sort Effect size is index 0, and frequencies are index 1
            the_keys.sort()
            self.indices[0] = (0, the_keys[0], the_keys[1])  # index of dataset, effect size key, frequency key
            self.dataset_length = self._archives[the_keys[0]].shape[0]

        

        self._archives = None

    @property
    def archives(self):
        if len(h5_paths) > 1:
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        else:
            self._archives = h5py.File(self.h5_paths[0], "r")
        return self._archives

    def __getitem__(self, index):
        ''' index is one item'''
        labels = {}
        if isinstance(self.archives, list): # if it is a list
            a = self.indices[index]
            data_freq = self.archives[a[0]][a[2]] # Get frequncies or SFS
            data_effect = self.archives[a[0]][a[1]] # Get effect sizes or betas
            freq = torch.from_numpy(np.stack(data_freq,axis=0))
            effect = torch.from_numpy(np.stack(data_effect,axis=0))
            
            #labels = dict(dataset.attrs)
        else:
            a = self.indices[0]
            data_freq = self.archives[a[2]][index] # Get frequncies or SFS
            data_effect = self.archives[a[1]][index] # Get effect sizes or betas
            freq = torch.from_numpy(data_freq)
            effect = torch.from_numpy(data_effect)
            labels['sel_coef'] = self.archives[a[2]].attrs['sel_coef']
            labels['mut_rate'] = self.archives[a[2]].attrs['mut_rate']
            labels['dom_coef'] = self.archives[a[2]].attrs['dom_coef']
            labels['pop_size'] = self.archives[a[2]].attrs['pop_size']
            labels['omega'] = self.archives[a[1]].attrs['omega']
 
        return {"Freq": freq, "Effect": effect, "Labels": labels}

    def __len__(self):
        return self.dataset_length
    
h5_paths = ['Slim_Experiments_with_effects_h5/Slim_Run_Experiment_1_75_with_effect.h5']
datasets = GwasH5Dataset(h5_paths, False, False)
loader = torch.utils.data.DataLoader(datasets, num_workers=2, batch_size = 4, shuffle=False )
#tepm = datasets.__getitem__(torch.tensor((1,2,3)))
batch = next(iter(loader))
print("stuff")
