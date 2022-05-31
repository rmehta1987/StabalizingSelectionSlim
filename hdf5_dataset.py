import torch
from torch.utils.data import Dataset
import h5py


class GwasH5Dataset(Dataset):
    """Represents an abstract HDF5 dataset for GWAS data.  

    Args:
        h5_paths: List of files containing the dataset (one or multiple HDF5 files), each hdf5 file has 
            one dataset for allle frequencies (site frequency spectrum) and another for effects.  Each dataset also
            has attributes of selection, mutation coefficient, population size, and/or dominance coefficient.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=1).
        transform: PyTorch transform to apply to every data instance (default=None).
    """    
    def __init__(self, h5_paths, load_data, data_cache_size=1, transform=None):
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self.indices = [None]*len(h5_paths)
        self.dataset_length = 0

        for a, archive in enumerate(self.archives):
            the_keys = list(archive.keys())
            assert archive[the_keys[0]].shape[0] == archive[the_keys[1]].shape[0], "Effect sizes and frequency datasets should be of the same size (same number of samples"
            self.dataset_length = self.dataset_length + archive[the_keys[0]].shape[0]
            self.indices[a] = (a,the_keys[0],the_keys[1])

        self._archives = None

    @property
    def archives(self):
        if self._archives is None: # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        ''' index is a batch of indicies for the dataset itself'''
        a = self.indices[index]
        data_freq = [self.archives[a[ind,1]] for ind in index]
        data_effect = [self.archives[a[ind,1]] for ind in index]
        freq = torch.from_numpy(np.stack(data_freq),axis=0)
        effect = torch.from_numpy(np.stack(data_effect),axis=0)
        #labels = dict(dataset.attrs)

        return {"Freq": freq, "Effect": effect}

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)
    
h5_paths = 'Slim_Experiments_with_effects_h5'
datasets = GwasH5Dataset(h5_paths, False, False)
loader = torch.utils.data.DataLoader(datasets, num_workers=0)
tepm = datasets.__getitem__(torch.tensor((1,2,3)))
#batch = next(iter(loader))
