
import h5py 
import torch
from torch.utils.data import DataLoader, random_split, Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_paths, limit=-1):
        self.limit = limit
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self.indices = {}
        idx = 0
        print(self.archives)
        for a, archive in enumerate(self.archives):
            for i in archive.keys():
                self.indices[idx] = (a, i)
                idx += 1

        self._archives = None

    @property
    def archives(self):
        if self._archives is None: # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        a, i = self.indices[index]
        archive = self.archives[a]
        dataset = archive[f"{i}"]
        data = torch.from_numpy(dataset[:])
        labels = dict(dataset.attrs)

        return {"data": data, "labels": labels}

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)
    
    
h5_paths = ['Slim_Run_Experiment_1_75.h5', 'Slim_Run_Experiment_1_75.h5']
loader = torch.utils.data.DataLoader(H5Dataset(h5_paths), num_workers=2)
batch = next(iter(loader))
