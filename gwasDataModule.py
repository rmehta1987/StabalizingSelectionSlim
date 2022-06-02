from typing import Callable
from numpy import full
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset
import os
from torchvision import transforms
import h5py
import torch

class H5Dataset(Dataset):
    def __init__(self, h5_paths, limit=-1):
        self.limit = limit
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self.indices = {}
        idx = 0
        for a, archive in enumerate(self.archives):
            for i in range(len(archive)):
                self.indices[idx] = (a, i)
                idx += 1

        self._archives = None


    '''
    The decorator approach for creating properties requires defining a first method using the 
    public name for the underlying managed attribute, which is .archives in this case. 
    This method should implement the getter logic. In the below function archives.
    '''
    @property
    def archives(self):
        """Lazy loading of hdf5 dataset, https://vict0rs.ch/2021/06/15/pytorch-h5/

        Returns:
            self.archives a list of hdf5 paths
        """        
        if self._archives is None: # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        a, i = self.indices[index]
        archive = self.archives[a]
        dataset = archive[f"{i}"]
        data = torch.from_numpy(dataset[:])
        labels = dict(dataset.attrs)

        return {"data": data, " ": labels}

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)


class gwasDataModule(LightningDataModule):
    
    def __init__(self, params: dict, transforms: Callable = None):
        super().__init__()
        self.params = params
        if transforms is None:
            print("No transformations of the dataset")
        elif transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = self.data_transforms()
        
        
    def prepare_data(self) -> None:
        if self.params['dataset'] == 'simulation':
            if os.path.exists(self.params['data_path'][0]):
                print("Will load dataset from {}".format(self.params['data_path']))                
            else:
                raise ValueError('Undefined dataset type or dataset folder not found')
        
    def setup(self, stage=None) -> None:
        if self.params['dataset'] == 'simulation':
            #pdb.set_trace()
            # self.h5_file = h5py.File(self.params['data_path'], "r")
            subset_train = 0.8 # change to getting from dict of params
            subset_test = 1 - subset_train
            full_dataset =  H5Dataset(self.params['data_path'])
            self.train_dataset, self.test.dataset = random_split(full_dataset,[int(len(full_dataset)*subset_train),int(len(full_dataset)*subset_test)])  
            self.dims=[len(self.train_dataset),len(self.test_dataset)]
        else:
            raise ValueError('Undefined dataset type')
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size= self.params['batch_size'], num_workers=4)

    #def val_dataloader(self):
    #    return DataLoader(self.val_dataset, batch_size= self.params['batch_size'], num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size= self.params['batch_size'], num_workers=4)
       
    def data_transforms(self):
        
        if self.params['dataset'] == 'celeba':
            
            SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
            SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))
        
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange, SetScale])
       
        else:
            raise ValueError('No transforms available for this dataset type')

        return transform
    
    def length_of_train_dataset(self):
        return len(self.train_dataset)

    #def length_of_val_dataset(self):
    #    return len(self.val_dataset)

    def length_of_test_dataset(self):
        return len(self.test_dataset)

    def teardown(self, stage: str = None) -> None:
        pass
    
    
h5_paths = ['Slim_Run_Experiment_1_75.h5', 'Slim_Run_Experiment_1_75.h5']
params={}
params['data_path']=h5_paths
params['dataset']='simulation'
params['batch_size']=2

loader = gwasDataModule(params=params)
loader.prepare_data()
loader.setup()

#batch = next(iter(loader))
