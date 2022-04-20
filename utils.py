from importlib.resources import as_file
from xml import dom
import numpy as np
import pandas as pd
import json
import argparse
from pytz import country_names
from tqdm import tqdm
import pdb
import h5py

parser = argparse.ArgumentParser(description='Utility functions for simulations from SLIM')
parser.add_argument('--path',action='store', type=str, metavar="path_npy", 
                    help='Path to numpy file to convert', default='')
args = parser.parse_args()

def convert_numpy_save(path_npy: str, create_sfs: bool):
    # assumes format is of numpy, and the numpy file was a converted dataframe (M x 4) into 
    # a numpy array of shape M x 1 where M is the number of rows, but each row is an object type
    # so we split the row into 4 columns again and then save it as a pickle 

    np_file = np.load(path_npy, allow_pickle=True)
    if '/' in path_npy:
        save_path = (path_npy.split('/')[-1])[:-4]
    else:
        save_path = path_npy[:-4]
    print("Saving pickled data to {}".format(save_path))
    list_of_df = []
    
    # Using pandas seems to be inefficient
    #columns = ['tag','position','selection_coefficient','freq']
    
    
    for i, afile in enumerate(tqdm(np_file)):
        to_df = np.array([x[0].split() for x in afile],dtype=object)
        df = pd.DataFrame(to_df,columns=columns)
        list_of_df.append(df)
        del df
        #if i > 3:
        #    break
    #pdb.set_trace()
    #df_to_export = pd.DataFrame(list_of_df)
    #json_out = df_to_export.to_json()
    np.save(save_path,list_of_df)
    #with open("{}.txt".format(save_path),"wb") as outfile:
    #    outfile.write(json_out)

def create_SFS(path_to_file: str, sel_coef: float=0, dom_coef: float=0, mut_rate: float=0, pop_size: int=1000):
    # Converts data to a SFS but drops singletons, usually the file that needs to be loaded is a 
    # a list of dataframes or matrices
    # assumes pop_size is 1000
    if '/' in path_to_file:
        save_path = (path_to_file.split('/')[-1])[:-4]
    else:
        save_path = path_to_file[:-4]
    hf = h5py.File('{}.h5'.format(save_path), 'w')

    np_file = np.load(path_to_file, allow_pickle=True)
    dataset = np.empty(shape=(np_file.shape[0],2*pop_size,2))
    dataset[:,:,0] = np.arange(0,2*pop_size)
    min_freq = float(1/(2*pop_size))
    for i,a_file in enumerate(np_file):
        #if i > 10:
        #    break
        if type(a_file) is pd.core.frame.DataFrame:
            a_file.iloc[:,-1] = a_file.iloc[:,-1].astype('float') # convert column to float
            a_file_filterd = (a_file[a_file.iloc[:,-1]>min_freq]).iloc[:,-1].value_counts() # histogram without singletons
            temp_dict = a_file_filterd.to_dict()
        if type(a_file) is np.ndarray:
            the_file= np.array([x[0].split() for x in a_file],dtype=object) # due to simulation, each line is an object, so need to split to get info
            del a_file
            a_file_filterd = the_file[np.greater(the_file[:,-1].astype(float),min_freq)]
            _,ind,counts = np.unique(a_file_filterd[:,-1],return_index=True, return_counts=True)
            vals = a_file_filterd[ind,-1].astype('float')*2*pop_size
            vals = np.expand_dims(vals, 1)
            counts = np.expand_dims(counts, 1)
            dataset[i,vals.astype('int'),1]=counts
            dataset[i,1,1]=the_file[np.less_equal(the_file[:,-1].astype(float),min_freq)].shape[0]
             # Save dataset as an hp5y file
    dset1 = hf.create_dataset("Simulation1",dataset.shape,dtype='f',data=dataset)
    dset1.attrs['sel_coef']=sel_coef
    dset1.attrs['mut_rate']=mut_rate
    dset1.attrs['dom_coef']=dom_coef
    dset1.attrs['pop_size']=pop_size
    hf.close()
        

if args.path:
    print(args.path)
    #convert_numpy_save_to_json(args.path)
    create_SFS('slim_run_df_2_5_e_6.npy')
    print(" Finished conversion")

