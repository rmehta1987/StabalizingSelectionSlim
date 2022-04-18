import torch
import numpy as np
import pandas as pd
import glob
import pathlib
from tqdm import tqdm


def get_files(the_path: str):
    
    path = pathlib.Path(the_path) 
    np_files = path.rglob("test.mixture") 
    np_dicts = [a.as_posix() for a in np_files] # Get file name and path
    data = []
    
    # load data of frequencies and selection coefficeints
    for a_file in tqdm(np_dicts):
        lines = []
        for line in pd.read_csv(a_file, encoding='utf-8', header=None, chunksize=1): # read every line from the silmuation from SLIM
            lines.append(line.iloc[0,0])
        df = pd.DataFrame(lines)
        df.columns = df.iloc[0]
        df.drop(df.index[0],inplace=True)
        df.reset_index(inplace=True,drop=True)
        data.append(df)
        del df
    
    np.save('slim_run_df_1_5_e_6',data)
    print("Finished")

def main():
    get_files('/mnt/sda/home/ludeep/Desktop/PopGen/EvolutionaryGWAS/Slim_simulations/StabalizingSelectionSlim/Run_sel_coef_1_5')


if __name__=="__main__":
    main()


