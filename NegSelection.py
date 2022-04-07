import subprocess
import msprime
import pyslim
import tskit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Load SLiM .trees
file = sys.argv[1]
pi = float(sys.argv[2])
nsnp = int(sys.argv[3])

ts = pyslim.load("{}.trees".format(file))
ts.recapitate(recombination_rate=1e-8, Ne=7310, random_seed=1)
overlaid = pyslim.SlimTreeSequence(msprime.mutate(ts, rate=1.65e-8*(1.0-pi), random_seed=1, keep=True))

# extract SNP info
af = []
snpID = []
mutID = []
pos = []
coef = []
for tree in overlaid.trees():
   for site in tree.sites():
      for mutation in site.mutations:
         af.append(tree.get_num_leaves(mutation.node) / tree.get_sample_size())
         snpID.append(site.id)
         mutID.append(mutation.id)
         pos.append(round(site.position))
         if (site.ancestral_state == ''):
            coef.append(mutation.metadata[0].selection_coeff)
         else:
            coef.append(0)

data = {'SnpID':snpID, 'Pos':pos, 'MutID':mutID, 'SelCoef':coef, 'Freq':af}
df = pd.DataFrame(data)

# select common SNPs
df_com = df.loc[(df['Freq'] > 0.01) & (df['Freq'] < 0.99), :]
df_com_qtl = df_com.loc[df_com['SelCoef'] !=0, :]
if nsnp > len(df_com.index[df_com['SelCoef'] == 0]): nsnp = len(df_com.index[df_com['SelCoef'] == 0])
com_mrk = np.random.choice(df_com.index[df_com['SelCoef'] == 0].tolist(), nsnp, replace=False)
df_com_mrk = df_com.loc[com_mrk,:]

selected_loci = df_com_qtl['SnpID'].tolist() + df_com_mrk['SnpID'].tolist()
loci_to_remove = list(set(df['SnpID']) - set(selected_loci))

df_save = df_com.loc[df_com['SnpID'].isin(selected_loci),:].drop(columns='MutID').sort_values(by='Pos')
df_save['SnpID'] = np.arange(1,df_save.shape[0]+1)
df_save.to_csv("{}.pyslim.selected_snps".format(file), sep=' ', index=False)

#Obtain individuals who are alive at the end of the simulation
inds_alive = ts.individuals_alive_at(0)
Populations = [np.where(ts.individual_populations[inds_alive] == k)[0] for k in range(1, ts.num_populations)]
sample_names = []
for k in Populations[0]:
   sample_names.append('ind_{}'.format(k+1))

# Write VCF
ts_res = overlaid.delete_sites(loci_to_remove)
#ts_res.dump("{}_selected.trees".format(file))
ts_res.get_num_sites()
with open("{}.vcf".format(file), "w") as vcf_file:
    ts_res.write_vcf(output=vcf_file,
                     individuals=Populations[0],
                     individual_names=sample_names,
                     contig_id='chr1')

