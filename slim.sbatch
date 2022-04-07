#!/bin/bash
#SBATCH --job-name=mehta5_LD
#SBATCH --partition=bigmem2
#SBATCH --time=12:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=generate_RSID.put
#SBATCH --error=generate_RSID.err
#SBATCH --mem=32G

module purge
module load SLiM


echo "Running File"

#slim simulation


SLiM NegSelectionSlim.slim 
